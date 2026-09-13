"""The Sidekick: a worker agent wrapped in an evaluator loop.

The shape is the **evaluator-optimizer** pattern. You give the Sidekick a task
*and a definition of success*. A worker agent — a single LangChain
`create_agent` with a browser, a filesystem, search and Wikipedia — attempts it.
A separate evaluator model then judges the result against your stated criteria
and returns a structured verdict, which decides what happens next:

    success_criteria_met  →  done, return the answer
    user_input_needed     →  stop and put the question to the user
    neither               →  send the worker back with the feedback (up to MAX_ATTEMPTS)

Around the worker sits a middleware stack that is the actual production story of
this project — planning, PII redaction, a model-call budget, human approval
gates, and tool-error tolerance. See `_middleware` below.
"""

from __future__ import annotations

import uuid

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    TodoListMiddleware,
)
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from sidekick.config import (
    DEFAULT_SUCCESS_CRITERIA,
    EVALUATOR_MODEL,
    MAX_ATTEMPTS,
    MODEL_CALL_LIMIT,
    SANDBOX,
    WORKER_MODEL,
)
from sidekick.prompts import evaluator_prompt, retry_prompt, worker_prompt
from sidekick.tools import get_all_tools

#: Tools the worker must get explicit human approval for before running. Both
#: reach outside the sandbox — one messages the user's phone, the other asks them
#: to act — so neither should fire without the human seeing it first.
APPROVAL_REQUIRED = {"send_push_notification": True, "request_human_help": True}


class EvaluatorOutput(BaseModel):
    """The evaluator's verdict, as structured output rather than prose.

    Three fields, and the third is what stops the loop from being stupid: an
    agent that asked a clarifying question has not failed, and retrying it with
    "try harder" would just produce the same question again.
    """

    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if the assistant has a question, needs clarification, or is stuck and needs the user"
    )


class TolerateToolErrors(AgentMiddleware):
    """Hand a tool failure back to the model as a message instead of crashing the run.

    Tools that touch the outside world fail routinely — a page times out, a
    selector moves, a site 503s. Without this, one flaky browser call ends the
    whole turn. With it, the model reads "that tool call failed, try another
    approach" and usually does exactly that.
    """

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as error:
            return ToolMessage(
                content=f"That tool call failed: {error}. Try another approach.",
                tool_call_id=request.tool_call["id"],
            )


def _middleware() -> list:
    """The production guardrail stack, in the order it is applied.

    Each entry answers a different question about deploying an agent that has a
    browser and a filesystem:

    * `TolerateToolErrors`   — what happens when a tool breaks? (resilience)
    * `TodoListMiddleware`   — what is it doing right now? (planning, and the UI
      reads the same todos, so the plan the user sees is the agent's real one)
    * `PIIMiddleware`        — what if personal data passes through? (safety;
      the credit-card rule also applies to *tool results*, so a card number
      scraped off a web page is caught on the way in, not just on the way out)
    * `ModelCallLimitMiddleware` — what stops it costing $400? (cost)
    * `HumanInTheLoopMiddleware` — what stops it doing something irreversible?
      (control)
    """
    return [
        TolerateToolErrors(),
        TodoListMiddleware(),
        PIIMiddleware("email"),
        PIIMiddleware("credit_card", apply_to_tool_results=True),
        ModelCallLimitMiddleware(run_limit=MODEL_CALL_LIMIT),
        HumanInTheLoopMiddleware(interrupt_on=APPROVAL_REQUIRED),
    ]


class Sidekick:
    """One Sidekick session: a worker, an evaluator, and the conversation between them."""

    def __init__(self) -> None:
        #: Doubles as the LangGraph thread id, so the checkpointer keeps this
        #: session's history separate from any other.
        self.session_id = str(uuid.uuid4())
        self.memory = InMemorySaver()
        self.tools: list = []
        self.sessions = None
        self.worker = None
        self.evaluator = None

        self.task = ""
        self.success_criteria = ""
        self.attempts = 0
        self.paused = False
        self.pending_actions = 0
        #: The worker's own plan, surfaced live in the UI by the todo middleware.
        self.todos: list = []

    # ------------------------------------------------------------------ setup

    async def setup(self) -> None:
        """Bring up the MCP servers and build both agents. Slow: it starts a browser."""
        SANDBOX.mkdir(parents=True, exist_ok=True)
        self.tools, self.sessions = await get_all_tools(SANDBOX)
        self.worker = create_agent(
            model=WORKER_MODEL,
            tools=self.tools,
            system_prompt=worker_prompt(),
            middleware=_middleware(),
            checkpointer=self.memory,
        )
        self.evaluator = ChatOpenAI(model=EVALUATOR_MODEL).with_structured_output(EvaluatorOutput)

    def cleanup(self) -> None:
        """Shut down the MCP servers; the browser window closes."""
        if self.sessions:
            self.sessions.stop()

    # ------------------------------------------------------------- evaluation

    async def evaluate(self, last_reply: str, tools_used: list[str]) -> EvaluatorOutput:
        """Judge the worker's answer against the user's success criteria."""
        return await self.evaluator.ainvoke(
            evaluator_prompt(self.task, self.success_criteria, last_reply, tools_used)
        )

    # ---------------------------------------------------------------- the turn

    async def run_turn(self, message: str, success_criteria: str, history: list) -> list:
        """One turn: the worker attempts the task, the evaluator checks it, repeat.

        Returns as soon as the worker pauses for approval, with `paused` set;
        `resume()` continues the *same* turn from the checkpoint.
        """
        self.task = message
        self.success_criteria = success_criteria or DEFAULT_SUCCESS_CRITERIA
        self.attempts = 0
        self.todos = []
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f"{message}\n\nThe success criteria for this task are: {self.success_criteria}",
                }
            ]
        }
        return await self._advance(payload, history + [{"role": "user", "content": message}])

    async def resume(self, history: list) -> list:
        """Approve the paused actions and continue the turn.

        The resume payload approves exactly the number of actions the interrupt
        raised. LangGraph replays from the checkpoint, so the worker picks up
        mid-run with its full context — including whatever the human just did in
        the browser window.
        """
        payload = Command(resume={"decisions": [{"type": "approve"}] * self.pending_actions})
        return await self._advance(payload, history)

    async def _advance(self, payload, history: list) -> list:
        """Drive the worker/evaluator loop until it finishes, pauses, or gives up."""
        config = {"configurable": {"thread_id": self.session_id}}
        while True:
            result = None
            async for result in self.worker.astream(payload, config=config, stream_mode="values"):
                # The todo middleware updates this as the worker replans; the UI
                # timer reads it, so the plan the user watches is the live one.
                self.todos = result.get("todos", self.todos)

            if "__interrupt__" in result:
                actions = result["__interrupt__"][0].value["action_requests"]
                self.paused = True
                self.pending_actions = len(actions)
                described = "\n".join(action["description"] for action in actions)
                return history + [
                    {"role": "assistant", "content": f"Waiting for your approval:\n{described}"}
                ]

            self.paused = False
            reply = result["messages"][-1].content
            tools_used = [
                call["name"]
                for message in result["messages"]
                for call in (getattr(message, "tool_calls", None) or [])
            ]
            self.attempts += 1

            verdict = await self.evaluate(reply, tools_used)
            done = verdict.success_criteria_met or verdict.user_input_needed
            if done or self.attempts >= MAX_ATTEMPTS:
                return history + [
                    {"role": "assistant", "content": reply},
                    {"role": "assistant", "content": f"Evaluator: {verdict.feedback}"},
                ]

            payload = {"messages": [{"role": "user", "content": retry_prompt(verdict.feedback)}]}

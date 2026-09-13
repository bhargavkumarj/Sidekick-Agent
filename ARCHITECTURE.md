# Architecture

## 1. Two agents, one loop

The Sidekick is the **evaluator-optimizer** pattern, made concrete.

```
run_turn(task, success_criteria)
  │
  ▼
┌──────────────────────────────────────────┐
│ _advance(payload)                        │  ← also the resume path
│   loop:                                  │
│     stream the worker to completion      │
│       ├─ collect todos as they change    │
│       └─ collect every tool call made    │
│                                          │
│     if __interrupt__ in result:          │
│         paused = True; return  ───────────────► UI shows Approve
│                                          │        resume() re-enters here
│     verdict = evaluator(task, criteria,  │
│                         reply, tools)    │
│                                          │
│     if met or needs_user or attempts≥3:  │
│         return the answer + the feedback │
│     payload = retry_prompt(feedback)     │
└──────────────────────────────────────────┘
```

Two details in that diagram carry most of the design.

**`_advance` is shared by both entry points.** `run_turn` starts a turn;
`resume` continues one. Both call `_advance`, so approval is not a special mode
— it is a pause in the middle of the normal loop, and the evaluation still
happens afterwards exactly as it would have.

**The evaluator sees the tool trace.** `tools_used` is collected by walking every
message in the stream result and pulling out `tool_calls`. That list goes into
the judging prompt alongside the reply. Judging prose alone rewards a confident
writer; the tool calls are the only evidence of what actually happened.

---

## 2. Why the evaluator is a separate model call

It would be cheaper to ask the worker "did you meet the criteria?" at the end.
That does not work, for the same reason you do not ask a student to grade their
own exam: the worker has spent its whole context justifying its approach, and
asking it to now find fault produces agreement.

A separate call with a fresh context, given only (task, criteria, reply, tools),
has no such investment. Making its output **structured** rather than prose is the
other half:

```python
class EvaluatorOutput(BaseModel):
    feedback: str
    success_criteria_met: bool
    user_input_needed: bool
```

Three fields, and the third is the one that keeps the loop sensible. Without
`user_input_needed`, an agent that asked *"which airport did you mean?"* looks
like a failure, gets retried with "please address the feedback", and asks the
same question again — three times, then gives up. Distinguishing *failed* from
*blocked on the user* is what makes the retry loop worth having.

`WORKER_MODEL` and `EVALUATOR_MODEL` are separate settings on purpose. Running a
cheap worker under a strong judge is a legitimate configuration, and so is the
reverse.

---

## 3. The middleware stack

This is the part of the project that separates a demo from something deployable.
Order matters: middleware wraps outward-in, so the first entry is the outermost.

```python
[
    TolerateToolErrors(),                                  # resilience
    TodoListMiddleware(),                                  # planning
    PIIMiddleware("email"),                                # safety
    PIIMiddleware("credit_card", apply_to_tool_results=True),
    ModelCallLimitMiddleware(run_limit=30),                # cost
    HumanInTheLoopMiddleware(interrupt_on=APPROVAL_REQUIRED),  # control
]
```

**`TolerateToolErrors`** is the only custom one, and it is eleven lines:

```python
async def awrap_tool_call(self, request, handler):
    try:
        return await handler(request)
    except Exception as error:
        return ToolMessage(content=f"That tool call failed: {error}. Try another approach.",
                           tool_call_id=request.tool_call["id"])
```

Without it, one timed-out page load ends the turn. With it, the model reads the
failure and routes around it — which is what you would want a person to do. The
`tool_call_id` matters: the model is waiting for a result for *that* call, and an
unmatched response corrupts the message history.

**`PIIMiddleware("credit_card", apply_to_tool_results=True)`** is worth pointing
at specifically. The default guards what the *model produces*. `apply_to_tool_results`
also guards what comes *back from a tool* — so a card number sitting on a web
page the browser just read is redacted before it enters the context at all. An
agent with a browser is an agent that can read anything on the internet, and the
inbound direction is the one people forget.

**`HumanInTheLoopMiddleware`** interrupts the LangGraph run before the named
tools execute. The interrupt surfaces in the stream result as `__interrupt__`,
carrying `action_requests` — the descriptions the UI shows. Approving is a
`Command(resume={"decisions": [{"type": "approve"}] * n})`, and LangGraph replays
from the checkpoint.

---

## 4. Human-in-the-loop, twice

The project uses the interrupt mechanism for two genuinely different things, and
the distinction is worth being able to draw.

**Approval** — `send_push_notification`. The action is outward-facing and
irreversible; the human is a gate. Nothing about the world changes while paused.

**Delegation** — `request_human_help`. The agent has hit something only a person
can do: a login, a captcha, 2FA. The tool's implementation is a no-op that
returns `"The user says it is done."` Its entire function is to trigger the
interrupt, so the agent's instructions reach the human, who acts *in the browser
window that is already open in front of them*, and then approves.

That second one only works because the browser is persistent across tool calls —
which is what `mcp_sessions.py` exists for.

---

## 5. Keeping MCP sessions alive

The single trickiest piece of code here, and the one most worth understanding.

An MCP stdio transport must be opened and closed **from the same asyncio task**.
The obvious implementation satisfies that constraint and destroys the product:

```python
# WRONG for this use case
async with client.session("playwright") as session:
    tools = await load_mcp_tools(session)
    ...one tool call...
```

That starts and stops the browser around every call. The agent navigates to a
page, then takes a snapshot — of a fresh, blank browser.

So `McpSessions` puts one background task in charge of the sessions' whole
lifetime:

```python
async def _run(self):
    async with AsyncExitStack() as stack:
        for name in self.connections:
            session = await stack.enter_async_context(client.session(name))
            self.tools += await load_mcp_tools(session, server_name=name)
        self._ready.set()      # publish the tools
        await self._stop.wait()  # hold here for the whole conversation
    # the stack unwinds on the same task that entered it
```

Everything else just uses `self.tools`. The browser keeps its cookies, its page
and its scroll position for the entire session.

`start()` has one further subtlety:

```python
await asyncio.wait([ready, self._task], return_when=asyncio.FIRST_COMPLETED)
if self._task.done():
    self._task.result()  # re-raise the real startup error
```

Waiting on readiness alone would hang forever if a server failed to start, since
`_ready` would never be set. Racing readiness against the task finishing means a
missing `npx` surfaces as the actual exception instead of a deadlock.

---

## 6. The sandbox is a server argument

```python
"filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", str(sandbox)],
}
```

The filesystem MCP server takes its allowed directories as command-line
arguments. Passing exactly one is the entire sandbox: the agent physically cannot
read or write outside it, because the server refuses, and no prompt injection can
argue with a process that was never given the capability.

This is the cleanest illustration in any of these projects of **capability
scoping over instruction**. The prompt does not say "only write to the sandbox".
It does not need to.

---

## 7. The UI has one non-obvious rule

The plan panel is updated **only** by a 1-second `gr.Timer`, never as an output
of the message handler.

If a long-running event declared the panel as an output, Gradio would mark it
pending for the duration of the run, and no live update would render. You would
watch a frozen placeholder for two minutes and then see the finished plan
appear — which is exactly the opposite of the point.

Giving the timer sole ownership means the panel is repainted while the worker
runs, from `self.todos`, which the todo middleware updates as the agent replans.
The plan the user watches is the agent's real plan, changing live.

---

## 8. What I would change

* **Persistent checkpointing.** `InMemorySaver` means a restart loses the
  conversation. `langgraph-checkpoint-sqlite` is already a dependency; switching
  is a one-line change and would let a paused approval survive a crash.
* **Streaming the worker's tokens.** Right now the chat updates once per attempt.
  The `astream` is already there; the UI just does not consume it token by token.
* **A structured evaluator on the tool trace, not just the text.** The evaluator
  gets tool *names*; it does not see arguments or results. Giving it the full
  trace would let it catch "you searched, but for the wrong thing".
* **Approval policy instead of an approval list.** `APPROVAL_REQUIRED` is a
  hardcoded dict. A predicate over the tool call — approve any `browser_click` on
  a checkout page, say — would generalise.
* **Cost accounting.** `ModelCallLimitMiddleware` counts calls, not tokens. Calls
  are a poor proxy for spend once context grows.

"""Tests for the Sidekick's control flow and guardrails.

The worker is an LLM with a browser, so it cannot be unit tested. What *can* be
tested — and is worth testing, because it is where the safety properties live —
is the loop around it: does the evaluator's verdict route correctly, does the
retry stop, is every sensitive tool actually behind an approval gate.

The worker and evaluator are replaced with fakes, so nothing here calls a model
or starts a browser.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
os.environ["SIDEKICK_SANDBOX"] = tempfile.mkdtemp(prefix="sidekick-tests-")

from sidekick.agent import APPROVAL_REQUIRED, EvaluatorOutput, Sidekick, _middleware  # noqa: E402
from sidekick.config import DEFAULT_SUCCESS_CRITERIA, MAX_ATTEMPTS  # noqa: E402
from sidekick.prompts import evaluator_prompt  # noqa: E402


class FakeMessage:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class FakeWorker:
    """Yields a scripted stream result per invocation, recording what it was sent."""

    def __init__(self, results: list[dict]) -> None:
        self.results = list(results)
        self.payloads: list[Any] = []

    async def astream(self, payload, config=None, stream_mode=None):
        self.payloads.append(payload)
        yield self.results.pop(0)


class FakeEvaluator:
    """Returns scripted verdicts and records the prompts it was given."""

    def __init__(self, verdicts: list[EvaluatorOutput]) -> None:
        self.verdicts = list(verdicts)
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> EvaluatorOutput:
        self.prompts.append(prompt)
        return self.verdicts.pop(0)


def make_sidekick(results: list[dict], verdicts: list[EvaluatorOutput]) -> Sidekick:
    """A Sidekick with both models faked out. No setup(), so no browser starts."""
    sidekick = Sidekick()
    sidekick.worker = FakeWorker(results)
    sidekick.evaluator = FakeEvaluator(verdicts)
    return sidekick


def reply(text: str, tools: list[str] = ()) -> dict:
    calls = [{"name": name, "id": f"call-{i}"} for i, name in enumerate(tools)]
    return {"messages": [FakeMessage(text, calls)], "todos": []}


def verdict(met: bool = False, needs_user: bool = False, feedback: str = "feedback") -> EvaluatorOutput:
    return EvaluatorOutput(feedback=feedback, success_criteria_met=met, user_input_needed=needs_user)


# ------------------------------------------------------------------ routing


async def test_a_met_criterion_ends_the_turn_immediately() -> None:
    sidekick = make_sidekick([reply("Done.")], [verdict(met=True)])
    history = await sidekick.run_turn("Do the thing", "It is done", [])

    assert sidekick.attempts == 1
    assert history[1]["content"] == "Done."
    assert history[2]["content"].startswith("Evaluator:")


async def test_a_failed_criterion_sends_the_worker_back_with_the_feedback() -> None:
    sidekick = make_sidekick(
        [reply("Half done."), reply("Fully done.")],
        [verdict(met=False, feedback="You missed the second half"), verdict(met=True)],
    )
    await sidekick.run_turn("Do the thing", "Both halves", [])

    assert sidekick.attempts == 2
    retry = sidekick.worker.payloads[1]["messages"][0]["content"]
    assert "You missed the second half" in retry


async def test_a_question_stops_the_loop_rather_than_retrying() -> None:
    """An agent that asked for clarification has not failed; retrying it with
    'try harder' would just produce the same question again."""
    sidekick = make_sidekick([reply("Which airport?")], [verdict(met=False, needs_user=True)])
    await sidekick.run_turn("Book a flight", "Booked", [])

    assert sidekick.attempts == 1
    assert sidekick.worker.payloads == sidekick.worker.payloads[:1]


async def test_retries_stop_at_max_attempts() -> None:
    """The evaluator-optimizer loop is bounded, or it never terminates."""
    sidekick = make_sidekick(
        [reply(f"Attempt {i}") for i in range(MAX_ATTEMPTS)],
        [verdict(met=False) for _ in range(MAX_ATTEMPTS)],
    )
    history = await sidekick.run_turn("Impossible task", "Never satisfied", [])

    assert sidekick.attempts == MAX_ATTEMPTS
    assert history[1]["content"] == f"Attempt {MAX_ATTEMPTS - 1}"


# ------------------------------------------------------- human in the loop


async def test_an_interrupt_pauses_the_turn_and_surfaces_the_request() -> None:
    interrupt = {
        "__interrupt__": [
            type("I", (), {"value": {"action_requests": [{"description": "Send a push saying hello"}]}})()
        ],
        "messages": [FakeMessage("...")],
        "todos": [],
    }
    sidekick = make_sidekick([interrupt], [])
    history = await sidekick.run_turn("Notify me", "Notified", [])

    assert sidekick.paused is True
    assert sidekick.pending_actions == 1
    assert "Send a push saying hello" in history[-1]["content"]
    assert sidekick.evaluator.prompts == []  # nothing judged: the work has not happened


async def test_resuming_approves_exactly_the_paused_actions() -> None:
    sidekick = make_sidekick([reply("Sent.")], [verdict(met=True)])
    sidekick.pending_actions = 2
    sidekick.paused = True

    await sidekick.resume([])

    resume_command = sidekick.worker.payloads[0]
    assert resume_command.resume == {"decisions": [{"type": "approve"}, {"type": "approve"}]}
    assert sidekick.paused is False


def test_every_outward_facing_tool_is_behind_an_approval_gate() -> None:
    """Both of these reach outside the sandbox — one messages the user's phone,
    the other asks them to act. Neither should fire unseen."""
    assert APPROVAL_REQUIRED == {"send_push_notification": True, "request_human_help": True}
    assert all(APPROVAL_REQUIRED.values())


# ------------------------------------------------------------- guardrails


def test_the_middleware_stack_covers_every_production_concern() -> None:
    names = [type(m).__name__ for m in _middleware()]
    assert names == [
        "TolerateToolErrors",        # resilience
        "TodoListMiddleware",        # planning
        "PIIMiddleware",             # safety: email
        "PIIMiddleware",             # safety: credit card, including tool results
        "ModelCallLimitMiddleware",  # cost
        "HumanInTheLoopMiddleware",  # control
    ]


def test_credit_card_redaction_also_applies_to_tool_results() -> None:
    """A card number scraped off a web page has to be caught on the way in, not
    only on the way out."""
    card = next(m for m in _middleware() if type(m).__name__ == "PIIMiddleware" and m.pii_type == "credit_card")
    assert card.apply_to_tool_results is True


# ----------------------------------------------------------------- prompts


def test_the_evaluator_is_given_the_tool_trace_as_evidence() -> None:
    """Judging the transcript alone rewards confident writing; the tool calls are
    the only evidence of what actually happened."""
    prompt = evaluator_prompt("Book a flight", "Booked", "I booked it.", ["browser_navigate", "browser_click"])
    assert "browser_navigate, browser_click" in prompt
    assert "using the tool calls as evidence" in prompt


def test_no_success_criteria_falls_back_to_a_default() -> None:
    sidekick = make_sidekick([reply("Done.")], [verdict(met=True)])
    import asyncio

    asyncio.run(sidekick.run_turn("Do the thing", "", []))
    assert sidekick.success_criteria == DEFAULT_SUCCESS_CRITERIA


def test_each_session_gets_its_own_checkpoint_thread() -> None:
    """The session id is the LangGraph thread id, so two Sidekicks cannot read
    each other's conversation."""
    assert Sidekick().session_id != Sidekick().session_id


@pytest.mark.parametrize("tools", [[], ["browser_navigate"]])
def test_evaluator_prompt_handles_any_tool_trace(tools: list[str]) -> None:
    prompt = evaluator_prompt("t", "c", "r", tools)
    assert ("none" in prompt) == (not tools)

"""Configuration for the Sidekick.

The numbers here are the safety envelope, and they are the first thing to look at
when someone asks how you keep an agent with a browser and a filesystem from
doing something expensive or irreversible.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent

#: The only directory the filesystem MCP server is allowed to touch. This is the
#: sandbox boundary: it is enforced by the server's own scoping, not by a prompt,
#: so the agent physically cannot write outside it.
SANDBOX = Path(os.getenv("SIDEKICK_SANDBOX", PROJECT_ROOT / "sandbox")).resolve()

#: The worker does the task; the evaluator judges it. Deliberately separable, so
#: you can run a cheaper worker under a stronger judge or the reverse.
WORKER_MODEL = os.getenv("WORKER_MODEL", "openai:gpt-5.4-mini")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "gpt-5.4-mini")

#: How many times the worker may be sent back with evaluator feedback before the
#: turn ends regardless. Without this the evaluator-optimizer loop is unbounded.
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))

#: Hard ceiling on model calls within a single worker run, enforced by LangChain's
#: ModelCallLimitMiddleware. This is the cost guardrail; MAX_ATTEMPTS is the
#: quality one.
MODEL_CALL_LIMIT = int(os.getenv("MODEL_CALL_LIMIT", "30"))

#: Used when the user gives a task but no definition of success.
DEFAULT_SUCCESS_CRITERIA = "The answer should be clear, correct and complete"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
PUSHOVER_USER = os.getenv("PUSHOVER_USER")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")


def search_configured() -> bool:
    """True when Google Serper is available, so the web search tool will work."""
    return bool(SERPER_API_KEY)


def push_configured() -> bool:
    """True when Pushover credentials are present and notifications will send."""
    return bool(PUSHOVER_USER and PUSHOVER_TOKEN)

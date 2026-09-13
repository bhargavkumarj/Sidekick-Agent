"""The Sidekick's tools: our own, ready-made LangChain ones, and MCP servers.

Three sources, one flat list handed to `create_agent`:

* **ours** — a push notification, and `request_human_help`, which is the
  interesting one (see below);
* **LangChain community** — Google Serper web search and Wikipedia;
* **MCP** — a real headed browser (Playwright) and a filesystem scoped to the
  sandbox directory.

`request_human_help` deserves a note. It is a tool whose implementation does
nothing at all — it just returns "the user says it is done". Its entire value is
that calling it triggers a `HumanInTheLoopMiddleware` interrupt, which pauses the
graph and surfaces the agent's request in the UI. It is a *control-flow* tool: the
model uses it to hand control back to a human when it hits a login, a captcha or
a 2FA prompt, and the run resumes from the checkpoint afterwards with the browser
in whatever state the human left it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests
import wikipedia
from langchain_community.tools import GoogleSerperRun, WikipediaQueryRun
from langchain_community.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper
from langchain_core.tools import tool

from sidekick.config import PUSHOVER_TOKEN, PUSHOVER_USER, push_configured, search_configured
from sidekick.tools.mcp_sessions import McpSessions, mcp_connections

logger = logging.getLogger(__name__)

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"
TIMEOUT_SECONDS = 15

# Wikimedia rejects the wikipedia library's default user agent, so identify properly.
wikipedia.set_user_agent("sidekick-agent (https://github.com/)")


@tool
def send_push_notification(text: str) -> str:
    """Send a short push notification to the user's phone."""
    if not push_configured():
        logger.info("Push (not configured, logged only): %s", text)
        return "Push notifications are not configured, so nothing was sent."
    response = requests.post(
        PUSHOVER_URL,
        data={"token": PUSHOVER_TOKEN, "user": PUSHOVER_USER, "message": text},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return "Notification sent"


@tool
def request_human_help(instructions: str) -> str:
    """Ask the user to do something in the browser window that you cannot do yourself,
    such as logging in to a site, passing a captcha, or approving two-factor
    authentication. Explain exactly what you need them to do. The run pauses until
    they have done it."""
    # Intentionally a no-op. The pause happens because HumanInTheLoopMiddleware is
    # configured to interrupt on this tool; by the time the body would run, the
    # human has already acted.
    return "The user says it is done. Continue with the task."


def local_tools() -> list:
    """Our own tools, plus the ready-made ones that are actually usable."""
    tools = [send_push_notification, request_human_help, WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())]
    if search_configured():
        tools.append(GoogleSerperRun(api_wrapper=GoogleSerperAPIWrapper()))
    else:
        # Handing the model a search tool that will 401 on every call is worse
        # than not having one: it burns turns and confuses the evaluator.
        logger.warning("SERPER_API_KEY is not set; the Sidekick has no web search tool.")
    return tools


async def get_all_tools(sandbox: Path) -> tuple[list, McpSessions]:
    """The full tool list and the session holder that must be stopped on cleanup."""
    sessions = McpSessions(mcp_connections(sandbox))
    mcp_tools = await sessions.start()
    return local_tools() + mcp_tools, sessions


__all__ = ["get_all_tools", "local_tools", "request_human_help", "send_push_notification"]

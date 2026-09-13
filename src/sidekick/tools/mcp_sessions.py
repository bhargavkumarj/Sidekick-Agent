"""Keeping MCP stdio sessions alive across many tool calls.

This module exists because of one asyncio constraint that is easy to get wrong.

An MCP stdio transport must be opened and closed **from the same task**. The
obvious implementation — `async with client.session(name)` around each tool call
— satisfies that, but it starts and stops the server every time, which means the
browser loses all its state between calls. An agent that navigates to a page and
then takes a snapshot would get a fresh, blank browser for the snapshot.

So one background task owns the sessions for their whole lifetime: it opens them
inside an `AsyncExitStack`, signals ready, waits, and unwinds the stack when
asked to stop. Everything else just uses the tools it produced. The browser keeps
its cookies, its page and its scroll position for the entire conversation.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


def mcp_connections(sandbox: Path) -> dict:
    """The MCP servers the Sidekick uses: a headed browser and a scoped filesystem.

    The filesystem server is started with exactly one allowed directory. That
    scoping is the sandbox — enforced by the server, not by an instruction in the
    prompt, so the agent cannot write outside it however it is asked.
    """
    return {
        "playwright": {
            "transport": "stdio",
            "command": "npx",
            "args": ["@playwright/mcp@latest", "--isolated"],
        },
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", str(sandbox)],
        },
    }


class McpSessions:
    """Owns a set of MCP sessions on one background task for their whole lifetime."""

    def __init__(self, connections: dict) -> None:
        self.connections = connections
        self.tools: list = []
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        """Open every session, publish the tools, then hold until told to stop."""
        client = MultiServerMCPClient(self.connections)
        async with AsyncExitStack() as stack:
            for name in self.connections:
                session = await stack.enter_async_context(client.session(name))
                self.tools += await load_mcp_tools(session, server_name=name)
            self._ready.set()
            await self._stop.wait()
        # Leaving the stack here — on the same task that entered it — is the whole
        # point of this class.

    async def start(self) -> list:
        """Start the servers and return their tools.

        Waits on *either* readiness or the task finishing, so a server that fails
        to start surfaces its real exception instead of hanging forever on an
        event that will never be set.
        """
        self._task = asyncio.create_task(self._run())
        ready = asyncio.create_task(self._ready.wait())
        await asyncio.wait([ready, self._task], return_when=asyncio.FIRST_COMPLETED)
        ready.cancel()
        if self._task.done():
            self._task.result()  # re-raises the real startup error
        return self.tools

    def stop(self) -> None:
        """Shut the servers down. The browser window closes."""
        self._stop.set()

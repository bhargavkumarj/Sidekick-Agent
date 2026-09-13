"""Entry point for the Sidekick.

    sidekick            launch the app
    sidekick doctor     check keys and binaries before starting a browser
"""

from __future__ import annotations

import argparse
import shutil
import sys

from sidekick import config


def _doctor() -> int:
    print("Sidekick environment check\n")

    ok = bool(config.OPENAI_API_KEY)
    print("  [ ok ] OPENAI_API_KEY" if ok else "  [FAIL] OPENAI_API_KEY is missing — nothing will run")

    for name, value, why in [
        ("SERPER_API_KEY", config.SERPER_API_KEY, "the Sidekick has no web search tool"),
        ("PUSHOVER_USER", config.PUSHOVER_USER, "push notifications are logged, not sent"),
        ("PUSHOVER_TOKEN", config.PUSHOVER_TOKEN, "push notifications are logged, not sent"),
    ]:
        print(f"  [ ok ] {name}" if value else f"  [warn] {name} not set — {why}")

    for name, why in [("npx", "the browser and filesystem MCP servers cannot start")]:
        path = shutil.which(name)
        print(f"  [ ok ] {name} ({path})" if path else f"  [FAIL] {name} not found — {why}")
        ok = ok and bool(path)

    print(f"\n  sandbox        {config.SANDBOX}")
    print(f"  worker         {config.WORKER_MODEL}")
    print(f"  evaluator      {config.EVALUATOR_MODEL}")
    print(f"  max attempts   {config.MAX_ATTEMPTS}   (evaluator retries)")
    print(f"  call limit     {config.MODEL_CALL_LIMIT}   (model calls per worker run)")
    print("\nReady." if ok else "\nNot ready: fix the FAIL lines above.")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sidekick", description="A personal agent with a browser.")
    parser.add_argument("command", nargs="?", choices=["run", "doctor"], default="run")
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return _doctor()

    from sidekick.ui.app import main as app_main

    app_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())

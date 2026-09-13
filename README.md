# Sidekick — an agent that works to a definition of success

Give it a task **and a definition of done**. A worker agent with a real Chrome
browser, a sandboxed filesystem, web search and Wikipedia goes and does it. A
separate evaluator agent then judges the result against your criteria and either
accepts it, hands feedback back for another attempt, or stops and puts a question
to you.

Around the worker sits the part that makes this more than a demo: a middleware
stack that redacts PII, caps model spend, pauses for human approval before
sensitive actions, and turns tool crashes into something the model can recover
from.

```
   you: task  +  success criteria
        │
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  WORKER  (LangChain create_agent, checkpointed)             │
 │                                                             │
 │   middleware stack, outermost first:                        │
 │     TolerateToolErrors ······· a tool crash becomes a message│
 │     TodoListMiddleware ······· the plan, streamed to the UI  │
 │     PIIMiddleware(email) ····· redaction                     │
 │     PIIMiddleware(card) ······ redaction, incl. tool RESULTS │
 │     ModelCallLimit(30) ······· the cost ceiling              │
 │     HumanInTheLoop ··········· pause → approve → resume      │
 │                                                             │
 │   tools:  Playwright MCP (real Chrome) · filesystem MCP      │
 │           (scoped to ./sandbox) · Serper · Wikipedia ·       │
 │           push · request_human_help                          │
 └───────────────────────┬─────────────────────────────────────┘
                         │ reply + the tools it actually called
                         ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  EVALUATOR  (structured output, not prose)                  │
 │    { feedback, success_criteria_met, user_input_needed }    │
 └───────────────────────┬─────────────────────────────────────┘
                         │
     met ────────────────┼──────────────► return the answer
     needs the user ─────┤──────────────► stop, surface the question
     neither ────────────┘──────────────► retry with feedback (≤ 3)
```

---

## Why this project is interesting

**It knows whether it succeeded.** Most agents return an answer and leave you to
judge it. This one is told the criteria up front and a second model checks
against them — and crucially, the evaluator is given **the list of tools the
worker actually called**, not just its prose. An agent that claims to have booked
a flight without ever opening the browser is caught by the tool trace. Judging
the transcript alone rewards confident writing.

**The guardrails are real code, not advice in a prompt.** Six middlewares, each
answering a different deployment question:

| Concern | Mechanism |
|---|---|
| What if a tool breaks? | `TolerateToolErrors` — the failure becomes a message the model recovers from |
| What is it doing right now? | `TodoListMiddleware` — the agent's own plan, live in the UI |
| What if personal data flows through? | `PIIMiddleware` for email and credit cards — the card rule applies to **tool results** too, so a card number scraped off a page is caught on the way in |
| What stops it costing $400? | `ModelCallLimitMiddleware(run_limit=30)` |
| What stops it doing something irreversible? | `HumanInTheLoopMiddleware` — the graph interrupts, the UI shows an Approve button, `resume()` continues the same turn from the checkpoint |
| What stops the retry loop? | `MAX_ATTEMPTS` in the evaluator loop |

**The sandbox is enforced, not requested.** The filesystem MCP server is started
with exactly one allowed directory. The agent cannot write outside it however it
is asked, because the server is the one saying no.

**Human-in-the-loop that actually works.** `request_human_help` is a tool whose
implementation does nothing. Its whole value is that calling it triggers an
interrupt — so when the agent hits a login, a captcha or a 2FA prompt, it pauses
and tells you exactly what to do in the browser window that is already open in
front of you. You do it, click Approve, and the run picks up mid-task from the
checkpoint with the browser in the state you left it.

---

## Quick start

```bash
cp .env.example .env    # add your OPENAI_API_KEY
```

```bash
uv sync && uv run sidekick doctor
```

```bash
uv run sidekick
```

A Chrome window opens alongside the app — that is the agent's browser, and you
can watch it work. The Go button stays disabled until the MCP servers are up.

### Requirements

* Python 3.12+
* Node.js 18+ — `npx` launches the Playwright and filesystem MCP servers
* `OPENAI_API_KEY`. A [Serper](https://serper.dev) key adds web search; without
  it the agent keeps Wikipedia and the browser and is simply told it has no
  search tool, rather than being handed one that fails on every call.

### Things to try

| Task | Success criteria |
|---|---|
| Find the three cheapest direct flights from London to Tokyo next month | A table with airline, date and price, saved to `flights.md` |
| Research the current state of MCP adoption and write me a briefing | 500 words, at least four sources cited with links |
| Summarise the Wikipedia article on the Antikythera mechanism | Under 200 words, no jargon, saved as `antikythera.md` |

Ask it to send you a push notification and you will see the approval gate fire.

---

## Layout

```
src/sidekick/
├── config.py              models, budgets, the sandbox path
├── cli.py                 run / doctor
├── agent.py               the Sidekick: worker + evaluator loop + middleware stack
├── prompts.py             both prompts, side by side
├── tools/
│   ├── __init__.py          our tools, LangChain tools, and the MCP tool list
│   └── mcp_sessions.py      keeping stdio MCP sessions alive across tool calls
└── ui/
    ├── app.py               Gradio: chat, live plan panel, approve button
    └── theme.py             styling
sandbox/                   the only directory the agent may write to
```

---

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** |
| `SERPER_API_KEY` | unset | Adds the web search tool |
| `PUSHOVER_USER` / `_TOKEN` | unset | Push notifications actually send |
| `WORKER_MODEL` | `openai:gpt-5.4-mini` | The agent that does the task |
| `EVALUATOR_MODEL` | `gpt-5.4-mini` | The agent that judges it — separable on purpose |
| `MAX_ATTEMPTS` | `3` | Evaluator retries before the turn ends regardless |
| `MODEL_CALL_LIMIT` | `30` | Model calls inside one worker run |
| `SIDEKICK_SANDBOX` | `./sandbox` | The filesystem boundary |

---

## Tests

```bash
uv run --extra dev pytest
```

14 tests, no model calls, no browser. The worker is an LLM and cannot be unit
tested; the loop around it can, and that is where the safety properties live —
verdict routing, retry bounding, the approval gate covering every outward-facing
tool, and PII redaction reaching tool results.

---
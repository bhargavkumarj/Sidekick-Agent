"""The Gradio front end.

Two things here are more interesting than they look.

**The plan panel has exactly one writer.** The `todos` panel is updated only by a
1-second timer, never as an output of the message handler. If a long-running
event declared the panel as an output, Gradio would mark it pending for the whole
run and no live update would render — you would watch a frozen plan for two
minutes and then see the finished one. Giving the timer sole ownership is what
makes the agent's plan visible *while* it works.

**Approval is a button, not a prompt.** When the worker pauses on a sensitive
action, `run_turn` returns immediately with `paused` set, the Approve button
becomes visible, and clicking it calls `resume()` — which continues the same
turn from the LangGraph checkpoint.
"""

from __future__ import annotations

import html

import gradio as gr

from sidekick.agent import Sidekick
from sidekick.ui import theme

LAUNCH_STYLE = {"theme": theme.THEME, "css": theme.CSS, "head": theme.JS}

HEADER = """
<div id="header">
    <div class="context-label">Your personal co-worker</div>
    <h1>Sidekick</h1>
    <div class="brand-bar"></div>
</div>
"""


def render_todos(todos: list | None) -> str:
    """The worker's live plan. These are the agent's own todos, not a progress bar."""
    if not todos:
        items = '<div class="placeholder">The Sidekick will write its plan here as it works</div>'
    else:
        items = (
            "<ul>"
            + "".join(
                f'<li class="{todo["status"]}"><span class="mark"></span>{html.escape(todo["content"])}</li>'
                for todo in todos
            )
            + "</ul>"
        )
    return f"<h3>Plan</h3>{items}"


async def setup():
    """Bring up a Sidekick, and only then enable the Go button.

    Setup starts a browser and a filesystem server over MCP, which takes a few
    seconds. The button stays disabled until the tools genuinely exist.
    """
    sidekick = Sidekick()
    await sidekick.setup()
    return sidekick, gr.update(interactive=True)


async def process_message(sidekick, message, success_criteria, history):
    if sidekick is None:  # clicked before setup finished bringing up the MCP servers
        return history, gr.update(visible=False), sidekick
    results = await sidekick.run_turn(message, success_criteria, history)
    return results, gr.update(visible=sidekick.paused), sidekick


async def approve(sidekick, history):
    results = await sidekick.resume(history)
    return results, gr.update(visible=sidekick.paused), sidekick


def watch_todos(sidekick):
    """The timer's tick. Sole writer of the plan panel — see the module docstring."""
    return render_todos(sidekick.todos if sidekick else [])


async def reset(sidekick):
    """Tear down the current session and start a clean one, browser and all."""
    if sidekick:
        sidekick.cleanup()
    fresh = Sidekick()
    await fresh.setup()
    return "", "", None, gr.update(visible=False), fresh


def free_resources(sidekick):
    """Called by Gradio when the session state is dropped, so the browser closes."""
    if sidekick:
        sidekick.cleanup()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Sidekick") as ui:
        gr.HTML(HEADER)
        sidekick = gr.State(delete_callback=free_resources)

        with gr.Row():
            chatbot = gr.Chatbot(label="Sidekick", height=320, scale=3, elem_id="chat")
            with gr.Column(scale=1):
                todos_panel = gr.HTML(render_todos([]), elem_id="plan-panel")

        with gr.Group(elem_id="ask-panel"):
            with gr.Row():
                message = gr.Textbox(show_label=False, placeholder="Your request to the Sidekick")
            with gr.Row():
                success_criteria = gr.Textbox(
                    show_label=False, placeholder="What are your success criteria?"
                )

        with gr.Row():
            reset_button = gr.Button("Reset", elem_id="reset-button")
            approve_button = gr.Button("Approve and continue", visible=False, elem_id="approve-button")
            go_button = gr.Button("Go!", elem_id="go-button", interactive=False)

        timer = gr.Timer(1)

        ui.load(setup, [], [sidekick, go_button])
        timer.tick(watch_todos, [sidekick], [todos_panel], show_progress="hidden")

        ask_inputs = [sidekick, message, success_criteria, chatbot]
        ask_outputs = [chatbot, approve_button, sidekick]
        message.submit(process_message, ask_inputs, ask_outputs)
        success_criteria.submit(process_message, ask_inputs, ask_outputs)
        go_button.click(process_message, ask_inputs, ask_outputs)

        approve_button.click(approve, [sidekick, chatbot], ask_outputs)
        reset_button.click(
            reset, [sidekick], [message, success_criteria, chatbot, approve_button, sidekick]
        )
    return ui


def main() -> None:
    build_ui().launch(inbrowser=True, **LAUNCH_STYLE)


if __name__ == "__main__":
    main()

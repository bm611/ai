"""Minimal full-screen chat using the existing streaming client."""

import copy
import io
import os
import time
from datetime import date
from importlib.metadata import version

from rich import console, markdown, text
from textual import app, binding, containers, widgets, work
from textual.worker import get_current_worker

from ai import client


class ChatApp(app.App):
    """Keep conversation, prompt, and model status on one terminal screen."""

    CSS = """
    Screen { background: #0b0b0b; color: #d4d4d4; padding: 1; }
    #heading { height: 1; margin-bottom: 1; color: #858585; }
    #conversation { display: none; height: 1fr; scrollbar-size: 1 1; }
    #conversation.active { display: block; }
    .message { height: auto; margin-bottom: 1; }
    .question { color: #ababab; }
    #composer { height: 1; }
    #marker { width: 2; color: #eeeeee; }
    #prompt {
        width: 1fr; height: 1; padding: 0; border: none;
        background: transparent; color: #eeeeee;
    }
    #prompt:focus { border: none; }
    #prompt > .input--cursor { background: #b9cdfb !important; color: #0b0b0b !important; }
    #status { height: 1; margin-top: 1; color: #858585; }
    """
    BINDINGS = [
        binding.Binding("ctrl+c", "quit", "Quit", priority=True),
        binding.Binding("ctrl+d", "quit", "Quit", priority=True),
    ]

    def __init__(
        self,
        model: str,
        provider: dict | None = None,
        theme: str = "auto",
        initial_prompt: str = "",
        show_heading: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.provider = provider
        self.chat_theme = theme
        self.initial_prompt = initial_prompt
        self.show_heading = show_heading
        self.messages = self._new_messages()
        self.busy = False

    @staticmethod
    def _new_messages() -> list[dict]:
        return [
            {
                "role": "system",
                "content": f"Today's date is {date.today().isoformat()}.",
            }
        ]

    def compose(self) -> app.ComposeResult:
        heading = text.Text.assemble(
            ("✦", "bold white"),
            f" v{version('ai')} · Run /help for commands",
        )
        yield widgets.Static(heading, id="heading")
        yield containers.VerticalScroll(id="conversation")
        with containers.Horizontal(id="composer"):
            yield widgets.Static("│", id="marker")
            yield widgets.Input(id="prompt")
        yield widgets.Static(id="status")

    async def on_mount(self) -> None:
        self.query_one("#heading").display = self.show_heading
        self._status("ready")
        self.query_one(widgets.Input).focus()
        if self.initial_prompt:
            await self._submit(self.initial_prompt)

    def _status(self, state: str) -> None:
        label = f"{self.chat_theme} · {self.model} · {state}"
        self.query_one("#status", widgets.Static).update(text.Text(label))

    async def on_input_submitted(self, event: widgets.Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        event.input.value = ""
        if prompt in ("/quit", "/exit", "q"):
            self.exit()
            return
        if self.busy:
            self._status("still responding · Ctrl+C to quit")
            event.input.value = prompt
            return
        if prompt == "/clear":
            await self.query_one("#conversation").remove_children()
            self.query_one("#conversation").remove_class("active")
            self.messages = self._new_messages()
            self._status("ready")
        elif prompt == "/help":
            await self._notice(
                "Enter sends · /clear starts fresh · /quit exits\n"
                "Scroll to review answers · Ctrl+C or Ctrl+D exits"
            )
        elif prompt.startswith("/"):
            self._status("unknown command · try /help")
        else:
            await self._submit(prompt)

    async def _notice(self, value: str) -> None:
        history = self.query_one("#conversation", containers.VerticalScroll)
        history.add_class("active")
        await history.mount(widgets.Static(text.Text(value), classes="message"))
        history.scroll_end(animate=False)

    async def _submit(self, prompt: str) -> None:
        self.busy = True
        history = self.query_one("#conversation", containers.VerticalScroll)
        history.add_class("active")
        answer = widgets.Static("", classes="message")
        await history.mount(
            widgets.Static(text.Text(f"› {prompt}"), classes="message question"),
            answer,
        )
        history.scroll_end(animate=False)
        self._status("thinking…")
        self._respond(prompt, answer)

    def _update_answer(self, answer: widgets.Static, content: str) -> None:
        history = self.query_one("#conversation", containers.VerticalScroll)
        follow = history.is_vertical_scroll_end
        answer.update(markdown.Markdown(client._prep_math(content)))
        self._status("responding…")
        if follow:
            history.scroll_end(animate=False)

    @work(thread=True, exclusive=True)
    def _respond(self, prompt: str, answer: widgets.Static) -> None:
        worker = get_current_worker()
        messages = copy.deepcopy(self.messages)
        messages.append({"role": "user", "content": prompt})
        started = time.perf_counter()
        last_render = 0.0

        def update(content: str) -> None:
            nonlocal last_render
            if worker.is_cancelled:
                raise InterruptedError("Chat closed")
            now = time.perf_counter()
            if now - last_render >= 0.05:
                self.call_from_thread(self._update_answer, answer, content)
                last_render = now

        try:
            key = (
                "MIMO_API_KEY"
                if client._is_mimo_model(self.model)
                else "OPENROUTER_API_KEY"
            )
            if not os.environ.get(key):
                raise ValueError(f"Set {key} in your shell, then restart ai.")
            url, headers, body = client._build_request(
                self.model, messages, self.provider, stream=True, tools=True
            )
            output = console.Console(file=io.StringIO())
            content, _, _, usage = client._do_stream(
                url,
                body,
                headers,
                messages,
                output,
                markdown.Markdown,
                on_text=update,
            )
            if worker.is_cancelled:
                return
            messages.append({"role": "assistant", "content": content})
            elapsed = time.perf_counter() - started
            tokens = usage.get("completion_tokens")
            state = f"{elapsed:.1f}s"
            if tokens is not None:
                state += f" · {tokens} tokens"
            self.call_from_thread(
                self._finish,
                answer,
                content or "No text returned.",
                state,
                messages,
            )
        except Exception as error:
            # Keep request failures inside the UI so users can retry.
            if not worker.is_cancelled:
                self.call_from_thread(
                    self._finish,
                    answer,
                    f"Request failed: {error}",
                    "error · try again",
                    None,
                )

    def _finish(
        self,
        answer: widgets.Static,
        content: str,
        state: str,
        messages: list[dict] | None,
    ) -> None:
        if messages is None:
            answer.update(text.Text(content, style="red"))
        else:
            self._update_answer(answer, content)
            self.messages = messages
        self.busy = False
        self._status(state)
        self.query_one(widgets.Input).focus()

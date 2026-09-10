"""Minimal full-screen chat using the existing streaming client."""

import copy
import difflib
import io
import os
import time
from datetime import date
from importlib.metadata import version
from typing import NamedTuple

from rich import console, markdown, text
from textual import app, binding, containers, events, widgets, work
from textual.css.query import NoMatches
from textual.suggester import SuggestFromList
from textual.theme import Theme
from textual.widgets import markdown as md
from textual.widgets.option_list import Option
from textual.worker import get_current_worker

from ai import client

# Hand text to the UI thread at most this often; the Markdown widget coalesces
# whatever arrives between flushes, so a slower cadence costs nothing visually.
_FLUSH_INTERVAL = 0.08


class Command(NamedTuple):
    """A slash command surfaced by the in-chat palette."""

    name: str
    args: str
    description: str


COMMANDS: tuple[Command, ...] = (
    Command("/help", "", "List commands and keyboard shortcuts"),
    Command("/new", "", "Start a new conversation"),
    Command("/clear", "", "Clear the screen, keep context"),
    Command("/cancel", "", "Stop the current request"),
    Command("/quit", "", "Exit and return to the shell"),
)

# Aliases match in the palette but always resolve to their canonical command.
COMMAND_ALIASES = {"/exit": "/quit"}

_COMMAND_LABEL_WIDTH = max(
    len(f"{command.name} {command.args}".strip()) for command in COMMANDS
)


def _command_names() -> list[str]:
    return [command.name for command in COMMANDS]


def _match_commands(value: str) -> list[Command]:
    """Commands whose name or alias starts with the typed slash value.

    The palette collapses once the value is a complete command so it doesn't
    linger while the user is ready to run it.
    """
    if not value.startswith("/") or " " in value:
        return []
    needle = value.lower()
    exact = {command.name.lower() for command in COMMANDS} | set(COMMAND_ALIASES)
    if needle in exact:
        return []
    matches = []
    for command in COMMANDS:
        aliases = [a for a, target in COMMAND_ALIASES.items() if target == command.name]
        if any(candidate.lower().startswith(needle) for candidate in (command.name, *aliases)):
            matches.append(command)
    return matches


def _unknown_command(value: str) -> str:
    close = difflib.get_close_matches(value, _command_names(), n=1, cutoff=0.5)
    if close:
        return f"unknown command · did you mean {close[0]}?"
    return "unknown command · try /help"


def _help_text() -> text.Text:
    """Formatted command reference shown by /help."""
    out = text.Text()
    out.append("Commands\n", "bold")
    for command in COMMANDS:
        label = f"{command.name} {command.args}".strip()
        out.append(f"  {label.ljust(_COMMAND_LABEL_WIDTH)}  ", "bold")
        out.append(f"{command.description}\n", "dim")
    out.append("\nShortcuts\n", "bold")
    for keys, description in (
        ("Enter", "Send a message"),
        ("↑ / ↓", "Pick a command"),
        ("Tab / →", "Complete a command"),
        ("Ctrl+C / Ctrl+D", "Exit"),
    ):
        out.append(f"  {keys.ljust(_COMMAND_LABEL_WIDTH)}  ", "bold")
        out.append(f"{description}\n", "dim")
    return out


# Textual palettes mirroring the rich themes the non-interactive CLI uses.
THEMES = {
    "dark": Theme(
        name="ai-dark",
        primary="#b9cdfb",
        secondary="#89b4fa",
        accent="#c8a2f0",
        foreground="#d4d4d4",
        background="#0b0b0b",
        surface="#131313",
        panel="#1c1c1c",
        success="#8ec07c",
        warning="#e0af68",
        error="#f47067",
        dark=True,
    ),
    "light": Theme(
        name="ai-light",
        primary="#3b5bdb",
        secondary="#1971c2",
        accent="#7048e8",
        foreground="#1f2328",
        background="#fbfbfa",
        surface="#f3f3f1",
        panel="#e7e7e4",
        success="#2f9e44",
        warning="#e8590c",
        error="#c92a2a",
        dark=False,
    ),
    "retro": Theme(
        name="ai-retro",
        primary="#33ff66",
        secondary="#22cc55",
        accent="#b6ff4d",
        foreground="#c8f7c5",
        background="#020602",
        surface="#0a120a",
        panel="#122212",
        success="#33ff66",
        warning="#d7ff4d",
        error="#ff5f5f",
        dark=True,
    ),
}


class CodeFence(md.MarkdownFence):
    """Fenced code in a framed block labelled with its language."""

    def on_mount(self) -> None:
        self._label()

    def _label(self) -> None:
        title = self.lexer or "code"
        if getattr(self, "border_title", None) != title:
            self.border_title = title

    def _copy_context(self, block: md.MarkdownBlock) -> None:
        # Streaming reuses the last block, so re-label when its language lands.
        super()._copy_context(block)
        self._label()


class Answer(widgets.Markdown):
    """Assistant reply, appended block by block while the answer streams."""

    BLOCKS = widgets.Markdown.BLOCKS | {"fence": CodeFence, "code_block": CodeFence}


class CommandPalette(widgets.OptionList):
    """Dropdown of matching slash commands shown above the composer."""

    can_focus = False

    def update_matches(self, value: str) -> None:
        matches = _match_commands(value)
        if not matches:
            self.clear_options()
            self.display = False
            return
        self.set_options(
            Option(
                text.Text.assemble(
                    (f"{command.name} {command.args}".strip().ljust(_COMMAND_LABEL_WIDTH), "bold"),
                    ("   ", ""),
                    (command.description, "dim"),
                ),
                id=command.name,
            )
            for command in matches
        )
        self.display = True
        self.highlighted = 0

    def command(self) -> str | None:
        """The canonical name of the highlighted command, if any."""
        option = self.highlighted_option
        return option.id if option is not None else None


class CommandInput(widgets.Input):
    """Composer input that hands navigation keys to the command palette."""

    async def _on_key(self, event: events.Key) -> None:
        palette = self._palette()
        if palette is not None and palette.display:
            if event.key in ("up", "down"):
                if event.key == "down":
                    palette.action_cursor_down()
                else:
                    palette.action_cursor_up()
                event.stop()
                event.prevent_default()
                return
            if event.key == "tab":
                command = palette.command()
                if command is not None:
                    self.value = command
                    self.cursor_position = len(command)
                event.stop()
                event.prevent_default()
                return
            if event.key == "escape":
                palette.display = False
                event.stop()
                event.prevent_default()
                return
            if event.key == "enter":
                command = palette.command()
                if command is not None and self.value.strip() != command:
                    # Turn a partial command into the highlighted one before the
                    # submit binding runs, so Enter always runs what's shown.
                    self.value = command
                    self.cursor_position = len(command)
        await super()._on_key(event)

    def _palette(self) -> CommandPalette | None:
        try:
            return self.screen.query_one(CommandPalette)
        except NoMatches:
            return None


class ChatApp(app.App):
    """Keep conversation, prompt, and model status on one terminal screen."""

    CSS = """
    Screen { background: $background; color: $foreground; padding: 1; }
    #heading { height: 1; margin-bottom: 1; color: $text-muted; }
    #conversation { display: none; height: 1fr; scrollbar-size: 1 1; }
    #conversation.active { display: block; }
    .message { height: auto; margin-bottom: 1; }
    .question { color: $foreground 80%; }
    .error { color: $text-error; }
    .muted { color: $text-muted; }
    #palette {
        display: none; height: auto; max-height: 8;
        border: round $panel; background: $surface;
        padding: 0 1; margin: 0 0 1 0;
    }
    #palette > .option-list--option { padding: 0 1; }
    #palette > .option-list--option-highlighted {
        color: $foreground; background: $primary 30%;
    }
    #composer { height: 1; }
    #marker { width: 2; color: $primary; }
    #prompt {
        width: 1fr; height: 1; padding: 0; border: none;
        background: transparent; color: $foreground;
    }
    #prompt:focus { border: none; }
    #prompt > .input--cursor { background: $primary !important; color: $background !important; }
    #status { height: 1; margin-top: 1; color: $text-muted; }

    Answer { padding: 0; background: transparent; }
    Answer MarkdownH1 {
        color: $primary; background: transparent;
        text-style: bold; content-align: left middle; margin: 0 0 1 0;
    }
    Answer MarkdownH2 {
        color: $primary; background: transparent; text-style: bold; margin: 1 0 0 0;
    }
    Answer MarkdownH3, Answer MarkdownH4 {
        color: $accent; background: transparent; text-style: bold; margin: 1 0 0 0;
    }
    Answer MarkdownBlockQuote {
        background: transparent; border-left: outer $primary 60%; padding: 0 1;
    }
    Answer MarkdownHorizontalRule { border-bottom: solid $panel; }
    Answer MarkdownBlock > .code_inline { background: $panel; color: $text-accent; }
    Answer MarkdownFence {
        background: $surface;
        border: round $panel;
        border-title-color: $text-muted;
        border-title-align: left;
        margin: 1 0;
        padding: 0 1;
    }
    Answer MarkdownFence > Label { padding: 0 1; }
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
        self.phase = "ready"
        self.request_started = 0.0
        self.active_answer = None
        self.stream = None
        self._last_status = ""
        # Resolved here so the macOS appearance probe stays off the UI thread.
        self.palette = THEMES[client._resolve_theme(theme)[2]]

    @staticmethod
    def _new_messages() -> list[dict]:
        return [
            {\
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
        yield CommandPalette(id="palette")
        with containers.Horizontal(id="composer"):
            yield widgets.Static("│", id="marker")
            yield CommandInput(
                id="prompt",
                suggester=SuggestFromList(_command_names(), case_sensitive=False),
            )
        yield widgets.Static(id="status")

    async def on_mount(self) -> None:
        self.register_theme(self.palette)
        self.theme = self.palette.name
        self.query_one("#heading").display = self.show_heading
        self._status("ready")
        self.set_interval(1, self._tick)
        self.query_one(widgets.Input).focus()
        if self.initial_prompt:
            await self._submit(self.initial_prompt)

    def _status(self, state: str) -> None:
        if self.busy:
            elapsed = time.perf_counter() - self.request_started
            state += f" · {elapsed:.0f}s · /cancel to stop"
        label = f"{self.chat_theme} · {self.model} · {state}"
        if getattr(self, "_last_status", None) == label:
            return
        self._last_status = label
        self.query_one("#status", widgets.Static).update(text.Text(label))

    def _tick(self) -> None:
        if self.busy:
            self._status(self.phase)

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self._status(phase)

    async def _cancel(self) -> None:
        if not self.busy:
            self._status("nothing to cancel")
            return
        self.workers.cancel_all()
        await self._end_stream()
        await self._replace(self.active_answer, "Cancelled.", "muted")
        self.active_answer = None
        self.busy = False
        self._status("cancelled · ready")

    async def on_input_changed(self, event: widgets.Input.Changed) -> None:
        self.query_one(CommandPalette).update_matches(event.value)

    async def on_input_submitted(self, event: widgets.Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        event.input.value = ""
        if prompt in ("/quit", "/exit", "q"):
            self.exit()
            return
        if prompt == "/cancel":
            await self._cancel()
            return
        if self.busy:
            self._status(self.phase)
            event.input.value = prompt
            return
        if not await self._run_command(prompt):
            await self._submit(prompt)

    async def _clear_screen(self) -> None:
        """Remove every message from view without touching conversation history."""
        history = self.query_one("#conversation", containers.VerticalScroll)
        await history.remove_children()
        history.remove_class("active")

    async def _run_command(self, prompt: str) -> bool:
        """Run a slash command, returning False if it should go to the model."""
        name = COMMAND_ALIASES.get(prompt, prompt)
        if name == "/new":
            await self._clear_screen()
            self.messages = self._new_messages()
            self._status("new conversation · ready")
            return True
        if name == "/clear":
            await self._clear_screen()
            self._status("cleared · context kept")
            return True
        if name == "/help":
            await self._notice(_help_text())
            return True
        if name == "/quit":
            self.exit()
            return True
        if prompt.startswith("/"):
            self._status(_unknown_command(prompt))
            return True
        return False

    async def _notice(self, value: str | text.Text) -> None:
        history = self.query_one("#conversation", containers.VerticalScroll)
        history.add_class("active")
        content = value if isinstance(value, text.Text) else text.Text(value)
        await history.mount(widgets.Static(content, classes="message"))
        history.scroll_end(animate=False)

    async def _submit(self, prompt: str) -> None:
        self.busy = True
        self.request_started = time.perf_counter()
        history = self.query_one("#conversation", containers.VerticalScroll)
        history.add_class("active")
        answer = Answer(classes="message")
        self.active_answer = answer
        await history.mount(
            widgets.Static(text.Text(f"› {prompt}"), classes="message question"),
            answer,
        )
        # Anchoring follows new content without a scroll call per chunk, and
        # releases itself as soon as the reader scrolls back up.
        history.anchor()
        self.stream = widgets.Markdown.get_stream(answer)
        self._set_phase("waiting for model…")
        self._respond(prompt)

    async def _append(self, chunk: str) -> None:
        """Add streamed markdown to the live answer (UI thread only)."""
        if self.stream is None:
            return
        await self.stream.write(chunk)
        if self.phase != "responding…":
            self._set_phase("responding…")

    async def _end_stream(self) -> None:
        stream, self.stream = self.stream, None
        if stream is not None:
            await stream.stop()

    @work(thread=True, exclusive=True)
    def _respond(self, prompt: str) -> None:
        worker = get_current_worker()
        messages = copy.deepcopy(self.messages)
        messages.append({"role": "user", "content": prompt})
        started = time.perf_counter()
        mirrored, document, pending = "", "", ""
        last_flush = 0.0
        last_phase = ""

        def report(phase: str) -> None:
            nonlocal last_phase
            if worker.is_cancelled:
                raise InterruptedError("Request cancelled")
            if phase and phase != last_phase:
                self.call_from_thread(self._set_phase, phase)
                last_phase = phase

        def flush(force: bool = False) -> None:
            nonlocal pending, last_flush
            now = time.perf_counter()
            if pending and (force or now - last_flush >= _FLUSH_INTERVAL):
                chunk, pending = pending, ""
                last_flush = now
                self.call_from_thread(self._append, chunk)

        def update(content: str) -> None:
            # The client reports the whole answer so far; append only the tail
            # so the widget re-renders one block instead of the document.
            nonlocal mirrored, document, pending
            if worker.is_cancelled:
                raise InterruptedError("Chat closed")
            if content.startswith(mirrored):
                delta = content[len(mirrored) :]
            else:
                delta = f"\n\n{content}"  # A tool call restarted the stream.
            mirrored = content
            document += delta
            pending += delta
            flush()

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
                on_status=report,
            )
            if worker.is_cancelled:
                return
            flush(force=True)
            messages.append({"role": "assistant", "content": content})
            elapsed = time.perf_counter() - started
            tokens = usage.get("completion_tokens")
            state = f"{elapsed:.1f}s"
            if tokens is not None:
                state += f" · {tokens} tokens"
            self.call_from_thread(
                self._finish,
                document or content or "No text returned.",
                state,
                messages,
            )
        except Exception as error:
            # Keep request failures inside the UI so users can retry.
            if not worker.is_cancelled:
                self.call_from_thread(
                    self._finish,
                    f"Request failed: {error}",
                    "error · try again",
                    None,
                )

    async def _finish(
        self,
        content: str,
        state: str,
        messages: list[dict] | None,
    ) -> None:
        answer = self.active_answer
        await self._end_stream()
        self.active_answer = None
        if messages is None:
            await self._replace(answer, content, "error")
        else:
            # Math needs the finished document, so re-render once at the end
            # rather than fighting the incremental append on every chunk.
            polished = client._prep_math(content)
            if answer is not None and polished != content:
                await answer.update(polished)
            self.messages = messages
        self.busy = False
        self._status(state)
        self.query_one(widgets.Input).focus()

    async def _replace(self, answer: Answer | None, message: str, style: str) -> None:
        """Swap a streaming answer for a plain note (cancellation, failure)."""
        if answer is None:
            return
        history = self.query_one("#conversation", containers.VerticalScroll)
        await answer.remove()
        await history.mount(
            widgets.Static(text.Text(message), classes=f"message {style}")
        )

"""Offline checks for terminal chat defaults and conversation history."""

import io
import json
import unittest
from unittest import mock

from rich.console import Console

from ai import cli, client


class ChatTest(unittest.TestCase):
    def test_chat_defaults_and_noninteractive_output(self):
        cases = [
            ([], True, True, True),
            (["--no-chat"], True, True, False),
            (["-c"], True, True, True),
            (["--plain"], True, True, False),
            ([], False, True, False),
            ([], True, False, False),
        ]
        for flags, stdin_tty, stdout_tty, expected in cases:
            with self.subTest(flags=flags, stdin=stdin_tty, stdout=stdout_tty):
                stdin = io.StringIO("")
                stdout = io.StringIO()
                with (
                    mock.patch.object(stdin, "isatty", return_value=stdin_tty),
                    mock.patch.object(stdout, "isatty", return_value=stdout_tty),
                    mock.patch("sys.stdin", stdin),
                    mock.patch("sys.stdout", stdout),
                    mock.patch.object(
                        cli, "load_config", return_value={"model": "test"}
                    ),
                    mock.patch.object(cli, "stream_prompt") as stream,
                    mock.patch("ai.tui.ChatApp") as tui,
                ):
                    cli.cli.main([*flags, "hello"], standalone_mode=False)
                self.assertEqual(tui.called, expected)
                if not expected:
                    self.assertFalse(stream.call_args.kwargs["chat"])

    def test_bare_command_opens_tui(self):
        stdin = io.StringIO()
        stdout = io.StringIO()
        with (
            mock.patch.object(stdin, "isatty", return_value=True),
            mock.patch.object(stdout, "isatty", return_value=True),
            mock.patch("sys.stdin", stdin),
            mock.patch("sys.stdout", stdout),
            mock.patch.object(cli, "load_config", return_value={"model": "test"}),
            mock.patch("ai.tui.ChatApp") as tui,
        ):
            cli.cli.main([], standalone_mode=False)
        tui.assert_called_once_with("test", None, "auto", "", True)
        tui.return_value.run.assert_called_once()

    def test_stream_callback_continues_after_tools_without_terminal_output(self):
        tool = {
            "index": 0,
            "id": "call1",
            "function": {"name": "search", "arguments": '{"query": "test"}'},
        }
        responses = []
        for delta in [{"tool_calls": [tool]}, {"content": "result"}]:
            response = mock.MagicMock()
            response.status_code = 200
            response.iter_lines.return_value = [
                "data: " + json.dumps({"choices": [{"delta": delta}]}),
                "data: [DONE]",
            ]
            responses.append(response)
        http = mock.MagicMock()
        http.stream.return_value.__enter__.side_effect = responses
        callback = mock.Mock()
        messages = []
        output = io.StringIO()
        with (
            mock.patch.object(client.httpx, "Client") as factory,
            mock.patch.object(client, "execute_tool", return_value="found"),
            mock.patch("sys.stdout", output),
        ):
            factory.return_value.__enter__.return_value = http
            result = client._do_stream(
                "url",
                {"model": "test"},
                {},
                messages,
                Console(file=io.StringIO()),
                str,
                on_text=callback,
            )
        self.assertEqual(result[0], "result")
        callback.assert_called_once_with("result")
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(output.getvalue(), "")

    def test_followup_keeps_conversation_history(self):
        body = {}
        with (
            mock.patch.object(client, "_build_request", return_value=("url", {}, body)),
            mock.patch.object(client, "_print_banner"),
            mock.patch.object(
                client, "_do_stream", return_value=("answer", 0, 0, 0)
            ) as stream,
            mock.patch.object(client.Prompt, "ask", side_effect=["tell me more", "q"]),
        ):
            client.stream_prompt("hello", "test", chat=True)
        self.assertEqual(stream.call_count, 2)
        self.assertEqual(
            body["messages"][1:],
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "tell me more"},
            ],
        )

    def test_stream_clears_thinking_and_keeps_complete_answer(self):
        for chunks in [[], ["hello"], ["x" * 900, " final tail"]]:
            with self.subTest(chunks=chunks):
                output = io.StringIO()
                response = mock.MagicMock()
                response.status_code = 200
                response.iter_lines.return_value = [
                    "data: " + json.dumps({"choices": [{"delta": {"content": text}}]})
                    for text in chunks
                ] + ["data: [DONE]"]
                http = mock.MagicMock()
                http.stream.return_value.__enter__.return_value = response
                with (
                    mock.patch.object(client, "_PLAIN", False),
                    mock.patch.object(client, "_print_stats"),
                    mock.patch.object(client.httpx, "Client") as factory,
                ):
                    factory.return_value.__enter__.return_value = http
                    result = client._do_stream(
                        "url",
                        {"model": "test"},
                        {},
                        [],
                        Console(file=output, width=1200, color_system=None),
                        lambda text: text,
                    )
                self.assertEqual(result[0], "".join(chunks))
                self.assertNotIn("thinking", output.getvalue())
                if chunks:
                    self.assertIn("".join(chunks), output.getvalue())

    def test_compact_heading_and_suppression(self):
        for plain, show, expected in [
            (False, True, "✦ brain snacks\n"),
            (True, True, ""),
            (False, False, ""),
        ]:
            output = io.StringIO()
            with (
                mock.patch.object(client, "_PLAIN", plain),
                mock.patch.object(client, "_SHOW_BANNER", show),
            ):
                client._print_banner(Console(file=output, color_system=None))
            self.assertEqual(output.getvalue(), expected)


if __name__ == "__main__":
    unittest.main()

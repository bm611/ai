"""Exercise the full-screen chat without network requests."""

import copy
import unittest
from unittest import mock

from textual import widgets

from ai.tui import ChatApp


class TuiTest(unittest.IsolatedAsyncioTestCase):
    async def test_conversation_commands_and_streaming(self):
        requests = []

        def stream(url, body, headers, messages, console, renderer, on_text):
            requests.append(copy.deepcopy(messages))
            on_text("A streamed")
            return "A streamed answer.", "test", "", {"completion_tokens": 4}

        chat = ChatApp("test-model")
        with (
            mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}),
            mock.patch("ai.client._do_stream", side_effect=stream),
        ):
            async with chat.run_test(size=(80, 24)) as pilot:
                prompt = chat.query_one(widgets.Input)
                self.assertIs(chat.focused, prompt)
                self.assertFalse(chat.query_one("#conversation").display)
                prompt.value = "hello"
                await pilot.press("enter")
                await chat.workers.wait_for_complete()
                await pilot.pause()
                self.assertFalse(chat.busy)
                self.assertIn("4 tokens", str(chat.query_one("#status").render()))
                self.assertNotIn("thinking", str(chat.query_one("#status").render()))
                prompt.value = "tell me more"
                await pilot.press("enter")
                await chat.workers.wait_for_complete()
                self.assertEqual(
                    requests[1][1:],
                    [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "A streamed answer."},
                        {"role": "user", "content": "tell me more"},
                    ],
                )
                prompt.value = "/clear"
                await pilot.press("enter")
                self.assertEqual(len(chat.messages), 1)
                self.assertFalse(chat.query_one("#conversation").display)
                prompt.value = "/help"
                await pilot.press("enter")
                self.assertTrue(chat.query_one("#conversation").display)
                await pilot.resize_terminal(42, 12)
                await pilot.pause()
                self.assertTrue(prompt.region.bottom <= 12)
                prompt.value = "/quit"
                await pilot.press("enter")

    async def test_request_error_leaves_input_usable(self):
        chat = ChatApp("test-model", initial_prompt="hello")
        with mock.patch.dict("os.environ", {}, clear=True):
            async with chat.run_test() as pilot:
                await chat.workers.wait_for_complete()
                await pilot.pause()
                self.assertFalse(chat.busy)
                self.assertEqual(len(chat.messages), 1)
                self.assertIs(chat.focused, chat.query_one(widgets.Input))
                self.assertIn("error", str(chat.query_one("#status").render()))
                await pilot.press("ctrl+c")


if __name__ == "__main__":
    unittest.main()

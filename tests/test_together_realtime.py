import json
import unittest
from types import SimpleNamespace

import aiohttp

from together_realtime import (
    TogetherRealtimeError,
    TogetherRealtimeSettings,
    TogetherRealtimeTranscriber,
)


class FakeWebSocket:
    def __init__(self, payloads):
        self.messages = [
            SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps(payload),
            )
            for payload in payloads
        ]
        self.sent = []
        self.closed = False

    async def send_json(self, payload):
        self.sent.append(payload)

    async def receive(self):
        if not self.messages:
            raise AssertionError("No fake WebSocket messages remain")
        return self.messages.pop(0)

    async def close(self):
        self.closed = True

    def exception(self):
        return None


class TogetherRealtimeTests(unittest.TestCase):
    def test_persistent_connection_streams_deltas_and_commits_each_chunk(self):
        websocket = FakeWebSocket(
            [
                {"type": "session.created"},
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "delta": "hello",
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "hello world",
                },
                {
                    "type": "conversation.item.input_audio_transcription.delta",
                    "delta": "hello world second",
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "hello world second part",
                },
            ]
        )
        connections = []

        async def connect(url, headers, params):
            connections.append((url, headers, params))
            return websocket

        deltas = []
        transcriber = TogetherRealtimeTranscriber(
            TogetherRealtimeSettings(
                api_key="test-key",
                frame_bytes=4096,
            ),
            connector=connect,
        )
        try:
            first = transcriber.transcribe(
                b"\x00\x00" * 2500,
                on_delta=deltas.append,
            )
            second = transcriber.transcribe(
                b"\x00\x00" * 2500,
                on_delta=deltas.append,
            )
        finally:
            transcriber.close()

        self.assertEqual(first.text, "hello world")
        self.assertEqual(second.text, "second part")
        self.assertEqual(deltas, ["hello", "second"])
        self.assertEqual(len(connections), 1)
        self.assertEqual(connections[0][1]["Authorization"], "Bearer test-key")
        self.assertEqual(connections[0][2]["turn_detection"], "none")
        self.assertEqual(connections[0][2]["language"], "auto")
        self.assertEqual(
            sum(item["type"] == "input_audio_buffer.commit" for item in websocket.sent),
            2,
        )
        self.assertEqual(
            sum(item["type"] == "input_audio_buffer.append" for item in websocket.sent),
            4,
        )
        self.assertTrue(websocket.closed)

    def test_manual_language_is_sent_in_connection_parameters(self):
        websocket = FakeWebSocket(
            [
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "こんにちは",
                }
            ]
        )
        captured_params = {}

        async def connect(url, headers, params):
            del url, headers
            captured_params.update(params)
            return websocket

        transcriber = TogetherRealtimeTranscriber(
            TogetherRealtimeSettings(api_key="test-key"),
            language="ja",
            connector=connect,
        )
        try:
            result = transcriber.transcribe(b"\x00\x00")
        finally:
            transcriber.close()

        self.assertEqual(result.text, "こんにちは")
        self.assertEqual(captured_params["language"], "ja")

    def test_failed_event_surfaces_service_message(self):
        websocket = FakeWebSocket(
            [
                {
                    "type": "conversation.item.input_audio_transcription.failed",
                    "error": {
                        "code": "invalid_audio",
                        "message": "Audio format is invalid",
                    },
                }
            ]
        )

        async def connect(url, headers, params):
            del url, headers, params
            return websocket

        transcriber = TogetherRealtimeTranscriber(
            TogetherRealtimeSettings(api_key="test-key"),
            connector=connect,
        )
        try:
            with self.assertRaisesRegex(
                TogetherRealtimeError,
                "Audio format is invalid",
            ):
                transcriber.transcribe(b"\x00\x00")
        finally:
            transcriber.close()


if __name__ == "__main__":
    unittest.main()

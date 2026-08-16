import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from media_audio_stream import MediaAudioSource, _ffmpeg_input_arguments
from youtube_srt import get_youtube_audio_stream_source


class FakeCookieJar:
    def get_cookie_header(self, url):
        return "session=stream-cookie" if "googlevideo.test" in url else None


class FakeYoutubeDL:
    calls = 0
    options_seen = []

    def __init__(self, options):
        self.options = options
        self.__class__.options_seen.append(options)
        self.cookiejar = (
            FakeCookieJar() if "cookiefile" in options else SimpleNamespace(
                get_cookie_header=lambda url: None
            )
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def extract_info(self, url, download=False):
        del url, download
        self.__class__.calls += 1
        suffix = "bad" if self.__class__.calls == 1 else "good"
        return {
            "id": "video-id",
            "url": f"https://googlevideo.test/{suffix}",
            "http_headers": {
                "User-Agent": "yt-dlp-agent",
                "Accept-Language": "en-US",
            },
        }


class YoutubeAudioStreamTests(unittest.TestCase):
    def test_stream_resolver_passes_cookies_and_skips_forbidden_candidate(self):
        FakeYoutubeDL.calls = 0
        FakeYoutubeDL.options_seen = []
        fake_module = SimpleNamespace(YoutubeDL=FakeYoutubeDL)
        with tempfile.TemporaryDirectory() as directory:
            cookies_file = Path(directory) / "cookies.txt"
            cookies_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
            with (
                mock.patch.dict(sys.modules, {"yt_dlp": fake_module}),
                mock.patch(
                    "youtube_srt.decode_media_prefix",
                    side_effect=[RuntimeError("Server returned 403 Forbidden"), b"\0\0"],
                ) as validate,
            ):
                source, info = get_youtube_audio_stream_source(
                    "https://www.youtube.com/watch?v=video-id",
                    cookies_file,
                )

        self.assertEqual(FakeYoutubeDL.calls, 2)
        self.assertNotIn("cookiefile", FakeYoutubeDL.options_seen[0])
        self.assertIn("cookiefile", FakeYoutubeDL.options_seen[1])
        self.assertEqual(source.location, "https://googlevideo.test/good")
        self.assertEqual(source.headers["Cookie"], "session=stream-cookie")
        self.assertEqual(source.headers["User-Agent"], "yt-dlp-agent")
        self.assertEqual(info["id"], "video-id")
        self.assertEqual(validate.call_count, 2)

    def test_ffmpeg_headers_remove_line_breaks(self):
        arguments = _ffmpeg_input_arguments(
            MediaAudioSource(
                location="https://googlevideo.test/audio",
                headers={
                    "User-Agent": "agent\r\nInjected: no",
                    "Cookie": "session=value\r\nInjected: no",
                },
            )
        )

        self.assertIn("agentInjected: no", arguments)
        headers = arguments[arguments.index("-headers") + 1]
        self.assertIn("Cookie: session=valueInjected: no\r\n", headers)
        self.assertNotIn("\r\nInjected", headers)


if __name__ == "__main__":
    unittest.main()

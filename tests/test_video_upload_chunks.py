import asyncio
import tempfile
import unittest
from pathlib import Path

from starlette.requests import Request

from youtube_live import (
    VideoUploadBatchSessionRequest,
    assemble_video_upload_chunks,
    completed_video_upload_chunks,
    create_youtube_live_router,
    expected_video_upload_chunk_bytes,
    video_upload_chunk_count,
)
from youtube_srt import YOUTUBE_AUDIO_FORMAT


class VideoUploadChunkTests(unittest.TestCase):
    def test_chunk_boundaries_cover_file_exactly(self) -> None:
        self.assertEqual(video_upload_chunk_count(10, 4), 3)
        self.assertEqual(
            [expected_video_upload_chunk_bytes(10, 4, index) for index in range(3)],
            [4, 4, 2],
        )
        with self.assertRaises(ValueError):
            expected_video_upload_chunk_bytes(10, 4, 3)

    def test_completed_chunks_ignore_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            chunks_dir = Path(directory)
            (chunks_dir / "00000000.part").write_bytes(b"abcd")
            (chunks_dir / "00000001.part").write_bytes(b"x")

            completed, uploaded_bytes = completed_video_upload_chunks(
                chunks_dir,
                10,
                4,
            )

        self.assertEqual(completed, [0])
        self.assertEqual(uploaded_bytes, 4)

    def test_assembly_preserves_chunk_order_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks_dir = root / "chunks"
            chunks_dir.mkdir()
            for index, value in enumerate((b"abcd", b"efgh", b"ij")):
                (chunks_dir / f"{index:08d}.part").write_bytes(value)
            output_path = root / "input.mp4"

            assemble_video_upload_chunks(chunks_dir, output_path, 10, 4)

            self.assertEqual(output_path.read_bytes(), b"abcdefghij")

    def test_default_youtube_audio_selector_caps_bitrate_first(self) -> None:
        self.assertTrue(YOUTUBE_AUDIO_FORMAT.startswith("bestaudio[abr<=96]"))

    def test_resumable_upload_endpoint_reports_committed_bytes(self) -> None:
        async def run_test() -> None:
            router = create_youtube_live_router(object(), Path.cwd(), None)

            def endpoint(path: str, method: str):
                for route in router.routes:
                    if route.path == path and method in (route.methods or set()):
                        return route.endpoint
                raise AssertionError(f"Route not found: {method} {path}")

            async def empty_receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            create_request = Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/api/video-upload/sessions/batch",
                    "headers": [],
                },
                empty_receive,
            )
            created = await endpoint(
                "/api/video-upload/sessions/batch",
                "POST",
            )(
                create_request,
                VideoUploadBatchSessionRequest(
                    files=[
                        {
                            "filename": "sample.mp4",
                            "size_bytes": 6,
                            "content_type": "video/mp4",
                        }
                    ]
                ),
            )
            session = created["sessions"][0]
            messages = [
                {
                    "type": "http.request",
                    "body": b"abcdef",
                    "more_body": False,
                }
            ]

            async def chunk_receive():
                return messages.pop(0)

            chunk_request = Request(
                {
                    "type": "http",
                    "method": "PUT",
                    "path": "",
                    "headers": [(b"content-length", b"6")],
                },
                chunk_receive,
            )
            progress = await endpoint(
                "/api/video-upload/sessions/{upload_id}/chunks/{chunk_index}",
                "PUT",
            )(
                chunk_request,
                session["upload_id"],
                0,
                session["upload_token"],
            )
            self.assertEqual(progress["completed_chunks"], [0])
            self.assertEqual(progress["uploaded_bytes"], 6)

            status = await endpoint(
                "/api/video-upload/sessions/{upload_id}",
                "GET",
            )(
                session["upload_id"],
                session["upload_token"],
            )
            self.assertEqual(status["uploaded_bytes"], 6)
            await endpoint(
                "/api/video-upload/sessions/{upload_id}",
                "DELETE",
            )(
                session["upload_id"],
                session["upload_token"],
            )

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from apps.dataset_builder import video_source as vs
from apps.dataset_builder.builder import InputMediaCollection, build_dataset, collect_input_media


def _fmt(format_id, height, width=1920, bitrate=1000, fps=30, codec="avc1.640028", ext="mp4"):
    return {
        "format_id": format_id,
        "height": height,
        "width": width,
        "vbr": bitrate,
        "fps": fps,
        "vcodec": codec,
        "ext": ext,
        "protocol": "https",
    }


class VideoSourceSelectionTestCase(unittest.TestCase):
    def test_selects_1080p_among_lower_formats(self):
        candidates = vs._rank_video_formats(
            [
                _fmt("360", 360, width=640),
                _fmt("720", 720, width=1280),
                _fmt("1080", 1080, width=1920),
            ]
        )
        attempts = vs._fallback_candidates(candidates, 720, True)

        self.assertEqual(attempts[0].format_id, "1080")

    def test_prefers_higher_bitrate_at_same_resolution(self):
        candidates = vs._rank_video_formats(
            [
                _fmt("1080-low", 1080, bitrate=1200),
                _fmt("1080-high", 1080, bitrate=4500),
            ]
        )

        self.assertEqual(candidates[0].format_id, "1080-high")

    @patch("apps.dataset_builder.video_source.probe_video")
    @patch("apps.dataset_builder.video_source._is_readable_video", return_value=True)
    @patch("apps.dataset_builder.video_source._download_format")
    @patch("apps.dataset_builder.video_source._fetch_video_info")
    def test_falls_back_when_best_download_fails(
        self, info_mock, download_mock, _readable_mock, probe_mock
    ):
        info_mock.return_value = {
            "formats": [
                _fmt("720", 720, width=1280),
                _fmt("1080", 1080, width=1920),
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            fallback_file = Path(directory) / "fallback.mp4"
            fallback_file.write_bytes(b"video")
            download_mock.side_effect = [RuntimeError("403"), fallback_file]
            probe_mock.return_value = {
                "selected_format_id": "720",
                "width": 1280,
                "height": 720,
                "fps": 30.0,
                "codec": "h264",
                "bitrate": 1200,
                "duration": 5.0,
                "verified": True,
                "verification_tool": "ffprobe",
            }

            result = vs.download_best_video_source(
                "https://example.test/video",
                Path(directory),
            )

        self.assertEqual(result.source_media["selected_format_id"], "720")
        self.assertIn("quality_fallback_used", result.source_media["warnings"])

    @patch("apps.dataset_builder.video_source.subprocess.run")
    def test_probe_video_reads_ffprobe_result(self, run_mock):
        run_mock.return_value = Mock(
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "width": 1920,
                            "height": 1080,
                            "avg_frame_rate": "30000/1001",
                            "codec_name": "h264",
                            "bit_rate": "5000000",
                        }
                    ],
                    "format": {"duration": "42.5", "bit_rate": "5000000"},
                }
            ),
        )

        result = vs.probe_video(Path("video.mp4"), selected_format_id="137")

        self.assertTrue(result["verified"])
        self.assertEqual(result["selected_format_id"], "137")
        self.assertEqual(result["width"], 1920)
        self.assertEqual(result["height"], 1080)
        self.assertAlmostEqual(result["fps"], 29.97002997)
        self.assertEqual(result["codec"], "h264")

    def test_source_only_480p_records_warning(self):
        candidates = vs._rank_video_formats([_fmt("480", 480, width=854)])
        attempts = vs._fallback_candidates(candidates, 720, True)
        media = vs._source_media_payload(
            attempts[0],
            {
                "selected_format_id": "480",
                "width": 854,
                "height": 480,
                "fps": 25.0,
                "codec": "h264",
                "bitrate": 900,
                "duration": 10.0,
                "verified": True,
                "verification_tool": "ffprobe",
            },
            min_video_height=720,
            fallback_used=False,
            url="https://example.test/video",
        )

        self.assertIn("source_video_resolution_low", media["warnings"][0])
        self.assertFalse(media["transcoded"])
        self.assertIsNone(media["transcode"])

    @patch("apps.dataset_builder.builder.create_portrait_report")
    @patch("apps.dataset_builder.builder.collect_input_media")
    def test_dataset_and_summary_store_source_media(self, collect_mock, report_mock):
        source_media = {
            "requested_quality": "best_available",
            "selected_format_id": "137",
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "codec": "h264",
            "bitrate": 5000000,
            "duration": 12.0,
            "download_strategy": "bestvideo",
            "verified": True,
            "warnings": [],
        }
        report_mock.return_value = {
            "schema_version": 3,
            "id": "PFR-test",
            "uuid": "12345678-1234-5678-1234-567812345678",
            "quality": {
                "status": "passed",
                "issues": [],
                "issue_codes": [],
                "metrics": {
                    "face_width_px": 420.0,
                    "face_height_px": 560.0,
                    "face_area_ratio": 0.19,
                },
            },
            "lic_core": {},
            "morphology": {},
            "measurements": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame000001.jpg"
            image.write_bytes(b"image")
            collect_mock.return_value = InputMediaCollection([image], source_media, [])

            summary = build_dataset("https://example.test/video", str(root / "dataset"))

            dataset_dir = Path(summary["dataset_dir"])
            dataset = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
            summary_json = json.loads((dataset_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(dataset["source_media"]["selected_format_id"], "137")
        self.assertEqual(summary_json["source_media"]["height"], 1080)
        self.assertEqual(dataset["face_effective_resolution"]["median_face_width_px"], 420.0)
        self.assertEqual(summary["face_effective_resolution"]["p10_face_height_px"], 560.0)

    @patch("apps.dataset_builder.builder._extract_video_frames")
    @patch("apps.dataset_builder.builder._local_video_source_media")
    def test_local_video_remains_supported(self, media_mock, extract_mock):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "local.mp4"
            video.write_bytes(b"video")
            frame = root / "frame000000.jpg"
            frame.write_bytes(b"frame")
            media_mock.return_value = {
                "requested_quality": "local_file",
                "selected_format_id": "local",
                "width": 1280,
                "height": 720,
                "download_strategy": "local_file",
                "warnings": [],
            }
            extract_mock.return_value = [frame]

            collection = collect_input_media(str(video), str(root / "frames"))

        self.assertEqual(collection.images, [frame])
        self.assertEqual(collection.source_media["download_strategy"], "local_file")


if __name__ == "__main__":
    unittest.main()

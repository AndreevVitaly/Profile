import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from apps.dataset_builder.video_sources import VideoSourceManager
from apps.dataset_builder.video_sources.dailymotion import DailymotionAdapter
from apps.dataset_builder.video_sources.direct_media import DirectMediaAdapter
from apps.dataset_builder.video_sources.generic import GenericVideoAdapter
from apps.dataset_builder.video_sources.ok import OKVideoAdapter
from apps.dataset_builder.video_sources.twitch import TwitchVODAdapter
from apps.dataset_builder.video_sources.vk import VKVideoAdapter


class ExpandedSourceDetectionTestCase(unittest.TestCase):
    def setUp(self):
        self.manager = VideoSourceManager([
            VKVideoAdapter,
            OKVideoAdapter,
            DailymotionAdapter,
            TwitchVODAdapter,
            GenericVideoAdapter,
            DirectMediaAdapter,
        ])

    def test_detects_additional_sites(self):
        cases = {
            "https://vk.com/video-1_2": VKVideoAdapter,
            "https://ok.ru/video/123": OKVideoAdapter,
            "https://dai.ly/abc": DailymotionAdapter,
            "https://www.twitch.tv/videos/123": TwitchVODAdapter,
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertIsInstance(self.manager.detect(source), expected)

    def test_unknown_domain_still_uses_generic_first(self):
        self.assertIsInstance(
            self.manager.detect("https://unknown.example/watch/42"),
            GenericVideoAdapter,
        )


class DirectMediaAdapterTestCase(unittest.TestCase):
    class Response:
        status = 200

        def __init__(self):
            self.headers = Message()
            self.headers["Content-Type"] = "video/mp4"
            self.headers["Content-Length"] = "5"
            self._chunks = iter((b"video", b""))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return next(self._chunks)

    @patch("apps.dataset_builder.video_sources.direct_media.probe_video")
    @patch("apps.dataset_builder.video_sources.direct_media.urlopen")
    def test_downloads_direct_video_without_domain_rules(self, urlopen_mock, probe_mock):
        urlopen_mock.return_value = self.Response()
        probe_mock.return_value = {
            "width": 1280,
            "height": 720,
            "fps": 25.0,
            "duration": 10.0,
            "codec": "h264",
            "verified": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            asset = DirectMediaAdapter().download(
                "https://unknown.example/download?id=42",
                Path(directory),
            )
            self.assertEqual(asset.downloaded_file.read_bytes(), b"video")

        self.assertEqual(asset.download_strategy, "direct_http")
        self.assertEqual(asset.source_type, "direct_media")
        self.assertTrue(asset.verified)


if __name__ == "__main__":
    unittest.main()
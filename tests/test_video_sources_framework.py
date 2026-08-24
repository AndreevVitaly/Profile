import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from apps.dataset_builder import video_source as legacy_video_source
from apps.dataset_builder.builder import InputMediaCollection, build_dataset
from apps.dataset_builder.video_sources import (
    UnifiedVideoAsset,
    VideoSourceAdapter,
    VideoSourceError,
    VideoSourceManager,
    VideoSourceProbe,
)
from apps.dataset_builder.video_sources.generic import GenericVideoAdapter
from apps.dataset_builder.video_sources.local_file import LocalVideoAdapter
from apps.dataset_builder.video_sources.rutube import RuTubeAdapter
from apps.dataset_builder.video_sources.vimeo import VimeoAdapter
from apps.dataset_builder.video_sources.yandex import YandexVideoAdapter
from apps.dataset_builder.video_sources.youtube import YouTubeAdapter


class VideoSourceDetectionTestCase(unittest.TestCase):
    def setUp(self):
        self.manager = VideoSourceManager([
            LocalVideoAdapter, YouTubeAdapter, YandexVideoAdapter,
            RuTubeAdapter, VimeoAdapter, GenericVideoAdapter,
        ])

    def test_detects_local_video(self):
        self.assertIsInstance(self.manager.detect("C:/video/example.mp4"), LocalVideoAdapter)

    def test_detects_youtube(self):
        self.assertIsInstance(self.manager.detect("https://youtu.be/abc"), YouTubeAdapter)
        self.assertIsInstance(
            self.manager.detect("https://www.youtube.com/watch?v=abc"), YouTubeAdapter
        )

    def test_detects_yandex_video(self):
        self.assertIsInstance(
            self.manager.detect("https://yandex.ru/video/preview/123"),
            YandexVideoAdapter,
        )
        self.assertIsInstance(
            self.manager.detect("https://ya.ru/video/preview/123"),
            YandexVideoAdapter,
        )

    @patch("apps.dataset_builder.video_sources.yandex.urlopen")
    def test_yandex_preview_extracts_embedded_video_url(self, urlopen_mock):
        response = Mock()
        response.headers.get_content_charset.return_value = "utf-8"
        response.read.return_value = (
            b'<iframe src="player#counters=%7B%22videoUrl%22%3A%22'
            b'http%3A%2F%2Fvk.com%2Fvideo-1_2%22%7D">'
        )
        urlopen_mock.return_value.__enter__.return_value = response

        embedded = YandexVideoAdapter._embedded_source(
            "https://ya.ru/video/preview/123"
        )

        self.assertEqual(embedded, "http://vk.com/video-1_2")

    @patch.object(YandexVideoAdapter, "_fetch_page")
    def test_yandex_extracts_hls_stream(self, fetch_page):
        fetch_page.return_value = (
            r'{"hls":"https:\/\/vkvd450.okcdn.ru\/video.m3u8?'
            r'ms=185.180.203.232\u0026id=42"}'
        )

        stream = YandexVideoAdapter._stream_source("https://vkvideo.ru/embed")

        self.assertEqual(
            stream,
            "https://vkvd450.okcdn.ru/video.m3u8?ms=185.180.203.232&id=42",
        )

    def test_yandex_uses_signed_media_ip_for_okcdn(self):
        stream, headers, no_check = YandexVideoAdapter._stream_transport(
            "https://vkvd450.okcdn.ru/video.m3u8?ms=185.180.203.232&id=42"
        )

        self.assertTrue(stream.startswith("https://185.180.203.232/"))
        self.assertEqual(headers["Host"], "vkvd450.okcdn.ru")
        self.assertTrue(no_check)

    @patch.object(YandexVideoAdapter, "_stream_source", return_value="stream")
    @patch.object(YandexVideoAdapter, "_embedded_source")
    def test_yandex_dns_failure_enters_network_pause(self, embedded, _stream):
        embedded.side_effect = [RuntimeError("getaddrinfo failed"), "embed"]
        network_wait = Mock(return_value=True)

        result = YandexVideoAdapter._resolve_stream(
            "https://ya.ru/video/preview/123",
            {"network_wait": network_wait},
        )

        self.assertEqual(result, ("embed", "stream"))
        network_wait.assert_called_once()
    def test_detects_rutube(self):
        self.assertIsInstance(
            self.manager.detect("https://rutube.ru/video/abc"), RuTubeAdapter
        )

    def test_detects_vimeo(self):
        self.assertIsInstance(self.manager.detect("https://vimeo.com/123"), VimeoAdapter)

    def test_unknown_http_url_uses_generic_adapter(self):
        self.assertIsInstance(
            self.manager.detect("https://media.example.test/video"), GenericVideoAdapter
        )

    def test_invalid_source_has_readable_diagnostic(self):
        with self.assertRaises(VideoSourceError) as caught:
            self.manager.detect("not-a-video-source")
        self.assertEqual(caught.exception.source_name, "Неизвестный источник")


class HlsFormatRankingTestCase(unittest.TestCase):
    def test_unknown_codec_with_video_dimensions_is_accepted(self):
        formats = legacy_video_source._rank_video_formats([{
            "format_id": "hls-1080",
            "width": 1920,
            "height": 1080,
            "vcodec": None,
            "protocol": "m3u8_native",
            "ext": "mp4",
        }])

        self.assertEqual(len(formats), 1)
        self.assertEqual(formats[0].codec, "unknown")

class YtDlpOptionForwardingTestCase(unittest.TestCase):
    @patch("apps.dataset_builder.video_sources.yt_dlp_adapter.legacy.download_best_video_source")
    def test_download_forwards_headers_and_tls_option(self, download_mock):
        download_mock.return_value = Mock(
            path=Path("downloads/video.mp4"),
            source_media={"verified": True, "selected_format_id": "hls"},
        )

        YouTubeAdapter().download(
            "https://youtu.be/example",
            Path("downloads"),
            http_headers={"Host": "media.example"},
            no_check_certificates=True,
        )

        kwargs = download_mock.call_args.kwargs
        self.assertEqual(kwargs["http_headers"], {"Host": "media.example"})
        self.assertTrue(kwargs["no_check_certificates"])

class LocalVideoAdapterTestCase(unittest.TestCase):
    @patch("apps.dataset_builder.video_sources.local_file.probe_video")
    def test_returns_unified_local_asset(self, probe_mock):
        probe_mock.return_value = {
            "width": 1920, "height": 1080, "fps": 30.0,
            "duration": 12.5, "codec": "h264", "bitrate": 4_000_000,
            "verified": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "source.mp4"
            video.write_bytes(b"video")
            asset = VideoSourceManager().resolve(video, Path(directory) / "downloads")

        self.assertEqual(asset.source_type, "local_file")
        self.assertEqual(asset.adapter, "LocalVideoAdapter")
        self.assertEqual(asset.width, 1920)
        self.assertTrue(asset.verified)
        self.assertEqual(asset.video_source_metadata()["download_strategy"], "local_file")


class VideoSourceFallbackTestCase(unittest.TestCase):
    class FailingAdapter(VideoSourceAdapter):
        source_type = "test"
        display_name = "Test source"
        priority = 10

        @classmethod
        def can_handle(cls, source):
            return source.startswith("https://test.example/")

        def probe(self, source, **options):
            raise RuntimeError("extractor changed")

        def download(self, source, destination, **options):
            raise RuntimeError("extractor changed")

    class WorkingFallback(VideoSourceAdapter):
        source_type = "generic"
        display_name = "Fallback"
        priority = -10
        fallback_adapter = True

        @classmethod
        def can_handle(cls, source):
            return source.startswith("https://")

        def probe(self, source, **options):
            return VideoSourceProbe(
                self.source_type, self.display_name, type(self).__name__, source
            )

        def download(self, source, destination, **options):
            return UnifiedVideoAsset(
                source_type=self.source_type,
                adapter=type(self).__name__,
                original_source=source,
                downloaded_file=Path(destination) / "video.mp4",
                verified=True,
            )

    def test_fallback_marks_asset(self):
        manager = VideoSourceManager([self.FailingAdapter, self.WorkingFallback])
        asset = manager.resolve("https://test.example/video", "downloads")
        self.assertTrue(asset.fallback_used)
        self.assertEqual(asset.adapter, "WorkingFallback")

    def test_all_failures_preserve_technical_details(self):
        manager = VideoSourceManager([self.FailingAdapter])
        with self.assertRaises(VideoSourceError) as caught:
            manager.resolve("https://test.example/video", "downloads")
        self.assertIn("extractor changed", caught.exception.technical_details())


class VideoSourceDatasetMetadataTestCase(unittest.TestCase):
    @patch("apps.dataset_builder.builder.create_portrait_report")
    @patch("apps.dataset_builder.builder.collect_input_media")
    def test_dataset_persists_video_source_provenance(self, collect_mock, report_mock):
        report_mock.return_value = {
            "schema_version": 3,
            "quality": {"status": "passed", "issues": [], "metrics": {}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame000001.jpg"
            image.write_bytes(b"image")
            video = root / "video.mp4"
            video.write_bytes(b"video")
            asset = UnifiedVideoAsset(
                source_type="youtube",
                adapter="YouTubeAdapter",
                original_source="https://youtube.com/watch?v=test",
                downloaded_file=video,
                width=1920,
                height=1080,
                fps=30.0,
                verified=True,
                download_strategy="yt-dlp",
                selected_stream="137",
            )
            collect_mock.return_value = InputMediaCollection(
                [image], asset.source_media_metadata(), [], None, asset
            )

            summary = build_dataset(
                asset.original_source,
                str(root / "dataset"),
            )
            dataset = json.loads(
                (Path(summary["dataset_dir"]) / "dataset.json").read_text(encoding="utf-8")
            )

        self.assertEqual(dataset["video_source"]["type"], "youtube")
        self.assertEqual(dataset["video_source"]["adapter"], "YouTubeAdapter")
        self.assertEqual(summary["video_source"]["selected_stream"], "137")

if __name__ == "__main__":
    unittest.main()
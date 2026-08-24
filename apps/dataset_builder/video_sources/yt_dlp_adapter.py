"""Reusable yt-dlp backend for site-specific adapters."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

from apps.dataset_builder import video_source as legacy

from .base import VideoSourceAdapter
from .models import UnifiedVideoAsset, VideoSourceProbe, VideoStream


class YtDlpVideoAdapter(VideoSourceAdapter):
    download_strategy = "yt-dlp"

    @classmethod
    @abstractmethod
    def can_handle(cls, source: str) -> bool:
        raise NotImplementedError

    def probe(self, source: str, **options: Any) -> VideoSourceProbe:
        info = legacy._fetch_video_info(
            source,
            log=options.get("log"),
            should_stop=options.get("should_stop"),
            http_headers=options.get("http_headers"),
            no_check_certificates=bool(options.get("no_check_certificates")),
        )
        streams = tuple(
            VideoStream(
                stream_id=item.format_id,
                width=item.width,
                height=item.height,
                fps=item.fps,
                codec=item.codec,
                bitrate=item.bitrate,
                ext=item.ext,
                raw=item.raw,
            )
            for item in legacy._rank_video_formats(info.get("formats") or [info])
        )
        return VideoSourceProbe(
            source_type=self.source_type,
            display_name=self.display_name,
            adapter=type(self).__name__,
            original_source=source,
            title=info.get("title"),
            streams=streams,
            metadata={"extractor": info.get("extractor")},
        )

    def download(
        self,
        source: str,
        destination: Path,
        **options: Any,
    ) -> UnifiedVideoAsset:
        accepted = {
            "video_quality",
            "min_video_height",
            "allow_quality_fallback",
            "log",
            "should_stop",
            "network_wait",
            "network_recovered",
            "http_headers",
            "no_check_certificates",
        }
        result = legacy.download_best_video_source(
            source,
            destination,
            **{key: value for key, value in options.items() if key in accepted},
        )
        media = dict(result.source_media)
        warnings = tuple(str(item) for item in media.pop("warnings", []))
        selected_stream = media.get("selected_format_id")
        return UnifiedVideoAsset(
            source_type=self.source_type,
            adapter=type(self).__name__,
            original_source=source,
            downloaded_file=result.path,
            width=int(media.get("width") or 0),
            height=int(media.get("height") or 0),
            fps=float(media.get("fps") or 0.0),
            duration=float(media.get("duration") or 0.0),
            codec=str(media.get("codec") or ""),
            bitrate=int(media.get("bitrate") or 0),
            verified=bool(media.get("verified")),
            download_strategy=self.download_strategy,
            selected_stream=str(selected_stream) if selected_stream is not None else None,
            warnings=warnings,
            extra=media,
        )

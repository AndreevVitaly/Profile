"""Local video adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apps.dataset_builder.video_source import VIDEO_SUFFIXES, probe_video

from .base import VideoSourceAdapter
from .models import UnifiedVideoAsset, VideoSourceProbe, VideoStream


class LocalVideoAdapter(VideoSourceAdapter):
    source_type = "local_file"
    display_name = "Local file"
    priority = 1000

    @classmethod
    def can_handle(cls, source: str) -> bool:
        value = str(source)
        path = Path(value).expanduser()
        return path.suffix.lower() in VIDEO_SUFFIXES and not value.lower().startswith(("http://", "https://"))

    def probe(self, source: str, **options: Any) -> VideoSourceProbe:
        value = str(source)
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Локальный видеофайл не найден: {path}")
        media = probe_video(path, selected_format_id="local")
        if not media.get("verified"):
            raise ValueError(f"Локальный видеофайл не удалось проверить: {path}")
        stream = VideoStream(
            stream_id="local",
            width=int(media.get("width") or 0),
            height=int(media.get("height") or 0),
            fps=float(media.get("fps") or 0.0),
            codec=str(media.get("codec") or ""),
            bitrate=int(media.get("bitrate") or 0),
            ext=path.suffix.lstrip("."),
            raw=media,
        )
        return VideoSourceProbe(
            source_type=self.source_type,
            display_name=self.display_name,
            adapter=type(self).__name__,
            original_source=str(path),
            title=path.name,
            streams=(stream,),
            metadata=media,
        )

    def download(
        self,
        source: str,
        destination: Path,
        **options: Any,
    ) -> UnifiedVideoAsset:
        probe = self.probe(source, **options)
        media = probe.metadata
        min_height = int(options.get("min_video_height", 720))
        warnings = ()
        height = int(media.get("height") or 0)
        if height and height < min_height:
            warnings = ("source_video_resolution_low",)
        return UnifiedVideoAsset(
            source_type=self.source_type,
            adapter=type(self).__name__,
            original_source=probe.original_source,
            downloaded_file=Path(probe.original_source),
            width=int(media.get("width") or 0),
            height=height,
            fps=float(media.get("fps") or 0.0),
            duration=float(media.get("duration") or 0.0),
            codec=str(media.get("codec") or ""),
            bitrate=int(media.get("bitrate") or 0),
            verified=bool(media.get("verified")),
            download_strategy="local_file",
            selected_stream="local",
            warnings=warnings,
            extra=media,
        )

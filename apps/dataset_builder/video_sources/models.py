"""Shared models for the Dataset Builder video source framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoStream:
    """A source stream exposed by an adapter probe."""

    stream_id: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = ""
    bitrate: int = 0
    ext: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class VideoSourceProbe:
    """Adapter-neutral source information available before download."""

    source_type: str
    display_name: str
    adapter: str
    original_source: str
    title: str | None = None
    streams: tuple[VideoStream, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnifiedVideoAsset:
    """A verified local video plus normalized provenance metadata."""

    source_type: str
    adapter: str
    original_source: str
    downloaded_file: Path
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    codec: str = ""
    bitrate: int = 0
    verified: bool = False
    download_strategy: str = ""
    selected_stream: str | None = None
    fallback_used: bool = False
    warnings: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def path(self) -> Path:
        """Compatibility alias used by the existing frame pipeline."""

        return self.downloaded_file

    def video_source_metadata(self) -> dict[str, Any]:
        url = self.original_source if self.source_type != "local_file" else None
        return {
            "type": self.source_type,
            "adapter": self.adapter,
            "url": url,
            "original_source": self.original_source,
            "downloaded_file": str(self.downloaded_file),
            "download_strategy": self.download_strategy,
            "selected_stream": self.selected_stream,
            "fallback_used": self.fallback_used,
            "verified": self.verified,
        }

    def source_media_metadata(self) -> dict[str, Any]:
        """Return the legacy-compatible media block enriched with provenance."""

        payload = dict(self.extra)
        payload.update(
            {
                "selected_format_id": self.selected_stream,
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "duration": self.duration,
                "codec": self.codec,
                "bitrate": self.bitrate,
                "download_strategy": self.download_strategy,
                "verified": self.verified,
                "warnings": list(self.warnings),
                "video_source": self.video_source_metadata(),
            }
        )
        return payload


@dataclass(frozen=True)
class VideoSourceFailure:
    adapter: str
    source_name: str
    technical_error: str


class VideoSourceError(RuntimeError):
    """User-facing failure with retained technical diagnostics."""

    def __init__(
        self,
        source: str,
        source_name: str,
        failures: list[VideoSourceFailure],
    ) -> None:
        self.source = source
        self.source_name = source_name
        self.failures = tuple(failures)
        self.attempts = "\n".join(
            f"- {item.source_name} ({item.adapter}): {item.technical_error}"
            for item in failures
        ) or "- Video Source Manager: подходящий способ загрузки не найден"
        recommendations = (
            "Рекомендации:\n"
            "• обновить yt-dlp;\n"
            "• попробовать другой источник;\n"
            "• скачать видео вручную;\n"
            "• использовать локальный файл."
        )
        super().__init__(
            "Не удалось получить информацию о видео.\n\n"
            f"Источник: {source_name}\n\n"
            "Причина: extractor не смог обработать источник.\n\n"
            f"{recommendations}"
        )

    def user_message(self) -> str:
        return f"{super().__str__()}\n\nИспробованные способы:\n{self.attempts}"
    def technical_details(self) -> str:
        return "\n".join(
            f"{item.adapter}: {item.technical_error}" for item in self.failures
        )

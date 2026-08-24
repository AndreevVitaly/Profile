"""Adapter contract and automatic adapter registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from pathlib import Path
from typing import Any, ClassVar

from .models import UnifiedVideoAsset, VideoSourceProbe, VideoStream


class VideoSourceAdapter(ABC):
    source_type: ClassVar[str] = "unknown"
    display_name: ClassVar[str] = "Unknown"
    priority: ClassVar[int] = 0
    fallback_adapter: ClassVar[bool] = False
    _registry: ClassVar[list[type["VideoSourceAdapter"]]] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None) and cls not in cls._registry:
            cls._registry.append(cls)

    @classmethod
    @abstractmethod
    def can_handle(cls, source: str) -> bool:
        """Return whether this adapter understands the source identifier."""

    @abstractmethod
    def probe(self, source: str, **options: Any) -> VideoSourceProbe:
        """Read source metadata without producing the final local asset."""

    def list_streams(self, probe: VideoSourceProbe) -> tuple[VideoStream, ...]:
        return probe.streams

    @abstractmethod
    def download(
        self,
        source: str,
        destination: Path,
        **options: Any,
    ) -> UnifiedVideoAsset:
        """Resolve the source into a verified local video asset."""

    def metadata(self, asset: UnifiedVideoAsset) -> dict[str, Any]:
        return asset.video_source_metadata()

    @classmethod
    def registered_adapters(cls) -> tuple[type["VideoSourceAdapter"], ...]:
        concrete = (item for item in cls._registry if not inspect.isabstract(item))
        return tuple(sorted(concrete, key=lambda item: item.priority, reverse=True))

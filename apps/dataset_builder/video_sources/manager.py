"""Adapter discovery, selection, fallback and diagnostics."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any

from .base import VideoSourceAdapter
from .models import UnifiedVideoAsset, VideoSourceError, VideoSourceFailure


class VideoSourceManager:
    def __init__(self, adapters: list[type[VideoSourceAdapter]] | None = None) -> None:
        if adapters is None:
            self._discover_adapters()
            adapters = list(VideoSourceAdapter.registered_adapters())
        self._adapter_types = sorted(adapters, key=lambda item: item.priority, reverse=True)

    @staticmethod
    def _discover_adapters() -> None:
        package = importlib.import_module(__package__)
        ignored = {"base", "manager", "models"}
        for module in pkgutil.iter_modules(package.__path__):
            if module.name.startswith("_") or module.name in ignored:
                continue
            importlib.import_module(f"{__package__}.{module.name}")

    @property
    def adapters(self) -> tuple[type[VideoSourceAdapter], ...]:
        return tuple(self._adapter_types)

    def detect(self, source: str) -> VideoSourceAdapter:
        source = str(source)
        for adapter_type in self._adapter_types:
            if adapter_type.fallback_adapter:
                continue
            if adapter_type.can_handle(source):
                return adapter_type()
        for adapter_type in self._adapter_types:
            if adapter_type.fallback_adapter and adapter_type.can_handle(source):
                return adapter_type()
        raise VideoSourceError(
            source,
            "Неизвестный источник",
            [VideoSourceFailure("VideoSourceManager", "Unknown", "No adapter matched")],
        )

    def source_label(self, source: str) -> str:
        return self.detect(source).display_name

    def resolve(
        self,
        source: str,
        destination: str | Path,
        **options: Any,
    ) -> UnifiedVideoAsset:
        source = str(source)
        primary = self.detect(source)
        candidates = [primary]
        candidates.extend(
            adapter_type()
            for adapter_type in self._adapter_types
            if adapter_type.fallback_adapter
            and adapter_type is not type(primary)
            and adapter_type.can_handle(source)
        )
        failures: list[VideoSourceFailure] = []
        for index, adapter in enumerate(candidates):
            try:
                asset = adapter.download(source, Path(destination), **options)
                if index == 0:
                    return asset
                return UnifiedVideoAsset(
                    **{
                        **asset.__dict__,
                        "fallback_used": True,
                    }
                )
            except Exception as error:  # adapters preserve diagnostics for the manager
                should_stop = options.get("should_stop")
                if should_stop and should_stop():
                    raise
                failures.append(
                    VideoSourceFailure(
                        type(adapter).__name__, adapter.display_name, str(error)
                    )
                )
                log = options.get("log")
                if log:
                    log(f"{type(adapter).__name__} failed: {error}")
        raise VideoSourceError(source, primary.display_name, failures)

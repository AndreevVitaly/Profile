from __future__ import annotations

from urllib.parse import urlparse

from .yt_dlp_adapter import YtDlpVideoAdapter


class GenericVideoAdapter(YtDlpVideoAdapter):
    source_type = "generic"
    display_name = "Generic URL"
    priority = -100
    fallback_adapter = True

    @classmethod
    def can_handle(cls, source: str) -> bool:
        parsed = urlparse(source)
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)

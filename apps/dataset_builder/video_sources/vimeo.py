from __future__ import annotations

from urllib.parse import urlparse

from .yt_dlp_adapter import YtDlpVideoAdapter


class VimeoAdapter(YtDlpVideoAdapter):
    source_type = "vimeo"
    display_name = "Vimeo"
    priority = 750

    @classmethod
    def can_handle(cls, source: str) -> bool:
        host = (urlparse(source).hostname or "").lower()
        return host == "vimeo.com" or host.endswith(".vimeo.com")

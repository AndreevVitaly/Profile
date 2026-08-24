from __future__ import annotations

from urllib.parse import urlparse

from .yt_dlp_adapter import YtDlpVideoAdapter


class RuTubeAdapter(YtDlpVideoAdapter):
    source_type = "rutube"
    display_name = "RuTube"
    priority = 800

    @classmethod
    def can_handle(cls, source: str) -> bool:
        host = (urlparse(source).hostname or "").lower()
        return host == "rutube.ru" or host.endswith(".rutube.ru")

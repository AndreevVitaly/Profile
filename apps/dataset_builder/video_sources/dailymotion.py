from __future__ import annotations

from urllib.parse import urlparse

from .yt_dlp_adapter import YtDlpVideoAdapter


class DailymotionAdapter(YtDlpVideoAdapter):
    source_type = "dailymotion"
    display_name = "Dailymotion"
    priority = 660

    @classmethod
    def can_handle(cls, source: str) -> bool:
        host = (urlparse(source).hostname or "").lower()
        return host in {"dai.ly", "dailymotion.com"} or host.endswith(".dailymotion.com")

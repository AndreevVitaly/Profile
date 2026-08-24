from __future__ import annotations

from urllib.parse import urlparse

from .yt_dlp_adapter import YtDlpVideoAdapter


class YouTubeAdapter(YtDlpVideoAdapter):
    source_type = "youtube"
    display_name = "YouTube"
    priority = 900

    @classmethod
    def can_handle(cls, source: str) -> bool:
        host = (urlparse(source).hostname or "").lower()
        return host == "youtu.be" or host.endswith("youtube.com")

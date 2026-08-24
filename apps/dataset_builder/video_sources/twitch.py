from __future__ import annotations

from urllib.parse import urlparse

from .yt_dlp_adapter import YtDlpVideoAdapter


class TwitchVODAdapter(YtDlpVideoAdapter):
    source_type = "twitch_vod"
    display_name = "Twitch VOD"
    priority = 640

    @classmethod
    def can_handle(cls, source: str) -> bool:
        parsed = urlparse(source)
        host = (parsed.hostname or "").lower()
        return (host == "twitch.tv" or host.endswith(".twitch.tv")) and "/videos/" in parsed.path

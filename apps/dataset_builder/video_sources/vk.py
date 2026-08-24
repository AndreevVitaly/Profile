from __future__ import annotations

from urllib.parse import urlparse

from .yt_dlp_adapter import YtDlpVideoAdapter


class VKVideoAdapter(YtDlpVideoAdapter):
    source_type = "vk_video"
    display_name = "VK Video"
    priority = 700

    @classmethod
    def can_handle(cls, source: str) -> bool:
        host = (urlparse(source).hostname or "").lower()
        return host in {"vk.com", "vkvideo.ru"} or host.endswith((".vk.com", ".vkvideo.ru"))

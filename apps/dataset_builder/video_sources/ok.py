from __future__ import annotations

from urllib.parse import urlparse

from .yt_dlp_adapter import YtDlpVideoAdapter


class OKVideoAdapter(YtDlpVideoAdapter):
    source_type = "ok_video"
    display_name = "OK Video"
    priority = 680

    @classmethod
    def can_handle(cls, source: str) -> bool:
        host = (urlparse(source).hostname or "").lower()
        return host in {"ok.ru", "odnoklassniki.ru"} or host.endswith((".ok.ru", ".odnoklassniki.ru"))

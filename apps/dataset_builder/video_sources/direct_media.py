"""Fallback adapter for direct HTTP(S) video file URLs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from apps.dataset_builder.video_source import VIDEO_SUFFIXES, probe_video

from .base import VideoSourceAdapter
from .models import UnifiedVideoAsset, VideoSourceProbe, VideoStream


VIDEO_CONTENT_TYPES = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/x-matroska": ".mkv",
    "video/mpeg": ".mpeg",
    "video/x-m4v": ".m4v",
    "video/mp2t": ".ts",
    "video/x-flv": ".flv",
    "video/ogg": ".ogv",
    "application/octet-stream": ".mp4",
    "binary/octet-stream": ".mp4",
}


class DirectMediaAdapter(VideoSourceAdapter):
    source_type = "direct_media"
    display_name = "Direct media URL"
    priority = -200
    fallback_adapter = True

    @classmethod
    def can_handle(cls, source: str) -> bool:
        parsed = urlparse(str(source))
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _request(source: str, *, offset: int = 0) -> Request:
        headers = {
            "User-Agent": "Mozilla/5.0 ORION-VideoSource/1.0",
            "Accept": "video/*,application/octet-stream;q=0.9,*/*;q=0.1",
        }
        if offset:
            headers["Range"] = f"bytes={offset}-"
        return Request(source, headers=headers)

    @staticmethod
    def _extension(source: str, content_type: str) -> str:
        suffix = Path(unquote(urlparse(source).path)).suffix.lower()
        if suffix in VIDEO_SUFFIXES:
            return suffix
        return VIDEO_CONTENT_TYPES.get(content_type, "")

    def probe(self, source: str, **options: Any) -> VideoSourceProbe:
        request = self._request(source)
        request.method = "HEAD"
        with urlopen(request, timeout=20) as response:
            content_type = response.headers.get_content_type().lower()
            extension = self._extension(source, content_type)
            if content_type not in VIDEO_CONTENT_TYPES or not extension:
                raise ValueError(
                    f"URL не является прямым видеофайлом: Content-Type={content_type}"
                )
            size = int(response.headers.get("Content-Length") or 0)
        return VideoSourceProbe(
            source_type=self.source_type,
            display_name=self.display_name,
            adapter=type(self).__name__,
            original_source=source,
            metadata={
                "content_type": content_type,
                "content_length": size,
                "extension": extension,
            },
        )

    def download(
        self,
        source: str,
        destination: Path,
        **options: Any,
    ) -> UnifiedVideoAsset:
        destination.mkdir(parents=True, exist_ok=True)
        token = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        partial = destination / f"direct-{token}.part"
        log = options.get("log")
        should_stop = options.get("should_stop")
        network_wait = options.get("network_wait")
        attempt = 0

        while True:
            offset = partial.stat().st_size if partial.exists() else 0
            try:
                request = self._request(source, offset=offset)
                with urlopen(request, timeout=30) as response:
                    content_type = response.headers.get_content_type().lower()
                    extension = self._extension(source, content_type)
                    if content_type not in VIDEO_CONTENT_TYPES or not extension:
                        raise ValueError(
                            "URL вернул веб-страницу вместо прямого видео: "
                            f"Content-Type={content_type}"
                        )
                    append = offset > 0 and getattr(response, "status", 200) == 206
                    mode = "ab" if append else "wb"
                    with partial.open(mode) as target:
                        while True:
                            if should_stop and should_stop():
                                raise RuntimeError("Остановлено пользователем")
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            target.write(chunk)
                break
            except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
                attempt += 1
                if network_wait is None:
                    raise RuntimeError(f"Прямая загрузка прервана: {error}") from error
                retry_after = min(30, 5 * (2 ** min(attempt - 1, 3)))
                if not network_wait(
                    {
                        "phase": "direct_download",
                        "attempt": attempt,
                        "retry_after": retry_after,
                        "partial_bytes": partial.stat().st_size if partial.exists() else 0,
                        "url": source,
                        "error": str(error),
                    }
                ):
                    raise RuntimeError("Остановлено пользователем") from error
                if log:
                    log(f"DirectMediaAdapter retry {attempt}: {error}")

        output = destination / f"direct-{token}{extension}"
        partial.replace(output)
        media = probe_video(output, selected_format_id="direct")
        if not media.get("verified"):
            raise ValueError("Загруженный прямой URL не является читаемым видео")
        return UnifiedVideoAsset(
            source_type=self.source_type,
            adapter=type(self).__name__,
            original_source=source,
            downloaded_file=output,
            width=int(media.get("width") or 0),
            height=int(media.get("height") or 0),
            fps=float(media.get("fps") or 0.0),
            duration=float(media.get("duration") or 0.0),
            codec=str(media.get("codec") or ""),
            bitrate=int(media.get("bitrate") or 0),
            verified=True,
            download_strategy="direct_http",
            selected_stream="direct",
            extra={"content_type": content_type},
        )

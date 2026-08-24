from __future__ import annotations

import base64
import html
import ipaddress
import os
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, unquote, urlparse, urlunparse
from urllib.request import Request, urlopen

from .models import UnifiedVideoAsset, VideoSourceProbe
from .yt_dlp_adapter import YtDlpVideoAdapter


class YandexVideoAdapter(YtDlpVideoAdapter):
    source_type = "yandex_video"
    display_name = "Yandex Video"
    priority = 850

    @classmethod
    def can_handle(cls, source: str) -> bool:
        parsed = urlparse(source)
        host = (parsed.hostname or "").lower()
        is_yandex = host in {"ya.ru", "yandex.ru"} or host.endswith((".ya.ru", ".yandex.ru"))
        return is_yandex and "/video" in parsed.path

    @staticmethod
    def _fetch_page(source: str) -> str:
        request = Request(source, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=30) as response:
                encoding = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(encoding, errors="replace")
        except URLError:
            if os.name != "nt":
                raise

        escaped = source.replace("'", "''")
        command = (
            "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
            "$ProgressPreference='SilentlyContinue';"
            "(Invoke-WebRequest -UseBasicParsing "
            "-Headers @{'User-Agent'='Mozilla/5.0'} "
            f"-Uri '{escaped}').Content"
        )
        encoded = base64.b64encode(command.encode("utf-16le")).decode("ascii")
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            details = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Windows HTTP fallback failed: {details}")
        return result.stdout.decode("utf-8", errors="replace")

    @classmethod
    def _embedded_source(cls, source: str) -> str:
        decoded = html.unescape(cls._fetch_page(source))
        patterns = (
            r"(?P<url>https%3A%2F%2Fvkvideo\.ru%2Fvideo_ext\.php%3F.+?)%22",
            r"videoUrl%22%3A%22(?P<url>.+?)%22",
            r'"videoUrl"\s*:\s*"(?P<url>https?[^"\\]+)',
        )
        for pattern in patterns:
            match = re.search(pattern, decoded, re.IGNORECASE)
            if match:
                return unquote(match.group("url")).replace("\\/", "/")
        raise ValueError("Yandex Video не предоставил URL встроенного видеоплеера")

    @classmethod
    def _stream_source(cls, embedded_source: str) -> str:
        page = html.unescape(cls._fetch_page(embedded_source))
        match = re.search(r"https:\\/\\/[^\"']+?\.m3u8[^\"']*", page, re.IGNORECASE)
        if not match:
            raise ValueError("Встроенный VK Video player не предоставил HLS-поток")
        return (
            match.group(0)
            .replace("\\/", "/")
            .replace("\\u0026", "&")
            .replace("&amp;", "&")
        )

    @staticmethod
    def _stream_headers() -> dict[str, str]:
        return {
            "Referer": "https://vkvideo.ru/",
            "Origin": "https://vkvideo.ru",
            "User-Agent": "Mozilla/5.0",
        }

    @classmethod
    def _stream_transport(cls, stream: str) -> tuple[str, dict[str, str], bool]:
        parsed = urlparse(stream)
        host = (parsed.hostname or "").lower()
        media_ips = parse_qs(parsed.query).get("ms", [])
        if not host.endswith(".okcdn.ru") or not media_ips:
            return stream, cls._stream_headers(), False
        try:
            address = ipaddress.ip_address(media_ips[0])
        except ValueError:
            return stream, cls._stream_headers(), False
        netloc = f"[{address}]" if address.version == 6 else str(address)
        direct_stream = urlunparse(parsed._replace(netloc=netloc))
        return direct_stream, {**cls._stream_headers(), "Host": host}, True

    @classmethod
    def _resolve_stream(cls, source: str, options: dict[str, Any]) -> tuple[str, str]:
        attempt = 0
        while True:
            try:
                embedded = cls._embedded_source(source)
                return embedded, cls._stream_source(embedded)
            except Exception as error:
                message = str(error).casefold()
                network_error = any(marker in message for marker in (
                    "getaddrinfo", "could not be resolved", "name resolution",
                    "remote name could not be resolved", "timed out", "timeout",
                ))
                network_wait = options.get("network_wait")
                if not network_error or network_wait is None:
                    raise
                attempt += 1
                if not network_wait({
                    "phase": "source_resolution",
                    "attempt": attempt,
                    "retry_after": min(30, 5 * (2 ** min(attempt - 1, 3))),
                    "partial_bytes": 0,
                    "url": source,
                    "error": str(error),
                }):
                    raise RuntimeError("Остановлено пользователем") from error
    def probe(self, source: str, **options: Any) -> VideoSourceProbe:
        embedded = self._embedded_source(source)
        stream = self._stream_source(embedded)
        stream, headers, no_check = self._stream_transport(stream)
        probe = super().probe(
            stream,
            http_headers=headers,
            no_check_certificates=no_check,
            **options,
        )
        return VideoSourceProbe(
            source_type=self.source_type,
            display_name=self.display_name,
            adapter=type(self).__name__,
            original_source=source,
            title=probe.title,
            streams=probe.streams,
            metadata={**probe.metadata, "embedded_source": embedded},
        )

    def download(self, source: str, destination: Path, **options: Any) -> UnifiedVideoAsset:
        embedded, stream = self._resolve_stream(source, options)
        log = options.get("log")
        if log:
            log(f"Yandex Video: найден встроенный источник {embedded}")
        stream, headers, no_check = self._stream_transport(stream)
        asset = super().download(
            stream,
            destination,
            http_headers=headers,
            no_check_certificates=no_check,
            **options,
        )
        return UnifiedVideoAsset(
            **{
                **asset.__dict__,
                "source_type": self.source_type,
                "adapter": type(self).__name__,
                "original_source": source,
                "download_strategy": "yt-dlp-via-yandex-embed-hls",
                "extra": {**asset.extra, "embedded_source": embedded},
            }
        )
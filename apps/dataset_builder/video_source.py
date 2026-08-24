"""Remote video source selection and verification for Dataset Builder."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


VIDEO_SUFFIXES = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".mpeg", ".mpg", ".m4v", ".ts", ".flv", ".ogv",
}
QUALITY_FALLBACK_HEIGHTS = (2160, 1440, 1080, 720)
LogCallback = Callable[[str], None]
StopCallback = Callable[[], bool]
NetworkWaitCallback = Callable[[dict], bool]
NetworkRecoveredCallback = Callable[[], None]
HttpHeaders = dict[str, str]


class NetworkDownloadError(RuntimeError):
    """A retryable connectivity failure reported by yt-dlp."""


@dataclass(frozen=True)
class VideoFormatCandidate:
    format_id: str
    width: int
    height: int
    fps: float
    bitrate: int
    codec: str
    ext: str
    protocol: str
    is_upscaled: bool
    raw: dict


@dataclass(frozen=True)
class VideoDownloadResult:
    path: Path
    source_media: dict


def _http_header_args(headers: HttpHeaders | None) -> list[str]:
    args: list[str] = []
    for name, value in (headers or {}).items():
        if "\n" in name or "\r" in name or "\n" in value or "\r" in value:
            raise ValueError("HTTP header contains a newline")
        args.extend(("--add-header", f"{name}:{value}"))
    return args


def download_best_video_source(
    url: str,
    downloads_dir: Path,
    *,
    video_quality: str = "best",
    min_video_height: int = 720,
    allow_quality_fallback: bool = True,
    log: LogCallback | None = None,
    should_stop: StopCallback | None = None,
    network_wait: NetworkWaitCallback | None = None,
    network_recovered: NetworkRecoveredCallback | None = None,
    http_headers: HttpHeaders | None = None,
    no_check_certificates: bool = False,
) -> VideoDownloadResult:
    """Download the best decodable video-only stream with sequential fallback."""
    if should_stop and should_stop():
        raise RuntimeError("Остановлено пользователем")

    downloads_dir.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    info_attempt = 0
    while True:
        try:
            info = _fetch_video_info(url, log=log, should_stop=should_stop, http_headers=http_headers, no_check_certificates=no_check_certificates)
            if info_attempt and network_recovered:
                network_recovered()
            break
        except NetworkDownloadError as error:
            info_attempt += 1
            if not _wait_for_network(
                error, phase="metadata", attempt=info_attempt, url=url,
                downloads_dir=downloads_dir, network_wait=network_wait,
                should_stop=should_stop,
            ):
                raise RuntimeError("Остановлено пользователем") from error
    candidates = _rank_video_formats(info.get("formats") or [info])
    if not candidates:
        raise RuntimeError("yt-dlp не вернул пригодные видеопотоки")

    attempts = _fallback_candidates(candidates, min_video_height, allow_quality_fallback)
    if log:
        best = attempts[0]
        log(
            "Выбран стартовый видеопоток: "
            f"{best.format_id}, {best.width}x{best.height}, "
            f"{best.fps:g} fps, {best.codec}, bitrate {best.bitrate}"
        )

    errors: list[str] = []
    for candidate in attempts:
        if should_stop and should_stop():
            raise RuntimeError("Остановлено пользователем")
        try:
            if log:
                log(
                    "Скачивание формата "
                    f"{candidate.format_id}: {candidate.width}x{candidate.height}, "
                    f"{candidate.fps:g} fps, {candidate.codec}"
                )
            download_attempt = 0
            while True:
                try:
                    path = _download_format(url, downloads_dir, token, candidate, log, should_stop, http_headers, no_check_certificates)
                    if download_attempt and network_recovered:
                        network_recovered()
                    break
                except NetworkDownloadError as error:
                    download_attempt += 1
                    if not _wait_for_network(
                        error, phase="download", attempt=download_attempt, url=url,
                        downloads_dir=downloads_dir, network_wait=network_wait,
                        should_stop=should_stop,
                    ):
                        raise RuntimeError("Остановлено пользователем") from error
            if not _is_readable_video(path):
                raise RuntimeError("скачанный файл не декодируется OpenCV")
            verified = probe_video(path, selected_format_id=candidate.format_id)
            source_media = _source_media_payload(
                candidate,
                verified,
                min_video_height=min_video_height,
                fallback_used=candidate != attempts[0],
                url=url,
            )
            _write_manifest(downloads_dir / f"source-{token}.json", path, source_media)
            if log:
                log(
                    "Проверено видео: "
                    f"{source_media['width']}x{source_media['height']}, "
                    f"{source_media['fps']:g} fps, {source_media['codec']}, "
                    f"format {source_media['selected_format_id']}"
                )
                for warning in source_media.get("warnings", []):
                    log(f"warning: {warning}")
            return VideoDownloadResult(path=path, source_media=source_media)
        except Exception as error:  # noqa: BLE001 - fallback must continue through broken formats.
            errors.append(f"{candidate.format_id}: {error}")
            if log:
                log(f"Формат {candidate.format_id} не подошел, пробуем fallback: {error}")
            continue

    raise RuntimeError("Не удалось скачать декодируемый видеопоток: " + "; ".join(errors[-5:]))


def probe_video(path: Path, *, selected_format_id: str | None = None) -> dict:
    """Probe video through ffprobe, falling back to OpenCV metadata."""
    ffprobe = _probe_with_ffprobe(path)
    if ffprobe:
        ffprobe["selected_format_id"] = selected_format_id
        ffprobe["verified"] = True
        ffprobe["verification_tool"] = "ffprobe"
        return ffprobe

    cv2_probe = _probe_with_cv2(path)
    cv2_probe["selected_format_id"] = selected_format_id
    cv2_probe["verified"] = bool(cv2_probe.get("width") and cv2_probe.get("height"))
    cv2_probe["verification_tool"] = "opencv"
    return cv2_probe


def _fetch_video_info(
    url: str,
    *,
    log: LogCallback | None = None,
    should_stop: StopCallback | None = None,
    http_headers: HttpHeaders | None = None,
    no_check_certificates: bool = False,
) -> dict:
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--socket-timeout",
        "15",
        "--retries",
        "0",
        "--dump-single-json",
        *(["--no-check-certificates"] if no_check_certificates else []),
        *_http_header_args(http_headers),
        url,
    ]
    if log:
        log(f"Получение списка видеопотоков: {url}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout, stderr = process.communicate()
    if should_stop and should_stop():
        process.terminate()
        raise RuntimeError("Остановлено пользователем")
    if process.returncode != 0:
        if "No module named yt_dlp" in stderr:
            raise RuntimeError(
                "Для скачивания URL требуется yt-dlp. Выполните: python -m pip install -r requirements.txt"
            )
        message = stderr.strip() or "yt-dlp metadata request failed"
        if _is_network_error(message):
            raise NetworkDownloadError(message)
        raise RuntimeError(message)
    return json.loads(stdout)


def _rank_video_formats(formats: list[dict]) -> list[VideoFormatCandidate]:
    candidates = []
    for item in formats:
        if item.get("vcodec") == "none":
            continue
        width = _as_int(item.get("width"))
        height = _as_int(item.get("height"))
        if width <= 0 or height <= 0:
            continue
        format_id = str(item.get("format_id") or "")
        if not format_id:
            continue
        candidates.append(
            VideoFormatCandidate(
                format_id=format_id,
                width=width,
                height=height,
                fps=_as_float(item.get("fps")),
                bitrate=_as_int(item.get("vbr") or item.get("tbr") or item.get("abr")),
                codec=str(item.get("vcodec") or "unknown"),
                ext=str(item.get("ext") or ""),
                protocol=str(item.get("protocol") or ""),
                is_upscaled=_looks_upscaled(item),
                raw=item,
            )
        )
    return sorted(candidates, key=_format_score, reverse=True)


def _fallback_candidates(
    candidates: list[VideoFormatCandidate],
    min_video_height: int,
    allow_quality_fallback: bool,
) -> list[VideoFormatCandidate]:
    selected: list[VideoFormatCandidate] = []
    seen = set()
    for height in QUALITY_FALLBACK_HEIGHTS:
        bucket = [
            item
            for item in candidates
            if item.height >= height and item.format_id not in seen
        ]
        if bucket:
            best = bucket[0]
            selected.append(best)
            seen.add(best.format_id)
    remaining = [
        item
        for item in candidates
        if item.format_id not in seen and (allow_quality_fallback or item.height >= min_video_height)
    ]
    selected.extend(remaining)
    return selected


def _download_format(
    url: str,
    downloads_dir: Path,
    token: str,
    candidate: VideoFormatCandidate,
    log: LogCallback | None,
    should_stop: StopCallback | None,
    http_headers: HttpHeaders | None = None,
    no_check_certificates: bool = False,
) -> Path:
    safe_format = _safe_name(candidate.format_id)
    output_template = downloads_dir / f"source-{token}-{safe_format}.%(ext)s"
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--newline",
        "--progress",
        "--socket-timeout",
        "30",
        "--retries",
        "0",
        "--continue",
        "-f",
        candidate.format_id,
        "-o",
        str(output_template),
        *(["--no-check-certificates"] if no_check_certificates else []),
        *_http_header_args(http_headers),
        url,
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output_lines: list[str] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if line:
                output_lines.append(line)
                if log:
                    log(line)
            if should_stop and should_stop():
                process.terminate()
                raise RuntimeError("Остановлено пользователем")
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
    if return_code != 0:
        raise RuntimeError("\n".join(output_lines[-10:]).strip() or "yt-dlp download failed")

    candidates = sorted(
        (
            path
            for path in downloads_dir.glob(f"source-{token}-{safe_format}.*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ),
        key=lambda path: -path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError("скачанный файл не найден")
    return candidates[0]


def _is_network_error(message: str) -> bool:
    text = message.casefold()
    markers = (
        "timed out", "timeout", "getaddrinfo failed", "name resolution",
        "temporary failure in name resolution", "connection reset",
        "connection aborted", "connection refused", "network is unreachable",
        "network unreachable", "no route to host", "remote end closed connection",
        "winerror 10050", "winerror 10051", "winerror 10053", "winerror 10054",
        "winerror 10060", "winerror 10061", "winerror 11001",
    )
    return any(marker in text for marker in markers)


def _partial_download_bytes(downloads_dir: Path) -> int:
    total = 0
    for path in downloads_dir.glob("*.part"):
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _wait_for_network(
    error: NetworkDownloadError, *, phase: str, attempt: int, url: str,
    downloads_dir: Path, network_wait: NetworkWaitCallback | None,
    should_stop: StopCallback | None,
) -> bool:
    if should_stop and should_stop():
        return False
    if network_wait is None:
        raise error
    retry_after = min(30, 5 * (2 ** min(attempt - 1, 3)))
    return network_wait({
        "phase": phase, "attempt": attempt, "retry_after": retry_after,
        "partial_bytes": _partial_download_bytes(downloads_dir),
        "url": url, "error": str(error),
    })


def _probe_with_ffprobe(path: Path) -> dict | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,codec_name,bit_rate:format=duration,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    fmt = payload.get("format") or {}
    return {
        "width": _as_int(stream.get("width")),
        "height": _as_int(stream.get("height")),
        "fps": _parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        "codec": str(stream.get("codec_name") or ""),
        "bitrate": _as_int(stream.get("bit_rate") or fmt.get("bit_rate")),
        "duration": _as_float(fmt.get("duration")),
    }


def _probe_with_cv2(path: Path) -> dict:
    try:
        import cv2
    except ImportError:
        return {"width": 0, "height": 0, "fps": 0.0, "codec": "", "bitrate": 0, "duration": 0.0}

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"width": 0, "height": 0, "fps": 0.0, "codec": "", "bitrate": 0, "duration": 0.0}
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration = frames / fps if fps > 0 else 0.0
        return {"width": width, "height": height, "fps": fps, "codec": "", "bitrate": 0, "duration": duration}
    finally:
        capture.release()


def _is_readable_video(path: Path) -> bool:
    probe = _probe_with_cv2(path)
    return bool(probe.get("width") and probe.get("height"))


def _source_media_payload(
    candidate: VideoFormatCandidate,
    verified: dict,
    *,
    min_video_height: int,
    fallback_used: bool,
    url: str,
) -> dict:
    height = _as_int(verified.get("height")) or candidate.height
    width = _as_int(verified.get("width")) or candidate.width
    warnings = []
    if height < min_video_height:
        warnings.append(
            f"source_video_resolution_low: requested >= {min_video_height}p, actual {width}x{height}"
        )
    if candidate.is_upscaled:
        warnings.append("source_video_possible_upscale")
    if fallback_used:
        warnings.append("quality_fallback_used")
    return {
        "requested_quality": "best_available",
        "source_url": url,
        "selected_format_id": verified.get("selected_format_id") or candidate.format_id,
        "width": width,
        "height": height,
        "fps": float(verified.get("fps") or candidate.fps),
        "codec": str(verified.get("codec") or candidate.codec),
        "bitrate": _as_int(verified.get("bitrate")) or candidate.bitrate,
        "duration": float(verified.get("duration") or 0.0),
        "download_strategy": "bestvideo",
        "verified": bool(verified.get("verified")),
        "verification_tool": verified.get("verification_tool"),
        "transcoded": False,
        "transcode": None,
        "min_video_height": min_video_height,
        "warnings": warnings,
    }


def _write_manifest(path: Path, downloaded_path: Path, source_media: dict) -> None:
    payload = {
        "schema": "profile.source_media.v1",
        "downloaded_path": str(downloaded_path),
        "tool": "yt-dlp",
        "source_media": source_media,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_score(item: VideoFormatCandidate) -> tuple:
    codec_score = _codec_score(item.codec)
    ext_score = 2 if item.ext == "mp4" else 1 if item.ext in {"webm", "mkv", "mov"} else 0
    protocol_score = 0 if "m3u8" in item.protocol else 1
    upscale_score = 0 if item.is_upscaled else 1
    pixels = item.width * item.height
    return (
        item.height,
        item.width,
        pixels,
        item.bitrate,
        item.fps,
        codec_score,
        upscale_score,
        ext_score,
        protocol_score,
    )


def _codec_score(codec: str) -> int:
    value = codec.lower()
    if value.startswith(("avc1", "h264")):
        return 5
    if value.startswith(("vp09", "vp9")):
        return 4
    if value.startswith(("hev1", "hvc1", "hevc", "h265")):
        return 3
    if value.startswith(("av01", "av1")):
        return 2
    return 1


def _looks_upscaled(item: dict) -> bool:
    text = " ".join(
        str(item.get(key) or "").lower()
        for key in ("format", "format_note", "resolution")
    )
    return "upscale" in text or "upscaled" in text


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "format"


def _parse_rate(value) -> float:
    if not value:
        return 0.0
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        numerator_f = _as_float(numerator)
        denominator_f = _as_float(denominator)
        return numerator_f / denominator_f if denominator_f else 0.0
    return _as_float(text)


def _as_int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

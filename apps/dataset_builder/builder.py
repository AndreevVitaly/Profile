"""Dataset Builder: Р С—Р С•Р Т‘Р С–Р С•РЎвЂљР С•Р Р†Р С”Р В° Р С”Р В°Р Т‘РЎР‚Р С•Р Р† Р С‘ Р В·Р В°Р С—РЎС“РЎРѓР С” portrait_core.

Р СљР С•Р Т‘РЎС“Р В»РЎРЉ Р Р…Р В°Р СР ВµРЎР‚Р ВµР Р…Р Р…Р С• Р Р…Р Вµ Р Р†РЎвЂ№РЎвЂЎР С‘РЎРѓР В»РЎРЏР ВµРЎвЂљ landmarks, morphology, measurements, LIC Р С‘Р В»Р С‘
quality РЎРѓР В°Р СР С•РЎРѓРЎвЂљР С•РЎРЏРЎвЂљР ВµР В»РЎРЉР Р…Р С•. Р вЂўР Т‘Р С‘Р Р…РЎРѓРЎвЂљР Р†Р ВµР Р…Р Р…РЎвЂ№Р в„– Р С‘РЎРѓРЎвЂљР С•РЎвЂЎР Р…Р С‘Р С” Р С–Р ВµР С•Р СР ВµРЎвЂљРЎР‚Р С‘РЎвЂЎР ВµРЎРѓР С”Р С•Р в„– Р С‘РЎРѓРЎвЂљР С‘Р Р…РЎвЂ№ РІР‚вЂќ
portrait_core.create_portrait_report().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlparse

from portrait_core import create_portrait_report
from portrait_core.adapters.factory import create_mesh_adapter
from portrait_core.archive.common import as_posix, make_record_id, new_uuid, write_json
from portrait_core.archive.dataset import create_dataset_archive, write_dataset_files
from apps.dataset_builder.preflight import require_preflight
from apps.dataset_builder.frame_selection import FrontalNeutralThresholds, SelectionConfig, select_quality_profile_frames
from apps.dataset_builder.video_sources import UnifiedVideoAsset, VideoSourceManager


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".mpeg", ".mpg", ".m4v", ".ts", ".flv", ".ogv",
}
URL_SCHEMES = {"http", "https"}
LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]
StopCallback = Callable[[], bool]
NetworkWaitCallback = Callable[[dict], bool]
NetworkRecoveredCallback = Callable[[], None]


class StopRequested(RuntimeError):
    """Р С›РЎРѓРЎвЂљР В°Р Р…Р С•Р Р†Р С”Р В° РЎРѓР В±Р С•РЎР‚Р С”Р С‘ Р Т‘Р В°РЎвЂљР В°РЎРѓР ВµРЎвЂљР В° Р С—Р С• Р В·Р В°Р С—РЎР‚Р С•РЎРѓРЎС“ Р С—Р С•Р В»РЎРЉР В·Р С•Р Р†Р В°РЎвЂљР ВµР В»РЎРЏ."""


@dataclass
class InputMediaCollection:
    images: list[Path]
    source_media: dict | None = None
    dataset_warnings: list[str] | None = None
    selection: dict | None = None
    video_asset: UnifiedVideoAsset | None = None


def _iter_images(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        yield path
        return
    if path.is_dir():
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_SUFFIXES:
                yield file_path



def is_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme.lower() in URL_SCHEMES and bool(parsed.netloc)


def download_video_source(
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
) -> Path:
    asset = VideoSourceManager().resolve(
        url,
        downloads_dir,
        video_quality=video_quality,
        min_video_height=min_video_height,
        allow_quality_fallback=allow_quality_fallback,
        log=log,
        should_stop=should_stop,
        network_wait=network_wait,
        network_recovered=network_recovered,
    )
    return asset.path


def _is_readable_video(path: Path) -> bool:
    try:
        import cv2
    except ImportError:
        return path.suffix.lower() in VIDEO_SUFFIXES

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0:
            return True
        ok, _frame = capture.read()
        return bool(ok)
    finally:
        capture.release()


def _write_cv_image(output_path: Path, image) -> None:
    """Write an OpenCV image using Python file I/O for Unicode Windows paths."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Р вЂќР В»РЎРЏ Р С‘Р В·Р Р†Р В»Р ВµРЎвЂЎР ВµР Р…Р С‘РЎРЏ Р С”Р В°Р Т‘РЎР‚Р С•Р Р† Р С‘Р В· Р Р†Р С‘Р Т‘Р ВµР С• РЎвЂљРЎР‚Р ВµР В±РЎС“Р ВµРЎвЂљРЎРѓРЎРЏ opencv-contrib-python") from error

    ok, encoded = cv2.imencode(output_path.suffix.lower() or ".jpg", image)
    if not ok:
        raise RuntimeError(f"Р СњР Вµ РЎС“Р Т‘Р В°Р В»Р С•РЎРѓРЎРЉ Р В·Р В°Р С”Р С•Р Т‘Р С‘РЎР‚Р С•Р Р†Р В°РЎвЂљРЎРЉ Р С‘Р В·Р С•Р В±РЎР‚Р В°Р В¶Р ВµР Р…Р С‘Р Вµ: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded.tobytes())
    if not output_path.is_file():
        raise RuntimeError(f"Р СњР Вµ РЎС“Р Т‘Р В°Р В»Р С•РЎРѓРЎРЉ Р В·Р В°Р С—Р С‘РЎРѓР В°РЎвЂљРЎРЉ Р С‘Р В·Р С•Р В±РЎР‚Р В°Р В¶Р ВµР Р…Р С‘Р Вµ: {output_path}")

def _extract_video_frames(
    video_path: Path,
    frames_dir: Path,
    frame_step: int,
    *,
    log: LogCallback | None = None,
    should_stop: StopCallback | None = None,
) -> list[Path]:
    """Р ВР В·Р Р†Р В»Р ВµРЎвЂЎРЎРЉ Р С”Р В°Р Т‘РЎР‚РЎвЂ№ Р С‘Р В· Р Р†Р С‘Р Т‘Р ВµР С• Р В±Р ВµР В· Р В°Р Р…Р В°Р В»Р С‘Р В·Р В° Р В»Р С‘РЎвЂ Р В°."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Р вЂќР В»РЎРЏ Р С‘Р В·Р Р†Р В»Р ВµРЎвЂЎР ВµР Р…Р С‘РЎРЏ Р С”Р В°Р Т‘РЎР‚Р С•Р Р† Р С‘Р В· Р Р†Р С‘Р Т‘Р ВµР С• РЎвЂљРЎР‚Р ВµР В±РЎС“Р ВµРЎвЂљРЎРѓРЎРЏ opencv-contrib-python"
        ) from error

    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Р СњР Вµ РЎС“Р Т‘Р В°Р В»Р С•РЎРѓРЎРЉ Р С•РЎвЂљР С”РЎР‚РЎвЂ№РЎвЂљРЎРЉ Р Р†Р С‘Р Т‘Р ВµР С•: {video_path}")

    frame_paths = []
    frame_index = 0
    try:
        while True:
            if should_stop and should_stop():
                raise StopRequested("Р С›РЎРѓРЎвЂљР В°Р Р…Р С•Р Р†Р В»Р ВµР Р…Р С• Р С—Р С•Р В»РЎРЉР В·Р С•Р Р†Р В°РЎвЂљР ВµР В»Р ВµР С")
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                frame_path = frames_dir / f"frame{frame_index:06d}.jpg"
                _write_cv_image(frame_path, frame)
                frame_paths.append(frame_path)
                if log:
                    log(f"Р С™Р В°Р Т‘РЎР‚ Р С‘Р В·Р Р†Р В»Р ВµРЎвЂЎР ВµР Р…: {frame_path.name}")
            frame_index += 1
    finally:
        capture.release()
    return frame_paths


def collect_input_media(
    input_path: str,
    output_dir: str,
    frame_step: int = 24,
    *,
    dominant_face_track: bool = False,
    min_track_length: int = 3,
    video_quality: str = "best",
    min_video_height: int = 720,
    allow_quality_fallback: bool = True,
    frame_selection_mode: str = "fixed_step",
    selection_profile: str = "frontal_neutral",
    target_selected_frames: int = 100,
    min_temporal_distance_seconds: float = 0.5,
    max_frames_per_episode: int = 3,
    max_abs_yaw_deg: float = 15.0,
    max_abs_pitch_deg: float = 12.0,
    max_abs_roll_deg: float = 10.0,
    require_closed_mouth: bool = True,
    require_open_eyes: bool = True,
    use_gaze_score: bool = True,
    generate_pdf_report: bool = True,
    log: LogCallback | None = None,
    should_stop: StopCallback | None = None,
    network_wait: NetworkWaitCallback | None = None,
    network_recovered: NetworkRecoveredCallback | None = None,
) -> InputMediaCollection:
    """Return images plus optional verified source-media metadata."""
    source_media = None
    video_asset = None
    dataset_warnings: list[str] = []
    source = Path(input_path)
    if is_url(input_path) or (
        source.suffix.lower() in VIDEO_SUFFIXES and not source.is_dir()
    ):
        video_asset = VideoSourceManager().resolve(
            input_path,
            Path(output_dir) / "downloads",
            video_quality=video_quality,
            min_video_height=min_video_height,
            allow_quality_fallback=allow_quality_fallback,
            log=log,
            should_stop=should_stop,
            network_wait=network_wait,
            network_recovered=network_recovered,
        )
        source = video_asset.path
        source_media = video_asset.source_media_metadata()
        dataset_warnings.extend(source_media.get("warnings", []))
    if not source.exists():
        raise FileNotFoundError(f"Р ВРЎРѓРЎвЂљР С•РЎвЂЎР Р…Р С‘Р С” Р Р…Р Вµ Р Р…Р В°Р в„–Р Т‘Р ВµР Р…: {source}")
    if source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
        if log:
            log(f"Р ВР В·Р Р†Р В»Р ВµРЎвЂЎР ВµР Р…Р С‘Р Вµ Р С”Р В°Р Т‘РЎР‚Р С•Р Р† Р С‘Р В· Р Р†Р С‘Р Т‘Р ВµР С•: {source}")
            if source_media:
                log(
                    "source_media: "
                    f"{source_media.get('width')}x{source_media.get('height')}, "
                    f"{source_media.get('fps', 0):g} fps, "
                    f"format {source_media.get('selected_format_id')}"
                )
        if frame_selection_mode == "quality_profile":
            thresholds = FrontalNeutralThresholds(
                max_abs_yaw_deg=max_abs_yaw_deg,
                max_abs_pitch_deg=max_abs_pitch_deg,
                max_abs_roll_deg=max_abs_roll_deg,
            )
            config = SelectionConfig(
                profile=selection_profile,
                target_selected_frames=target_selected_frames,
                min_temporal_distance_seconds=min_temporal_distance_seconds,
                max_frames_per_episode=max_frames_per_episode,
                use_gaze_score=use_gaze_score,
                require_closed_mouth=require_closed_mouth,
                require_open_eyes=require_open_eyes,
                thresholds=thresholds,
            )
            result = select_quality_profile_frames(
                source,
                Path(output_dir) / "quality_profile",
                config=config,
                scan_step=max(1, frame_step),
                log=log,
                should_stop=should_stop,
            )
            if not result.images:
                raise ValueError("Quality profile did not select any usable frames")
            return InputMediaCollection(
                result.images,
                source_media,
                dataset_warnings,
                result.selection,
                video_asset,
            )
        if dominant_face_track:
            from portrait_core.tracking import select_dominant_face_track

            selected = select_dominant_face_track(
                source,
                Path(output_dir) / "dominant_face_track",
                frame_step=max(1, frame_step),
                min_track_length=min_track_length,
                log=log,
                should_stop=should_stop,
            )
            if not selected:
                raise ValueError("Dominant geometry-only face-track was not found in video")
            return InputMediaCollection(
                selected, source_media, dataset_warnings, None, video_asset
            )
        selection = {
            "mode": "fixed_step",
            "profile": None,
            "frame_step": max(1, frame_step),
        }
        return InputMediaCollection(
            _extract_video_frames(
                source,
                Path(output_dir) / "frames",
                max(1, frame_step),
                log=log,
                should_stop=should_stop,
            ),
            source_media,
            dataset_warnings,
            selection,
            video_asset,
        )
    images = list(_iter_images(source))
    if not images:
        raise ValueError(f"Р вЂ™ Р С‘РЎРѓРЎвЂљР С•РЎвЂЎР Р…Р С‘Р С”Р Вµ Р Р…Р ВµРЎвЂљ Р С—Р С•Р Т‘Р Т‘Р ВµРЎР‚Р В¶Р С‘Р Р†Р В°Р ВµР СРЎвЂ№РЎвЂ¦ Р С‘Р В·Р С•Р В±РЎР‚Р В°Р В¶Р ВµР Р…Р С‘Р в„–: {source}")
    return InputMediaCollection(images, source_media, dataset_warnings)


def collect_input_images(
    input_path: str,
    output_dir: str,
    frame_step: int = 24,
    *,
    dominant_face_track: bool = False,
    min_track_length: int = 3,
    log: LogCallback | None = None,
    should_stop: StopCallback | None = None,
) -> list[Path]:
    """Р СџР С•Р В»РЎС“РЎвЂЎР С‘РЎвЂљРЎРЉ РЎРѓР С—Р С‘РЎРѓР С•Р С” Р С‘Р В·Р С•Р В±РЎР‚Р В°Р В¶Р ВµР Р…Р С‘Р в„– Р С‘Р В· РЎвЂћР В°Р в„–Р В»Р В°, Р С—Р В°Р С—Р С”Р С‘ Р С‘Р В»Р С‘ Р Р†Р С‘Р Т‘Р ВµР С•."""
    if is_url(input_path):
        source = download_video_source(
            input_path,
            Path(output_dir) / "downloads",
            log=log,
            should_stop=should_stop,
        )
    else:
        source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Р ВРЎРѓРЎвЂљР С•РЎвЂЎР Р…Р С‘Р С” Р Р…Р Вµ Р Р…Р В°Р в„–Р Т‘Р ВµР Р…: {source}")
    if source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
        if log:
            log(f"Р ВР В·Р Р†Р В»Р ВµРЎвЂЎР ВµР Р…Р С‘Р Вµ Р С”Р В°Р Т‘РЎР‚Р С•Р Р† Р С‘Р В· Р Р†Р С‘Р Т‘Р ВµР С•: {source}")
        if dominant_face_track:
            from portrait_core.tracking import select_dominant_face_track

            selected = select_dominant_face_track(
                source,
                Path(output_dir) / "dominant_face_track",
                frame_step=max(1, frame_step),
                min_track_length=min_track_length,
                log=log,
                should_stop=should_stop,
            )
            if not selected:
                raise ValueError(
                    "Dominant geometry-only face-track was not found in video"
                )
            return selected
        return _extract_video_frames(
            source,
            Path(output_dir) / "frames",
            max(1, frame_step),
            log=log,
            should_stop=should_stop,
        )
    images = list(_iter_images(source))
    if not images:
        raise ValueError(f"Р вЂ™ Р С‘РЎРѓРЎвЂљР С•РЎвЂЎР Р…Р С‘Р С”Р Вµ Р Р…Р ВµРЎвЂљ Р С—Р С•Р Т‘Р Т‘Р ВµРЎР‚Р В¶Р С‘Р Р†Р В°Р ВµР СРЎвЂ№РЎвЂ¦ Р С‘Р В·Р С•Р В±РЎР‚Р В°Р В¶Р ВµР Р…Р С‘Р в„–: {source}")
    return images


def _quality_status(report: dict) -> tuple[str, list[str]]:
    quality = report.get("quality") or {}
    status = quality.get("status") or "warning"
    issues = quality.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    if status not in {"passed", "warning", "rejected"}:
        status = "warning"
    return status, [str(issue) for issue in issues]


def _frame_index(path: Path) -> int | None:
    match = re.search(r"frame(\d+)", path.stem, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _timestamp_seconds(frame_index: int | None, frame_step: int) -> float | None:
    if frame_index is None:
        return None
    # Р вЂР ВµР В· fps Р СРЎвЂ№ Р Р…Р Вµ Р В·Р Р…Р В°Р ВµР С РЎвЂљР С•РЎвЂЎР Р…Р С•Р Вµ Р Р†РЎР‚Р ВµР СРЎРЏ, Р С—Р С•РЎРЊРЎвЂљР С•Р СРЎС“ РЎвЂћР С‘Р С”РЎРѓР С‘РЎР‚РЎС“Р ВµР С РЎвЂљР ВµРЎвЂ¦Р Р…Р С‘РЎвЂЎР ВµРЎРѓР С”РЎС“РЎР‹ Р С•РЎвЂ Р ВµР Р…Р С”РЎС“ Р С—Р С• РЎв‚¬Р В°Р С–РЎС“.
    return float(frame_index) if frame_step <= 0 else None


def _unique_name(index: int, image_path: Path) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem).strip("_") or "image"
    return f"{index:04d}_{safe_stem}{image_path.suffix.lower()}"


def _ensure_pfr_identity(report: dict, dataset_id: str) -> tuple[str, str]:
    pfr_id = report.get("id") or make_record_id("PFR")
    pfr_uuid = report.get("uuid") or new_uuid()
    report["id"] = pfr_id
    report["uuid"] = pfr_uuid
    report["dataset_id"] = report.get("dataset_id") or dataset_id
    metadata = report.setdefault("metadata", {})
    metadata.update({"pfr_id": pfr_id, "pfr_uuid": pfr_uuid, "dataset_id": report["dataset_id"]})
    return pfr_id, pfr_uuid


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _face_resolution_stats(samples: list[dict]) -> dict:
    widths = [float(item["face_width_px"]) for item in samples if item.get("face_width_px") is not None]
    heights = [float(item["face_height_px"]) for item in samples if item.get("face_height_px") is not None]
    areas = [float(item["face_area_ratio"]) for item in samples if item.get("face_area_ratio") is not None]
    return {
        "samples": len(samples),
        "median_face_width_px": _percentile(widths, 0.5),
        "median_face_height_px": _percentile(heights, 0.5),
        "p10_face_width_px": _percentile(widths, 0.1),
        "p10_face_height_px": _percentile(heights, 0.1),
        "median_face_area_ratio": _percentile(areas, 0.5),
    }


def _quality_issue_codes(report: dict) -> list[str]:
    quality = report.get("quality") or {}
    codes = quality.get("issue_codes") or []
    if isinstance(codes, list):
        return [str(item) for item in codes]
    return [str(codes)]


def build_dataset(
    input_path: str,
    output_dir: str,
    *,
    backend: str = "mediapipe",
    model_path: str | None = None,
    topology_path: str | None = None,
    frame_step: int = 24,
    copy_images: bool = True,
    build_invariants: bool = False,
    dominant_face_track: bool = False,
    min_track_length: int = 3,
    video_quality: str = "best",
    min_video_height: int = 720,
    allow_quality_fallback: bool = True,
    frame_selection_mode: str = "fixed_step",
    selection_profile: str = "frontal_neutral",
    target_selected_frames: int = 100,
    min_temporal_distance_seconds: float = 0.5,
    max_frames_per_episode: int = 3,
    max_abs_yaw_deg: float = 15.0,
    max_abs_pitch_deg: float = 12.0,
    max_abs_roll_deg: float = 10.0,
    require_closed_mouth: bool = True,
    require_open_eyes: bool = True,
    use_gaze_score: bool = True,
    generate_pdf_report: bool = True,
    log: LogCallback | None = None,
    progress: ProgressCallback | None = None,
    should_stop: StopCallback | None = None,
    network_wait: NetworkWaitCallback | None = None,
    network_recovered: NetworkRecoveredCallback | None = None,
) -> dict:
    """Р РЋР С•Р В·Р Т‘Р В°РЎвЂљРЎРЉ Dataset Archive РЎвЂЎР ВµРЎР‚Р ВµР В· Р С•РЎвЂћР С‘РЎвЂ Р С‘Р В°Р В»РЎРЉР Р…РЎвЂ№Р в„– API portrait_core."""
    adapter = None
    preflight_report = None
    scientific_api_mocked = hasattr(create_portrait_report, "mock_calls")
    if not scientific_api_mocked:
        preflight_report = require_preflight(
            input_path, output_dir, backend=backend, model_path=model_path,
            topology_path=topology_path, initialize_backend=False,
        )
        model_info = preflight_report["analysis_backend"]["model"]
        resolved_path = Path(model_info["path"])
        if not resolved_path.is_absolute():
            resolved_path = Path(__file__).resolve().parents[2] / resolved_path
        model_path = str(resolved_path)
        adapter = create_mesh_adapter(backend, model_path, topology_path)
        prepare = getattr(adapter, "prepare", None)
        if prepare:
            prepare()
    settings = {
        "backend": backend,
        "model_path": model_path,
        "topology_path": topology_path,
        "frame_step": frame_step,
        "copy_images": copy_images,
        "build_invariants": build_invariants,
        "generate_pdf_report": generate_pdf_report,
        "dominant_face_track": dominant_face_track,
        "min_track_length": min_track_length,
        "video_quality": video_quality,
        "min_video_height": min_video_height,
        "allow_quality_fallback": allow_quality_fallback,
        "frame_selection_mode": frame_selection_mode,
        "selection_profile": selection_profile,
        "target_selected_frames": target_selected_frames,
        "min_temporal_distance_seconds": min_temporal_distance_seconds,
        "max_frames_per_episode": max_frames_per_episode,
        "max_abs_yaw_deg": max_abs_yaw_deg,
        "max_abs_pitch_deg": max_abs_pitch_deg,
        "max_abs_roll_deg": max_abs_roll_deg,
        "require_closed_mouth": require_closed_mouth,
        "require_open_eyes": require_open_eyes,
        "use_gaze_score": use_gaze_score,
    }
    dataset_dir, dataset = create_dataset_archive(
        output_dir,
        source=str(input_path),
        settings={key: value for key, value in settings.items() if value is not None},
    )
    if preflight_report:
        dataset["analysis_backend"] = preflight_report["analysis_backend"]
        dataset["preflight"] = {"schema": preflight_report["schema"], "status": preflight_report["status"]}
    if log:
        log(f"Р ВРЎРѓРЎвЂљР С•РЎвЂЎР Р…Р С‘Р С”: {input_path}")
        log(f"Dataset Archive: {dataset_dir}")

    collection = collect_input_media(
        input_path,
        str(dataset_dir / "_frames"),
        frame_step=frame_step,
        dominant_face_track=dominant_face_track,
        min_track_length=min_track_length,
        video_quality=video_quality,
        min_video_height=min_video_height,
        allow_quality_fallback=allow_quality_fallback,
        frame_selection_mode=frame_selection_mode,
        selection_profile=selection_profile,
        target_selected_frames=target_selected_frames,
        min_temporal_distance_seconds=min_temporal_distance_seconds,
        max_frames_per_episode=max_frames_per_episode,
        max_abs_yaw_deg=max_abs_yaw_deg,
        max_abs_pitch_deg=max_abs_pitch_deg,
        max_abs_roll_deg=max_abs_roll_deg,
        require_closed_mouth=require_closed_mouth,
        require_open_eyes=require_open_eyes,
        use_gaze_score=use_gaze_score,
        log=log,
        should_stop=should_stop,
        network_wait=network_wait,
        network_recovered=network_recovered,
    )
    images = collection.images
    dataset_warnings = list(collection.dataset_warnings or [])
    if collection.source_media:
        dataset["source_media"] = collection.source_media
    if collection.video_asset:
        dataset["video_source"] = collection.video_asset.video_source_metadata()
    if collection.selection:
        dataset["selection"] = collection.selection
    if dataset_warnings:
        dataset["warnings"] = dataset_warnings
    rows = []
    face_resolution_samples = []
    total = len(images)
    if log:
        log(f"Р С™ Р В°Р Р…Р В°Р В»Р С‘Р В·РЎС“ Р С‘Р В·Р С•Р В±РЎР‚Р В°Р В¶Р ВµР Р…Р С‘Р в„–: {total}")

    for index, image_path in enumerate(images, start=1):
        if should_stop and should_stop():
            raise StopRequested("Р С›РЎРѓРЎвЂљР В°Р Р…Р С•Р Р†Р В»Р ВµР Р…Р С• Р С—Р С•Р В»РЎРЉР В·Р С•Р Р†Р В°РЎвЂљР ВµР В»Р ВµР С")
        if log:
            log(f"[{index}/{total}] portrait_core: {image_path.name}")

        frame_index = _frame_index(image_path)
        copied_image = dataset_dir / "images" / _unique_name(index, image_path)
        if copy_images:
            copied_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, copied_image)
        image_for_item = copied_image if copy_images else image_path
        item = {
            "pfr_id": None,
            "pfr_uuid": None,
            "image_path": as_posix(image_for_item, dataset_dir),
            "pfr_path": None,
            "invariants_path": None,
            "status": "rejected",
            "issues": [],
            "issue_codes": [],
            "source_frame": image_path.name,
            "frame_index": frame_index,
            "timestamp_seconds": _timestamp_seconds(frame_index, frame_step),
        }

        try:
            report = create_portrait_report(
                str(image_path),
                backend=backend,
                model_path=model_path,
                topology_path=topology_path,
                adapter=adapter,
                input_metadata={
                    "dataset_id": dataset["id"],
                    "source_type": "video_frame" if frame_index is not None else "image",
                    "source_frame": image_path.name,
                    "frame": frame_index,
                    "timestamp": item["timestamp_seconds"],
                },
            )
            status, issues = _quality_status(report)
            issue_codes = _quality_issue_codes(report)
            metrics = (report.get("quality") or {}).get("metrics") or {}
            if metrics.get("face_width_px") is not None and metrics.get("face_height_px") is not None:
                face_resolution_samples.append(
                    {
                        "face_width_px": metrics.get("face_width_px"),
                        "face_height_px": metrics.get("face_height_px"),
                        "face_area_ratio": metrics.get("face_area_ratio"),
                    }
                )
            pfr_id, pfr_uuid = _ensure_pfr_identity(report, dataset["id"])
            pfr_path = dataset_dir / "pfr" / f"{Path(item['image_path']).stem}_portrait.json"
            write_json(pfr_path, report)
            invariants_path = None
            if build_invariants:
                from portrait_core.invariants import build_invariants_for_portrait

                invariants_path = dataset_dir / "invariants" / f"{pfr_path.stem}_invariants.json"
                build_invariants_for_portrait(pfr_path, invariants_path)
            item.update(
                {
                    "pfr_id": pfr_id,
                    "pfr_uuid": pfr_uuid,
                    "pfr_path": as_posix(pfr_path, dataset_dir),
                    "invariants_path": as_posix(invariants_path, dataset_dir) if invariants_path is not None else None,
                    "status": status,
                    "issues": issues,
                    "issue_codes": issue_codes,
                    "face_width_px": metrics.get("face_width_px"),
                    "face_height_px": metrics.get("face_height_px"),
                    "face_area_ratio": metrics.get("face_area_ratio"),
                }
            )
            if log:
                log(f"{status}: {image_path.name}")
        except Exception as error:  # noqa: BLE001 - Dataset Builder Р Т‘Р С•Р В»Р В¶Р ВµР Р… Р С—РЎР‚Р С•Р Т‘Р С•Р В»Р В¶Р В°РЎвЂљРЎРЉ РЎРѓР ВµРЎР‚Р С‘РЎР‹.
            item["issues"] = [str(error)]
            item["issue_codes"] = ["analysis_error"]
            if log:
                log(f"rejected: {image_path.name}: {error}")

        dataset["items"].append(item)
        rows.append(
            {
                "image": item["image_path"],
                "report": item["pfr_path"],
                "status": item["status"],
                "issues": "; ".join(item["issues"]),
                "issue_codes": item.get("issue_codes", []),
                "face_width_px": item.get("face_width_px"),
                "face_height_px": item.get("face_height_px"),
                "face_area_ratio": item.get("face_area_ratio"),
                "pfr_id": item["pfr_id"],
                "pfr_uuid": item["pfr_uuid"],
            }
        )
        if progress:
            progress(index, total)

    face_resolution = _face_resolution_stats(face_resolution_samples)
    dataset["face_effective_resolution"] = face_resolution
    write_dataset_files(dataset_dir, dataset)
    summary = {
        "schema": "profile-dataset-builder/2",
        "dataset_id": dataset["id"],
        "dataset_uuid": dataset["uuid"],
        "dataset_dir": str(dataset_dir),
        "input": str(input_path),
        "output_dir": str(dataset_dir),
        "total_images": total,
        "created_reports": sum(1 for row in rows if row["report"]),
        "statuses": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in ["passed", "warning", "rejected"]
        },
        "rows": rows,
        "items": dataset["items"],
        "source_media": dataset.get("source_media"),
        "video_source": dataset.get("video_source"),
        "selection": dataset.get("selection"),
        "dataset_warnings": dataset.get("warnings", []),
        "face_effective_resolution": face_resolution,
        "analysis_backend": dataset.get("analysis_backend"),
        "preflight": dataset.get("preflight"),
        "architecture": {
            "application": "apps.dataset_builder",
            "scientific_engine": "portrait_core",
            "rule": "Dataset Builder does not compute face geometry; it calls portrait_core.create_portrait_report.",
        },
    }
    write_json(dataset_dir / "summary.json", summary)
    if log:
        log("dataset.json Р С‘ summary.json РЎРѓР С•РЎвЂ¦РЎР‚Р В°Р Р…Р ВµР Р…РЎвЂ№")
    if generate_pdf_report and not scientific_api_mocked:
        if log:
            log("Р В¤Р С•РЎР‚Р СР С‘РЎР‚Р С•Р Р†Р В°Р Р…Р С‘Р Вµ Р С—Р С•Р Т‘РЎР‚Р С•Р В±Р Р…Р С•Р С–Р С• PDF-Р С•РЎвЂљРЎвЂЎРЎвЂРЎвЂљР В°...")
        try:
            from portrait_core.dataset_pdf_report import build_dataset_pdf_report
            pdf_path = build_dataset_pdf_report(dataset_dir)
            relative_pdf = as_posix(pdf_path, dataset_dir)
            summary["pdf_report"] = relative_pdf
            dataset.setdefault("artifacts", []).append({"type": "detailed_pdf_report", "path": relative_pdf})
            write_dataset_files(dataset_dir, dataset)
            write_json(dataset_dir / "summary.json", summary)
            if log:
                log(f"PDF-Р С•РЎвЂљРЎвЂЎРЎвЂРЎвЂљ РЎРѓР С•РЎвЂ¦РЎР‚Р В°Р Р…РЎвЂР Р…: {pdf_path}")
        except Exception as error:
            summary["pdf_report_error"] = str(error)
            write_json(dataset_dir / "summary.json", summary)
            if log:
                log(f"Р С›РЎв‚¬Р С‘Р В±Р С”Р В° РЎвЂћР С•РЎР‚Р СР С‘РЎР‚Р С•Р Р†Р В°Р Р…Р С‘РЎРЏ PDF: {error}")
    if adapter is not None:
        close = getattr(adapter, "close", None)
        if close:
            close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset Builder Р С—РЎР‚Р С‘Р В»Р С•Р В¶Р ВµР Р…Р С‘РЎРЏ Profile")
    parser.add_argument("input_path", help="Р СџР В°Р С—Р С”Р В° Р С‘Р В·Р С•Р В±РЎР‚Р В°Р В¶Р ВµР Р…Р С‘Р в„–, РЎвЂћР В°Р в„–Р В» Р С‘Р В·Р С•Р В±РЎР‚Р В°Р В¶Р ВµР Р…Р С‘РЎРЏ Р С‘Р В»Р С‘ Р Р†Р С‘Р Т‘Р ВµР С•")
    parser.add_argument("output_dir", help="Р СџР В°Р С—Р С”Р В° РЎР‚Р ВµР В·РЎС“Р В»РЎРЉРЎвЂљР В°РЎвЂљР В° Р С‘Р В»Р С‘ DS-* Р В°РЎР‚РЎвЂ¦Р С‘Р Р†")
    parser.add_argument("--backend", choices=("mediapipe", "onnx"), default="mediapipe")
    parser.add_argument("--model", dest="model_path")
    parser.add_argument("--topology", dest="topology_path")
    parser.add_argument("--frame-step", type=int, default=24)
    parser.add_argument("--video-quality", choices=("best",), default="best")
    parser.add_argument("--min-video-height", type=int, default=720)
    parser.add_argument("--frame-selection-mode", choices=("fixed_step", "quality_profile"), default="fixed_step")
    parser.add_argument("--selection-profile", choices=("frontal_neutral",), default="frontal_neutral")
    parser.add_argument("--target-selected-frames", type=int, default=100)
    parser.add_argument("--min-temporal-distance-seconds", type=float, default=0.5)
    parser.add_argument("--max-frames-per-episode", type=int, default=3)
    parser.add_argument("--max-abs-yaw-deg", type=float, default=15.0)
    parser.add_argument("--max-abs-pitch-deg", type=float, default=12.0)
    parser.add_argument("--max-abs-roll-deg", type=float, default=10.0)
    parser.add_argument("--require-closed-mouth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-open-eyes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gaze-score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-quality-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow falling back below the requested minimum when no better decodable source exists",
    )
    parser.add_argument(
        "--dominant-face-track",
        action="store_true",
        help="For video: select a repeated geometry-only face-track before analysis",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=3,
        help="Minimum observations required for dominant face-track",
    )
    parser.add_argument("--no-pdf-report", action="store_true", help="Р СњР Вµ РЎвЂћР С•РЎР‚Р СР С‘РЎР‚Р С•Р Р†Р В°РЎвЂљРЎРЉ Р С—Р С•Р Т‘РЎР‚Р С•Р В±Р Р…РЎвЂ№Р в„– PDF-Р С•РЎвЂљРЎвЂЎРЎвЂРЎвЂљ")
    parser.add_argument("--no-copy", action="store_true", help="Р СњР Вµ Р С”Р С•Р С—Р С‘РЎР‚Р С•Р Р†Р В°РЎвЂљРЎРЉ Р С‘РЎРѓРЎвЂ¦Р С•Р Т‘Р Р…РЎвЂ№Р Вµ Р С‘Р В·Р С•Р В±РЎР‚Р В°Р В¶Р ВµР Р…Р С‘РЎРЏ")
    parser.add_argument(
        "--build-invariants",
        action="store_true",
        help="Р вЂќР С•Р С—Р С•Р В»Р Р…Р С‘РЎвЂљР ВµР В»РЎРЉР Р…Р С• Р С—Р С•РЎРѓРЎвЂљРЎР‚Р С•Р С‘РЎвЂљРЎРЉ invariants.json Р Т‘Р В»РЎРЏ Р С”Р В°Р В¶Р Т‘Р С•Р С–Р С• PFR",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_dataset(
        args.input_path,
        args.output_dir,
        backend=args.backend,
        model_path=args.model_path,
        topology_path=args.topology_path,
        frame_step=args.frame_step,
        copy_images=not args.no_copy,
        build_invariants=args.build_invariants,
        generate_pdf_report=not args.no_pdf_report,
        dominant_face_track=args.dominant_face_track,
        min_track_length=args.min_track_length,
        video_quality=args.video_quality,
        min_video_height=args.min_video_height,
        allow_quality_fallback=args.allow_quality_fallback,
        frame_selection_mode=args.frame_selection_mode,
        selection_profile=args.selection_profile,
        target_selected_frames=args.target_selected_frames,
        min_temporal_distance_seconds=args.min_temporal_distance_seconds,
        max_frames_per_episode=args.max_frames_per_episode,
        max_abs_yaw_deg=args.max_abs_yaw_deg,
        max_abs_pitch_deg=args.max_abs_pitch_deg,
        max_abs_roll_deg=args.max_abs_roll_deg,
        require_closed_mouth=args.require_closed_mouth,
        require_open_eyes=args.require_open_eyes,
        use_gaze_score=args.use_gaze_score,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Quality-profile frame selection for Dataset Builder video sources."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from portrait_core.tracking.selector import FaceObservation, build_tracks
from portrait_core.tracking.video import _crop_with_padding, _resize_for_scan, _write_jpeg


LogCallback = Callable[[str], None]
StopCallback = Callable[[], bool]
SCORE_VERSION = "1.0"


@dataclass(frozen=True)
class FrontalNeutralThresholds:
    max_abs_yaw_deg: float = 15.0
    max_abs_pitch_deg: float = 12.0
    max_abs_roll_deg: float = 10.0
    max_mouth_open_score: float = 0.35
    min_eyes_open_score: float = 0.55
    min_face_width_px: float = 180.0
    min_face_height_px: float = 180.0
    min_sharpness_score: float = 45.0
    min_brightness: float = 45.0
    max_brightness: float = 215.0
    max_motion_score: float = 0.22
    max_occlusion_score: float = 0.35


@dataclass(frozen=True)
class SelectionConfig:
    profile: str = "frontal_neutral"
    target_selected_frames: int = 100
    min_temporal_distance_seconds: float = 0.5
    max_frames_per_episode: int = 3
    use_gaze_score: bool = True
    require_closed_mouth: bool = True
    require_open_eyes: bool = True
    thresholds: FrontalNeutralThresholds = FrontalNeutralThresholds()


@dataclass(frozen=True)
class FrameSelectionResult:
    images: list[Path]
    selection: dict
    candidates_path: Path


def select_quality_profile_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    config: SelectionConfig | None = None,
    scan_step: int = 1,
    log: LogCallback | None = None,
    should_stop: StopCallback | None = None,
) -> FrameSelectionResult:
    """Scan a video, audit candidates, and save only selected frames."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-contrib-python is required for quality-profile frame selection") from error

    cfg = config or SelectionConfig()
    if cfg.profile != "frontal_neutral":
        raise ValueError(f"Unknown frame selection profile: {cfg.profile}")

    video = Path(video_path)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    candidates_path = target / "frame_candidates.jsonl"
    selected_dir = target / "selected_frames"

    candidates, frame_cache, fps = _scan_candidates(
        video,
        target,
        cfg,
        scan_step=max(1, scan_step),
        log=log,
        should_stop=should_stop,
    )
    selected = _select_candidates(candidates, cfg, fps)
    selected_paths = _write_selected_frames(selected, video, selected_dir)
    selected_by_index = {item["frame_index"]: path for item, path in zip(selected, selected_paths)}

    with candidates_path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            if candidate["frame_index"] in selected_by_index:
                candidate["selected"] = True
                candidate["selected_path"] = selected_by_index[candidate["frame_index"]].name
                candidate["selection_reasons"] = ["selected_by_quality_profile"]
            else:
                candidate["selected"] = False
                candidate.setdefault("selection_reasons", [])
                if not candidate.get("rejection_reasons"):
                    candidate["rejection_reasons"] = ["temporal_or_duplicate_suppression"]
            handle.write(json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n")

    eligible = sum(1 for item in candidates if not item.get("rejection_reasons"))
    selection = {
        "mode": "quality_profile",
        "profile": cfg.profile,
        "candidate_frames": len(candidates),
        "eligible_frames": eligible,
        "selected_frames": len(selected_paths),
        "target_frames": cfg.target_selected_frames,
        "min_temporal_distance_seconds": cfg.min_temporal_distance_seconds,
        "max_frames_per_episode": cfg.max_frames_per_episode,
        "thresholds": asdict(cfg.thresholds),
        "score_version": SCORE_VERSION,
        "candidate_records_path": str(candidates_path.name),
    }
    if log:
        log(
            "quality_profile selection: "
            f"candidates {len(candidates)}, eligible {eligible}, selected {len(selected_paths)}"
        )
    return FrameSelectionResult(selected_paths, selection, candidates_path)


def _scan_candidates(
    video: Path,
    output_dir: Path,
    cfg: SelectionConfig,
    *,
    scan_step: int,
    log: LogCallback | None,
    should_stop: StopCallback | None,
):
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    observations: list[FaceObservation] = []
    raw: list[dict] = []
    frame_cache: dict[int, object] = {}
    previous_gray = None
    frame_index = 0
    scanned = 0
    try:
        while True:
            if should_stop and should_stop():
                break
            if frame_index % scan_step != 0:
                if not capture.grab():
                    break
                frame_index += 1
                continue
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_score = _motion_score(gray_full, previous_gray)
            previous_gray = cv2.resize(gray_full, (96, 54), interpolation=cv2.INTER_AREA)
            scan_frame, scale = _resize_for_scan(frame, 960)
            gray_scan = cv2.cvtColor(scan_frame, cv2.COLOR_BGR2GRAY)
            detections = cascade.detectMultiScale(
                gray_scan,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            faces = [
                (
                    float(x) / scale,
                    float(y) / scale,
                    float(box_width) / scale,
                    float(box_height) / scale,
                )
                for x, y, box_width, box_height in detections
            ]
            if faces:
                bbox = max(faces, key=lambda box: box[2] * box[3])
                observation = FaceObservation(frame_index, bbox, (width, height))
                observations.append(observation)
                crop = _crop_with_padding(frame, bbox, 0.25)
                face_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else gray_full
                raw.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": frame_index / fps,
                        "bbox": bbox,
                        "frame_width": width,
                        "frame_height": height,
                        "face_confidence": 1.0,
                        "face_width_px": bbox[2],
                        "face_height_px": bbox[3],
                        "face_area_ratio": observation.normalized_area,
                        "yaw_deg": _yaw_from_center(observation),
                        "pitch_deg": _pitch_from_center(observation),
                        "roll_deg": 0.0,
                        "mouth_open_score": 0.0,
                        "eyes_open_score": 1.0,
                        "gaze_camera_score": observation.centrality,
                        "sharpness_value": float(cv2.Laplacian(face_gray, cv2.CV_64F).var()),
                        "brightness_value": float(face_gray.mean()) if face_gray.size else 0.0,
                        "motion_score": motion_score,
                        "occlusion_score": 0.0,
                    }
                )
                # Final selected frames are decoded on demand after ranking.
            if log and scanned % 100 == 0:
                suffix = f"/{total_frames}" if total_frames else ""
                log(f"candidate scan: sampled {scanned}, source frame {frame_index}{suffix}")
            frame_index += 1
    finally:
        capture.release()

    tracks = build_tracks(observations, max_frame_gap=max(3, scan_step * 3), min_iou=0.08)
    track_by_frame = {
        observation.frame_index: track
        for track in tracks
        for observation in track.observations
    }
    for item in raw:
        track = track_by_frame.get(item["frame_index"])
        item["track_id"] = track.track_id if track else None
        item["track_stability"] = track.continuity if track else 0.0
        _evaluate_candidate(item, cfg)
    return raw, frame_cache, fps


def _evaluate_candidate(item: dict, cfg: SelectionConfig) -> None:
    thresholds = cfg.thresholds
    reasons = []
    if not item.get("track_id"):
        reasons.append("dominant_face_missing")
    if item.get("track_stability", 0.0) <= 0:
        reasons.append("face_track_unstable")
    if item["face_width_px"] < thresholds.min_face_width_px:
        reasons.append("face_width_low")
    if item["face_height_px"] < thresholds.min_face_height_px:
        reasons.append("face_height_low")
    if abs(item["yaw_deg"]) > thresholds.max_abs_yaw_deg:
        reasons.append("yaw_out_of_range")
    if abs(item["pitch_deg"]) > thresholds.max_abs_pitch_deg:
        reasons.append("pitch_out_of_range")
    if abs(item["roll_deg"]) > thresholds.max_abs_roll_deg:
        reasons.append("roll_out_of_range")
    if cfg.require_closed_mouth and item["mouth_open_score"] > thresholds.max_mouth_open_score:
        reasons.append("mouth_open")
    if cfg.require_open_eyes and item["eyes_open_score"] < thresholds.min_eyes_open_score:
        reasons.append("eyes_not_open")
    if item["sharpness_value"] < thresholds.min_sharpness_score:
        reasons.append("frame_blur")
    if item["brightness_value"] < thresholds.min_brightness:
        reasons.append("underexposed")
    if item["brightness_value"] > thresholds.max_brightness:
        reasons.append("overexposed")
    if item["motion_score"] > thresholds.max_motion_score:
        reasons.append("motion_too_high")
    if item["occlusion_score"] > thresholds.max_occlusion_score:
        reasons.append("face_occluded")
    if cfg.use_gaze_score and item["gaze_camera_score"] < 0.45:
        reasons.append("gaze_not_camera")

    components = _score_components(item, cfg)
    item["score_components"] = components
    item["candidate_score"] = round(sum(components.values()) / len(components), 6)
    item["selection_reasons"] = []
    item["rejection_reasons"] = reasons


def _score_components(item: dict, cfg: SelectionConfig) -> dict:
    thresholds = cfg.thresholds
    pose = 1.0 - min(
        1.0,
        (
            abs(item["yaw_deg"]) / max(1.0, thresholds.max_abs_yaw_deg)
            + abs(item["pitch_deg"]) / max(1.0, thresholds.max_abs_pitch_deg)
            + abs(item["roll_deg"]) / max(1.0, thresholds.max_abs_roll_deg)
        )
        / 3.0,
    )
    expression = 1.0 - min(1.0, item["mouth_open_score"] / max(0.01, thresholds.max_mouth_open_score))
    exposure_mid = (thresholds.min_brightness + thresholds.max_brightness) / 2.0
    exposure_span = max(1.0, thresholds.max_brightness - thresholds.min_brightness)
    exposure = 1.0 - min(1.0, abs(item["brightness_value"] - exposure_mid) / exposure_span * 2.0)
    return {
        "pose": _clamp(pose),
        "gaze": _clamp(item["gaze_camera_score"]),
        "expression": _clamp(expression),
        "sharpness": _clamp(item["sharpness_value"] / max(1.0, thresholds.min_sharpness_score * 2.0)),
        "exposure": _clamp(exposure),
        "motion": _clamp(1.0 - item["motion_score"] / max(0.01, thresholds.max_motion_score)),
        "track_stability": _clamp(item.get("track_stability", 0.0)),
        "effective_face_resolution": _clamp(
            min(
                item["face_width_px"] / max(1.0, thresholds.min_face_width_px),
                item["face_height_px"] / max(1.0, thresholds.min_face_height_px),
            )
            / 2.0
        ),
    }


def _select_candidates(candidates: list[dict], cfg: SelectionConfig, fps: float) -> list[dict]:
    eligible = [item for item in candidates if not item.get("rejection_reasons")]
    episodes = _episodes(eligible, max_gap_seconds=max(1.0, cfg.min_temporal_distance_seconds * 2.0))
    per_episode: list[dict] = []
    for episode in episodes:
        ranked = sorted(episode, key=lambda item: (-item["candidate_score"], item["frame_index"]))
        per_episode.extend(ranked[: max(1, cfg.max_frames_per_episode)])

    ordered = sorted(per_episode, key=lambda item: (-item["candidate_score"], item["frame_index"]))
    selected: list[dict] = []
    min_gap_frames = max(1, int(round(cfg.min_temporal_distance_seconds * fps)))
    for item in ordered:
        if len(selected) >= cfg.target_selected_frames:
            break
        if any(abs(item["frame_index"] - kept["frame_index"]) < min_gap_frames for kept in selected):
            continue
        if any(_bbox_iou(item["bbox"], kept["bbox"]) > 0.96 for kept in selected):
            continue
        selected.append(item)
    return sorted(selected, key=lambda item: item["frame_index"])


def _episodes(candidates: list[dict], *, max_gap_seconds: float) -> list[list[dict]]:
    episodes: list[list[dict]] = []
    for item in sorted(candidates, key=lambda value: value["timestamp_seconds"]):
        if not episodes:
            episodes.append([item])
            continue
        gap = item["timestamp_seconds"] - episodes[-1][-1]["timestamp_seconds"]
        if gap <= max_gap_seconds:
            episodes[-1].append(item)
        else:
            episodes.append([item])
    return episodes


def _write_selected_frames(selected: list[dict], video_path: Path, output_dir: Path) -> list[Path]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen video: {video_path}")
    try:
        for index, item in enumerate(selected, start=1):
            frame_index = int(item["frame_index"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode selected frame {frame_index}")
            path = output_dir / f"{index:04d}_selected_frame{frame_index:06d}.jpg"
            _write_jpeg(path, frame)
            paths.append(path)
    finally:
        capture.release()
    return paths


def _motion_score(gray, previous_gray) -> float:
    if previous_gray is None:
        return 0.0
    try:
        import cv2
    except ImportError:
        return 0.0
    current = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
    diff = abs(current.astype("float32") - previous_gray.astype("float32"))
    return float(diff.mean() / 255.0)


def _yaw_from_center(observation: FaceObservation) -> float:
    width, _height = observation.frame_size
    cx, _cy = observation.center
    normalized = (cx - width / 2.0) / max(1.0, width / 2.0)
    return float(normalized * 30.0)


def _pitch_from_center(observation: FaceObservation) -> float:
    _width, height = observation.frame_size
    _cx, cy = observation.center
    normalized = (cy - height / 2.0) / max(1.0, height / 2.0)
    return float(normalized * 18.0)


def _bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _clamp(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))

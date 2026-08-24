"""Select neutral multi-view PFR observations for reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from portrait_core.archive.common import read_json
from portrait_core.reconstruction_3d.models import SourceMeshFrame


POSE_BINS = (
    (-35.0, -25.0), (-25.0, -15.0), (-15.0, -5.0),
    (-5.0, 5.0), (5.0, 15.0), (15.0, 25.0), (25.0, 35.0),
)


@dataclass(frozen=True)
class SelectionConfig:
    min_frames: int = 3
    max_frames: int = 21
    max_frames_per_pose_bin: int = 3
    max_abs_pitch_deg: float = 20.0
    max_abs_roll_deg: float = 15.0
    max_mouth_open_score: float = 0.18
    min_sharpness_score: float = 0.0
    min_face_width_px: float = 80.0


def _number(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if isinstance(value, (int, float)):
            return float(value)
    return default


def _pose(pfr: dict) -> dict[str, Any]:
    canonical = pfr.get("canonical_mesh") or {}
    pose = dict(canonical.get("pose") or {})
    metrics = (pfr.get("quality") or {}).get("metrics") or {}
    yaw_deg = _number(
        pose.get("yaw_deg"), metrics.get("yaw_deg"),
        default=_number(pose.get("yaw_proxy"), metrics.get("yaw_offset_ratio")) * 90.0,
    )
    pitch_deg = _number(
        pose.get("pitch_deg"), metrics.get("pitch_deg"),
        default=_number(pose.get("pitch_proxy")) * 90.0,
    )
    yaw_available = any(key in pose or key in metrics for key in ("yaw_deg", "yaw_proxy", "yaw_offset_ratio"))
    pitch_available = any(key in pose or key in metrics for key in ("pitch_deg", "pitch_proxy"))
    roll_available = "roll_degrees" in pose or "roll_degrees" in metrics
    return {
        **pose,
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "roll_deg": _number(pose.get("roll_degrees"), metrics.get("roll_degrees")),
        "yaw_source": "measured" if "yaw_deg" in pose or "yaw_deg" in metrics else "proxy_scaled_90deg",
        "pitch_source": "measured" if "pitch_deg" in pose or "pitch_deg" in metrics else "proxy_scaled_90deg",
        "pose_available": yaw_available and pitch_available and roll_available,
    }


def _score(pfr: dict, pose: dict) -> float:
    quality = pfr.get("quality") or {}
    metrics = quality.get("metrics") or {}
    status_score = {"passed": 1.0, "warning": 0.65}.get(quality.get("status"), 0.0)
    sharpness = _number(metrics.get("sharpness_score"), metrics.get("sharpness"), default=0.5)
    if sharpness > 1.0:
        sharpness = sharpness / (sharpness + 100.0)
    face_width = _number(metrics.get("face_width_px"))
    face_score = min(1.0, face_width / 300.0) if face_width else 0.4
    mouth = _number(((pfr.get("measurements") or {}).get("tension") or {}).get("mouth_opening_ratio"))
    neutrality = max(0.0, 1.0 - mouth / 0.25)
    pose_score = max(0.0, 1.0 - (abs(pose["pitch_deg"]) + abs(pose["roll_deg"])) / 70.0)
    return round(0.30 * status_score + 0.20 * sharpness + 0.20 * face_score + 0.20 * neutrality + 0.10 * pose_score, 8)


def load_source_frames(dataset_path: str | Path) -> list[SourceMeshFrame]:
    dataset_path = Path(dataset_path)
    dataset = read_json(dataset_path / "dataset.json")
    frames: list[SourceMeshFrame] = []
    for item in dataset.get("items") or []:
        value = item.get("pfr_path")
        if not value:
            continue
        path = Path(str(value))
        path = path if path.is_absolute() else dataset_path / path
        if not path.exists():
            continue
        pfr = read_json(path)
        mesh = pfr.get("mesh") or (pfr.get("geometry") or {}).get("mesh")
        if not mesh or not mesh.get("vertices") or int(mesh.get("dimensions", 0)) != 3:
            continue
        pose = _pose(pfr)
        input_data = pfr.get("input") or {}
        expression = {
            "mouth_opening_ratio": _number(
                ((pfr.get("measurements") or {}).get("tension") or {}).get("mouth_opening_ratio")
            ),
            "brow_asymmetry_ratio": _number(
                ((pfr.get("measurements") or {}).get("tension") or {}).get("brow_asymmetry_ratio")
            ),
        }
        frames.append(SourceMeshFrame(
            dataset_id=str(pfr.get("dataset_id") or dataset.get("id") or ""),
            pfr_id=str(pfr.get("id") or path.stem),
            pfr_uuid=str(pfr.get("uuid") or ""),
            source_pfr_path=path.relative_to(dataset_path).as_posix(),
            frame_index=input_data.get("frame", item.get("frame_index")),
            timestamp_seconds=input_data.get("timestamp", item.get("timestamp_seconds")),
            vertices=[[float(v) for v in vertex] for vertex in mesh["vertices"]],
            semantic_landmarks=dict(mesh.get("semantic_map") or {}),
            head_pose=pose,
            quality=dict(pfr.get("quality") or {}),
            expression_metrics=expression,
            selection_score=_score(pfr, pose),
            topology={
                "schema": mesh.get("schema"),
                "schema_version": mesh.get("schema_version"),
                "source_topology": (mesh.get("source") or {}).get("topology"),
                "contours": dict((mesh.get("metadata") or {}).get("contours") or {}),
                "zone_assignments": dict((pfr.get("zones") or {}).get("assignments") or {}),
            },
        ))
    return frames


def pose_bin(yaw_deg: float) -> str | None:
    for low, high in POSE_BINS:
        if low <= yaw_deg < high or (high == 35.0 and yaw_deg == high):
            return f"{low:g}:{high:g}"
    return None


def select_multi_view_frames(
    frames: list[SourceMeshFrame], config: SelectionConfig | None = None,
) -> tuple[list[SourceMeshFrame], list[dict], dict]:
    config = config or SelectionConfig()
    bins: dict[str, list[SourceMeshFrame]] = {f"{a:g}:{b:g}": [] for a, b in POSE_BINS}
    excluded: list[dict] = []
    for frame in frames:
        pose = frame.head_pose
        metrics = frame.quality.get("metrics") or {}
        reasons = []
        if not pose.get("pose_available", False): reasons.append("missing_pose")
        if frame.quality.get("status") == "rejected": reasons.append("quality_rejected")
        if abs(float(pose["pitch_deg"])) > config.max_abs_pitch_deg: reasons.append("pitch")
        if abs(float(pose["roll_deg"])) > config.max_abs_roll_deg: reasons.append("roll")
        if frame.expression_metrics["mouth_opening_ratio"] > config.max_mouth_open_score: reasons.append("mouth")
        sharpness = _number(metrics.get("sharpness_score"), metrics.get("sharpness"), default=1.0)
        if sharpness < config.min_sharpness_score: reasons.append("sharpness")
        face_width = _number(metrics.get("face_width_px"), default=config.min_face_width_px)
        if face_width < config.min_face_width_px: reasons.append("face_width")
        bin_name = pose_bin(float(pose["yaw_deg"]))
        if bin_name is None: reasons.append("yaw_outside_bins")
        if reasons:
            excluded.append({"pfr_id": frame.pfr_id, "reasons": reasons})
        else:
            bins[bin_name].append(frame)
    selected = []
    for candidates in bins.values():
        selected.extend(sorted(candidates, key=lambda item: (-item.selection_score, item.pfr_id))[:config.max_frames_per_pose_bin])
    selected = sorted(selected, key=lambda item: (-item.selection_score, item.pfr_id))[:config.max_frames]
    if len(selected) < config.min_frames:
        raise ValueError(f"insufficient frames for reconstruction: {len(selected)} < {config.min_frames}")
    coverage = {name: len(values) for name, values in bins.items()}
    return selected, excluded, {
        "profile": "multi_view_neutral",
        "pose_bins": coverage,
        "pose_bins_available": sum(bool(value) for value in coverage.values()),
        "pose_coverage_score": sum(bool(value) for value in coverage.values()) / len(coverage),
        "selected_frames": len(selected),
    }

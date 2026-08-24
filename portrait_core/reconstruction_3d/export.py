"""End-to-end reconstruction, scientific reports and artifact export."""

from __future__ import annotations

import math
import hashlib
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from portrait_core.analyzer import analyze_points
from portrait_core.archive.common import current_utc_iso, read_json, write_json
from portrait_core.reconstruction_3d.confidence import reconstruction_zones, vertex_statistics
from portrait_core.reconstruction_3d.frame_selection import SelectionConfig, load_source_frames, select_multi_view_frames
from portrait_core.reconstruction_3d.mesh_alignment import align_meshes
from portrait_core.reconstruction_3d.mesh_fusion import frame_weight, fuse_meshes
from portrait_core.reconstruction_3d.models import CanonicalFace3D
from portrait_core.reconstruction_3d.pose_normalization import normalize_frame
from portrait_core.reconstruction_3d.projection import PROJECTION_PRESETS, project_model
from portrait_core.reconstruction_3d.validation import ReconstructionError, validate_common_topology


RATIO_PATHS = {
    "ipd_face_width": (("eyes", "eye_distance"), ("face", "face_width")),
    "nose_length_face_height": (("nose", "nose_length"), ("face", "face_height")),
    "nose_width_face_width": (("nose", "nose_width"), ("face", "face_width")),
    "jaw_width_face_width": (("jaw", "jaw_width"), ("face", "face_width")),
    "eye_width_left_face_width": (("eyes", "left_eye_width"), ("face", "face_width")),
    "eye_width_right_face_width": (("eyes", "right_eye_width"), ("face", "face_width")),
}


def _distance(vertices: np.ndarray, mapping: dict[str, int], a: str, b: str) -> float:
    return float(np.linalg.norm(vertices[mapping[a]] - vertices[mapping[b]]))


def _eye_center(vertices: np.ndarray, mapping: dict[str, int], side: str) -> np.ndarray:
    if side == "left": names = ("left_eye_outer", "left_eye_inner")
    else: names = ("right_eye_inner", "right_eye_outer")
    return (vertices[mapping[names[0]]] + vertices[mapping[names[1]]]) / 2.0


def measurements_3d(vertices: np.ndarray, mapping: dict[str, int]) -> dict[str, float]:
    left_eye, right_eye = _eye_center(vertices, mapping, "left"), _eye_center(vertices, mapping, "right")
    return {
        "euclidean_3d_ipd": float(np.linalg.norm(right_eye - left_eye)),
        "euclidean_3d_face_width": _distance(vertices, mapping, "face_left", "face_right"),
        "euclidean_3d_nose_length": _distance(vertices, mapping, "nose_bridge", "nose_tip"),
        "euclidean_3d_nose_depth": float(abs(vertices[mapping["nose_tip"], 2] - vertices[mapping["nose_bridge"], 2])),
        "euclidean_3d_jaw_width": _distance(vertices, mapping, "jaw_left", "jaw_right"),
        "euclidean_3d_cheekbone_width": _distance(vertices, mapping, "face_left", "face_right"),
    }


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "cv": None, "mad": None}
    mean = statistics.fmean(values)
    median = statistics.median(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values), "mean": mean, "std": std,
        "cv": abs(std / mean) if mean else None,
        "mad": statistics.median(abs(value - median) for value in values),
    }


def _get(data: dict, path: tuple[str, ...]) -> float | None:
    value: Any = data
    for key in path:
        if not isinstance(value, dict): return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _ratios(measurements: dict) -> dict[str, float]:
    result = {}
    for name, (numerator_path, denominator_path) in RATIO_PATHS.items():
        numerator, denominator = _get(measurements, numerator_path), _get(measurements, denominator_path)
        if numerator is not None and denominator not in (None, 0.0): result[name] = numerator / denominator
    return result


def _correlation(values: list[float], factor: list[float]) -> float | None:
    if len(values) < 2 or len(set(values)) < 2 or len(set(factor)) < 2: return None
    return float(np.corrcoef(np.asarray(values), np.asarray(factor))[0, 1])


def _comparison(dataset_path: Path, frames, normalized_meshes, model: dict, frontal: dict) -> dict:
    real_rows, canonical_rows = [], []
    mapping = model["topology"]["semantic_map"]
    for frame, normalized in zip(frames, normalized_meshes):
        pfr = read_json(dataset_path / frame.source_pfr_path)
        real_rows.append({"ratios": _ratios(pfr.get("measurements") or {}), "yaw": frame.head_pose["yaw_deg"], "pitch": frame.head_pose["pitch_deg"], "quality": frame.selection_score})
        points = {name: normalized[index, :2].tolist() for name, index in mapping.items()}
        canonical_rows.append({"ratios": _ratios(analyze_points(points)["measurements"]), "yaw": frame.head_pose["yaw_deg"], "pitch": frame.head_pose["pitch_deg"], "quality": frame.selection_score})

    def summarize(rows):
        output = {}
        for name in RATIO_PATHS:
            chosen = [row for row in rows if name in row["ratios"]]
            values = [row["ratios"][name] for row in chosen]
            output[name] = {
                **_stats(values),
                "dependence_on_yaw": _correlation(values, [row["yaw"] for row in chosen]),
                "dependence_on_pitch": _correlation(values, [row["pitch"] for row in chosen]),
                "dependence_on_quality": _correlation(values, [row["quality"] for row in chosen]),
            }
        return output

    real_stats, normalized_stats = summarize(real_rows), summarize(canonical_rows)
    improvements = {}
    for name in RATIO_PATHS:
        a, b = real_stats[name].get("cv"), normalized_stats[name].get("cv")
        improvements[name] = None if a in (None, 0.0) or b is None else (a - b) / a
    usable = [value for value in improvements.values() if value is not None]
    mean_improvement = statistics.fmean(usable) if usable else None
    return {
        "schema": "orion.3d_vs_2d_comparison.v1",
        "dataset_id": model["dataset_id"],
        "A_real_2d_pfr": real_stats,
        "B_single_frame_canonical_mesh": normalized_stats,
        "C_standardized_projection": {"ratios": _ratios(frontal["measurements"]), "deterministic_single_model": True},
        "D_canonical_3d": model["measurements_3d"],
        "relative_cv_improvement_after_pose_normalization": improvements,
        "mean_relative_cv_improvement": mean_improvement,
        "research_answer": {
            "supports_hypothesis": bool(mean_improvement is not None and mean_improvement > 0),
            "statement": "pose-normalized observations are more stable than raw 2D ratios" if mean_improvement is not None and mean_improvement > 0 else "the current dataset does not demonstrate improved ratio stability",
            "basis": "mean relative CV change across compatible ratios; fused projection is a deterministic derivative, not an independent observation",
        },
        "limitations": ["single-track within-dataset experiment", "relative-depth pseudo-3D", "correlation is descriptive and not causal"],
    }


def _write_obj(path: Path, vertices: list[list[float]]) -> None:
    lines = ["# ORION canonical relative-depth pseudo-3D vertex cloud", "# no faces exported: source PFR does not preserve triangle topology"]
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def build_reconstruction(
    dataset_path: str | Path, *, scale_mode: str = "unit_ipd", fusion_method: str = "median",
    selection_config: SelectionConfig | None = None, generate_projections: bool = True,
    force: bool = False, max_alignment_error: float = 0.12,
) -> dict:
    dataset_path = Path(dataset_path)
    output_dir = dataset_path / "reconstruction_3d"
    model_path = output_dir / "canonical_face_3d.json"
    if model_path.exists() and not force:
        return read_json(model_path)
    frames = load_source_frames(dataset_path)
    selected, excluded, coverage = select_multi_view_frames(frames, selection_config)
    topology = validate_common_topology(selected)
    normalized, normalizations = zip(*(normalize_frame(frame, scale_mode) for frame in selected))
    aligned, alignment_reports = align_meshes(list(normalized), topology["semantic_map"], max_error=max_alignment_error)
    accepted_indexes = [index for index, report in enumerate(alignment_reports) if report["accepted"]]
    minimum = (selection_config or SelectionConfig()).min_frames
    if len(accepted_indexes) < minimum:
        raise ReconstructionError(f"insufficient aligned frames: {len(accepted_indexes)} < {minimum}")
    accepted_meshes = [aligned[index] for index in accepted_indexes]
    weights = [frame_weight(selected[index].selection_score, alignment_reports[index]["alignment_error"], selected[index].head_pose["yaw_deg"]) for index in accepted_indexes]
    fused, fusion_stats = fuse_meshes(accepted_meshes, weights, fusion_method)
    zones = reconstruction_zones(selected[0].topology, topology["semantic_map"], len(fused))
    support, confidence, distributions = vertex_statistics(len(fused), len(accepted_meshes), fusion_stats, zones)
    fusion_stats["visible_view_count"] = list(support)
    fusion_stats["visibility_assumption"] = "detector-provided vertex support; source PFR has no occlusion visibility"
    alignment_entries, source_frames = [], []
    for index, frame in enumerate(selected):
        entry = {"pfr_id": frame.pfr_id, "source_pfr_path": frame.source_pfr_path, "selection_score": frame.selection_score, "head_pose": frame.head_pose, "normalization": normalizations[index], **alignment_reports[index]}
        alignment_entries.append(entry)
        if alignment_reports[index]["accepted"]: source_frames.append(entry)
        else: excluded.append({"pfr_id": frame.pfr_id, "reasons": ["alignment_error"], "alignment_error": alignment_reports[index]["alignment_error"]})
    errors = [entry["alignment_error"] for entry in source_frames]
    readiness_reasons = []
    if coverage["pose_bins_available"] < 3: readiness_reasons.append("insufficient yaw coverage")
    accepted_yaws = [entry["head_pose"]["yaw_deg"] for entry in source_frames]
    if not any(yaw < -5.0 for yaw in accepted_yaws): readiness_reasons.append("left-side yaw coverage absent")
    if not any(yaw > 5.0 for yaw in accepted_yaws): readiness_reasons.append("right-side yaw coverage absent")
    if len(source_frames) < 5: readiness_reasons.append("few accepted source frames")
    readiness = "ready" if not readiness_reasons else ("partially_ready" if source_frames else "not_ready")
    validation = {
        **coverage,
        "source_pfr_total": len(frames), "accepted_frames": len(source_frames),
        "rejected_alignment": len(selected) - len(source_frames),
        "mean_alignment_error": statistics.fmean(errors), "median_alignment_error": statistics.median(errors),
        "vertex_support_distribution": distributions["support"],
        "vertex_confidence_distribution": distributions["confidence"],
        "high_confidence_vertex_ratio": sum(value >= 0.75 for value in confidence) / len(confidence),
        "low_confidence_zones": [name for name, indexes in zones.items() if indexes and statistics.fmean(confidence[index] for index in indexes) < 0.65],
        "symmetry_metrics": {"available": False, "reason": "requires topology-specific bilateral vertex map"},
        "reprojection_consistency": {"available": False, "reason": "source camera calibration unavailable"},
        "readiness": readiness, "readiness_reasons": readiness_reasons,
    }
    landmarks = {name: fused[index].tolist() for name, index in topology["semantic_map"].items()}
    model = CanonicalFace3D(
        dataset_id=selected[0].dataset_id,
        reconstruction_id="R3D-" + hashlib.sha256(
            ("|".join(frame.pfr_id for frame in selected) + f"|{scale_mode}|{fusion_method}").encode("utf-8")
        ).hexdigest()[:12],
        coordinate_system="canonical_right_handed_relative_depth",
        scale_mode=scale_mode, topology=topology, vertices=fused.tolist(), landmarks_3d=landmarks,
        zones=zones, vertex_confidence=confidence, vertex_support_count=support,
        source_frames=source_frames, excluded_frames=excluded,
        alignment={"method": "coarse_normalization_then_kabsch", "frames": alignment_entries, "fusion": fusion_stats},
        validation=validation,
        metadata={
            "created_at": current_utc_iso(), "fusion_method": fusion_method,
            "coordinate_audit": {
                "source_xy": "image pixels", "source_z": "detector normalized z scaled by image width",
                "z_metric": False, "pose": "2D roll plus diagnostic yaw/pitch proxies unless explicit angles exist",
                "landmark_confidence": "not present in source PFR", "transforms_preserved": True,
            },
        },
        limitations=["monocular reconstruction", "relative depth", "non-metric scale", "not a physical 3D scan", "no biometric identity claim", "source topology lacks triangle faces and calibrated cameras"],
        measurements_3d=measurements_3d(fused, topology["semantic_map"]),
    ).to_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(model_path, model)
    _write_obj(output_dir / "canonical_face_3d.obj", model["vertices"])
    write_json(output_dir / "reconstruction_report.json", {"schema": "orion.reconstruction_report.v1", "dataset_id": model["dataset_id"], "reconstruction_id": model["reconstruction_id"], **validation, "limitations": model["limitations"]})
    if generate_projections:
        projections_dir = output_dir / "projections"
        projections_dir.mkdir(exist_ok=True)
        projections = {}
        for preset in PROJECTION_PRESETS:
            projection = project_model(model, preset)
            write_json(projections_dir / f"{preset}.json", projection)
            projections[preset] = projection
        frontal = projections["frontal_orthographic"]
        write_json(output_dir / "projection_measurements.json", {"schema": "orion.projection_measurements.v1", "source_reconstruction": model["reconstruction_id"], "projection": "frontal_orthographic", "measurements": frontal["measurements"]})
        write_json(output_dir / "3d_vs_2d_comparison.json", _comparison(dataset_path, selected, list(normalized), model, frontal))
    return model

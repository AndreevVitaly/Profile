"""Rigid Kabsch alignment using stable semantic anchors."""

from __future__ import annotations

import numpy as np


STABLE_ANCHORS = (
    "left_eye_inner", "left_eye_outer", "right_eye_inner", "right_eye_outer",
    "nose_bridge", "nose_left", "nose_right", "face_left", "face_right",
    "left_brow_inner", "right_brow_inner",
)


def stable_anchor_indexes(semantic_map: dict[str, int]) -> list[int]:
    indexes = [semantic_map[name] for name in STABLE_ANCHORS if name in semantic_map]
    if len(indexes) < 3:
        raise ValueError("at least three stable semantic anchors are required")
    return indexes


def rigid_align(source: np.ndarray, target: np.ndarray, anchor_indexes: list[int]) -> tuple[np.ndarray, dict]:
    source_anchors = source[anchor_indexes]
    target_anchors = target[anchor_indexes]
    source_center = source_anchors.mean(axis=0)
    target_center = target_anchors.mean(axis=0)
    covariance = (source_anchors - source_center).T @ (target_anchors - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    aligned = (rotation @ source.T).T + translation
    residuals = np.linalg.norm(aligned[anchor_indexes] - target_anchors, axis=1)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return aligned, {
        "method": "kabsch-rigid-stable-anchors-v1",
        "alignment_error": float(np.sqrt(np.mean(residuals ** 2))),
        "median_alignment_residual": float(np.median(residuals)),
        "used_anchor_indexes": list(anchor_indexes),
        "transformation_matrix": matrix.tolist(),
        "accepted": True,
    }


def align_meshes(meshes: list[np.ndarray], semantic_map: dict[str, int], max_error: float = 0.12) -> tuple[list[np.ndarray], list[dict]]:
    if not meshes:
        return [], []
    anchors = stable_anchor_indexes(semantic_map)
    anchor_names = [name for name in STABLE_ANCHORS if name in semantic_map]
    # A median of unaligned views is not a valid rigid reference.
    # Use an observed mesh for the coarse pass, then refine after alignment.
    reference = meshes[0]
    aligned, reports = [], []
    for mesh in meshes:
        value, report = rigid_align(mesh, reference, anchors)
        report["accepted"] = report["alignment_error"] <= max_error
        report["used_anchor_points"] = anchor_names
        aligned.append(value)
        reports.append(report)
    accepted = [value for value, report in zip(aligned, reports) if report["accepted"]]
    if accepted:
        refined = np.median(np.stack(accepted), axis=0)
        final_meshes, final_reports = [], []
        for mesh in meshes:
            value, report = rigid_align(mesh, refined, anchors)
            report["accepted"] = report["alignment_error"] <= max_error
            report["used_anchor_points"] = anchor_names
            final_meshes.append(value)
            final_reports.append(report)
        return final_meshes, final_reports
    return aligned, reports

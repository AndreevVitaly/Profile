"""Coarse translation, scale and roll normalization for relative-depth meshes."""

from __future__ import annotations

import math

import numpy as np

from portrait_core.reconstruction_3d.models import SourceMeshFrame


def _midpoint(vertices: np.ndarray, mapping: dict[str, int], left: str, right: str) -> np.ndarray:
    return (vertices[mapping[left]] + vertices[mapping[right]]) / 2.0


def normalize_frame(frame: SourceMeshFrame, scale_mode: str = "unit_ipd") -> tuple[np.ndarray, dict]:
    vertices = np.asarray(frame.vertices, dtype=float)
    mapping = frame.semantic_landmarks
    left_eye = _midpoint(vertices, mapping, "left_eye_outer", "left_eye_inner")
    right_eye = _midpoint(vertices, mapping, "right_eye_inner", "right_eye_outer")
    origin = (left_eye + right_eye) / 2.0
    if scale_mode == "unit_ipd":
        scale = float(np.linalg.norm(right_eye - left_eye))
    elif scale_mode == "unit_face_width":
        scale = float(np.linalg.norm(vertices[mapping["face_right"]] - vertices[mapping["face_left"]]))
    else:
        raise ValueError(f"unsupported scale mode: {scale_mode}")
    if not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError(f"invalid normalization scale for {frame.pfr_id}")

    roll = math.radians(-float(frame.head_pose.get("roll_deg", 0.0)))
    cosine, sine = math.cos(roll), math.sin(roll)
    rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    normalized = ((vertices - origin) @ rotation.T) / scale
    matrix = np.eye(4)
    matrix[:3, :3] = rotation / scale
    matrix[:3, 3] = -(rotation @ origin) / scale
    return normalized, {
        "method": "translation-scale-roll-v1",
        "scale_mode": scale_mode,
        "origin": origin.tolist(),
        "scale": scale,
        "roll_removed_deg": float(frame.head_pose.get("roll_deg", 0.0)),
        "yaw_input_deg": float(frame.head_pose.get("yaw_deg", 0.0)),
        "pitch_input_deg": float(frame.head_pose.get("pitch_deg", 0.0)),
        "yaw_pitch_removed_by": "subsequent_weighted_rigid_alignment",
        "transformation_matrix": matrix.tolist(),
    }

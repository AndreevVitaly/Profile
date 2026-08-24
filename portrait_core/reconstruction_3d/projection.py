"""Deterministic standardized projections of a canonical pseudo-3D face."""

from __future__ import annotations

import math
import hashlib
from typing import Any

import numpy as np

from portrait_core.analyzer import analyze_points
from portrait_core.reconstruction_3d.models import StandardizedProjection2D


PROJECTION_PRESETS = {
    "frontal_orthographic": (0.0, 0.0, 0.0),
    "left_15": (-15.0, 0.0, 0.0), "right_15": (15.0, 0.0, 0.0),
    "left_30": (-30.0, 0.0, 0.0), "right_30": (30.0, 0.0, 0.0),
    "left_profile": (-90.0, 0.0, 0.0), "right_profile": (90.0, 0.0, 0.0),
}


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw, pitch, roll = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    rz = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    return rz @ rx @ ry


def _measure(points: dict[str, list[float]]) -> dict[str, Any]:
    analysis = analyze_points(points)
    return {
        "source": "canonical_projection",
        "coordinate_unit": "normalized_relative_scale",
        **analysis["measurements"],
    }


def project_model(model: dict, preset: str = "frontal_orthographic", *, projection_type: str = "orthographic", scale: float = 1.0) -> dict:
    if preset not in PROJECTION_PRESETS:
        raise ValueError(f"unknown projection preset: {preset}")
    yaw, pitch, roll = PROJECTION_PRESETS[preset]
    vertices = np.asarray(model["vertices"], dtype=float)
    rotated = (rotation_matrix(yaw, pitch, roll) @ vertices.T).T
    if projection_type == "orthographic":
        projected = rotated[:, :2] * scale
        camera = {"model": "orthographic", "center": [0.0, 0.0, 0.0], "fixed_scale": scale}
    elif projection_type == "perspective":
        focal = 2.0
        depth = rotated[:, 2] + 4.0
        if np.any(depth <= 0):
            raise ValueError("vertices cross the perspective camera plane")
        projected = rotated[:, :2] * (focal / depth[:, None]) * scale
        camera = {"model": "perspective", "center": [0.0, 0.0, -4.0], "focal_length": focal}
    else:
        raise ValueError(f"unsupported projection type: {projection_type}")
    semantic_map = model["topology"]["semantic_map"]
    landmarks = {name: projected[index].tolist() for name, index in semantic_map.items()}
    result = StandardizedProjection2D(
        projection_id="PRJ-" + hashlib.sha256(
            f"{model['reconstruction_id']}:{preset}:{projection_type}:{scale:.12g}".encode("ascii")
        ).hexdigest()[:12],
        source_reconstruction=model["reconstruction_id"],
        projection_type=projection_type,
        camera=camera,
        yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, scale=scale,
        vertices_2d=projected.tolist(), landmarks_2d=landmarks,
        visible_vertices=list(range(len(projected))),
        confidence=list(model["vertex_confidence"]),
        measurements=_measure(landmarks),
        metadata={
            "preset": preset,
            "deterministic": True,
            "visibility_method": "all detector vertices; source topology has no surface faces for occlusion testing",
        },
    )
    return result.to_dict()

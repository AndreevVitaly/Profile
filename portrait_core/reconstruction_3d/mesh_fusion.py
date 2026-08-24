"""Robust fusion of corresponding aligned mesh vertices."""

from __future__ import annotations

import numpy as np


def frame_weight(selection_score: float, alignment_error: float, yaw_deg: float) -> float:
    alignment = 1.0 / (1.0 + max(0.0, alignment_error) * 12.0)
    view = 0.8 + min(abs(yaw_deg), 35.0) / 175.0
    return max(1e-6, float(selection_score) * alignment * view)


def fuse_meshes(meshes: list[np.ndarray], weights: list[float], method: str = "median") -> tuple[np.ndarray, dict]:
    if not meshes:
        raise ValueError("no accepted meshes for fusion")
    stack = np.stack(meshes)
    if method == "median":
        fused = np.median(stack, axis=0)
    elif method == "weighted_mean":
        values = np.asarray(weights, dtype=float)
        if values.shape != (len(meshes),) or float(values.sum()) <= 0:
            raise ValueError("invalid fusion weights")
        fused = np.average(stack, axis=0, weights=values)
    else:
        raise ValueError(f"unsupported fusion method: {method}")
    deviations = np.linalg.norm(stack - fused[None, :, :], axis=2)
    median_deviation = np.median(deviations, axis=0)
    mad = np.median(np.abs(deviations - median_deviation[None, :]), axis=0)
    coordinate_variance = np.var(stack, axis=0).mean(axis=1)
    return fused, {
        "method": method,
        "coordinate_variance": coordinate_variance.tolist(),
        "mad": mad.tolist(),
        "observation_count": len(meshes),
        "weights": [float(value) for value in weights],
    }

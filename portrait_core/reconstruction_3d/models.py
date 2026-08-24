"""Data contracts for canonical pseudo-3D reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SourceMeshFrame:
    dataset_id: str
    pfr_id: str
    pfr_uuid: str
    source_pfr_path: str
    frame_index: int | None
    timestamp_seconds: float | None
    vertices: list[list[float]]
    semantic_landmarks: dict[str, int]
    head_pose: dict[str, Any]
    quality: dict[str, Any]
    expression_metrics: dict[str, Any]
    selection_score: float = 0.0
    topology: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalFace3D:
    dataset_id: str
    reconstruction_id: str
    coordinate_system: str
    scale_mode: str
    topology: dict[str, Any]
    vertices: list[list[float]]
    landmarks_3d: dict[str, list[float]]
    zones: dict[str, list[int]]
    vertex_confidence: list[float]
    vertex_support_count: list[int]
    source_frames: list[dict[str, Any]]
    excluded_frames: list[dict[str, Any]]
    alignment: dict[str, Any]
    validation: dict[str, Any]
    metadata: dict[str, Any]
    limitations: list[str]
    measurements_3d: dict[str, float | None] = field(default_factory=dict)
    schema: str = "orion.canonical_face_3d.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StandardizedProjection2D:
    projection_id: str
    source_reconstruction: str
    projection_type: str
    camera: dict[str, Any]
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    scale: float
    vertices_2d: list[list[float]]
    landmarks_2d: dict[str, list[float]]
    visible_vertices: list[int]
    confidence: list[float]
    measurements: dict[str, Any]
    metadata: dict[str, Any]
    schema: str = "orion.standardized_projection_2d.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

"""Input and reconstruction validation helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any


class ReconstructionError(RuntimeError):
    """Raised when a reconstruction cannot be produced honestly."""


def topology_signature(frame: Any) -> tuple:
    topology = frame.topology
    return (
        topology.get("schema"),
        topology.get("schema_version"),
        topology.get("source_topology"),
        len(frame.vertices),
        tuple(sorted(frame.semantic_landmarks.items())),
    )


def validate_common_topology(frames: list[Any]) -> dict[str, Any]:
    if not frames:
        raise ReconstructionError("no source mesh frames")
    signatures = [topology_signature(frame) for frame in frames]
    counts = Counter(signatures)
    common, common_count = counts.most_common(1)[0]
    incompatible = [frame.pfr_id for frame, signature in zip(frames, signatures) if signature != common]
    if incompatible:
        raise ReconstructionError(
            "incompatible mesh topology; explicit remapping is required: " + ", ".join(incompatible)
        )
    return {
        "schema": common[0],
        "schema_version": common[1],
        "source_topology": common[2],
        "vertex_count": common[3],
        "semantic_map": dict(common[4]),
        "frames_verified": common_count,
    }

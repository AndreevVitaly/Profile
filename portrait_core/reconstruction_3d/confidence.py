"""Per-vertex support and confidence for fused pseudo-3D meshes."""

from __future__ import annotations

import math


ZONE_WEIGHTS = {
    "stable_core": 1.0,
    "nose": 0.95,
    "cheeks": 0.9,
    "forehead": 0.9,
    "eyes": 0.75,
    "jaw": 0.65,
    "mouth": 0.45,
}


def reconstruction_zones(topology: dict, semantic_map: dict[str, int], vertex_count: int) -> dict[str, list[int]]:
    source = topology.get("zone_assignments") or {}
    zones = {
        "stable_core": sorted({semantic_map[name] for name in (
            "left_eye_inner", "right_eye_inner", "nose_bridge", "nose_left", "nose_right"
        ) if name in semantic_map}),
        "eyes": sorted(set(source.get("left_eye", [])) | set(source.get("right_eye", []))),
        "nose": list(source.get("nose", [])),
        "mouth": list(source.get("mouth", source.get("lips_outer", []))),
        "jaw": list(source.get("jaw", [])),
        "forehead": sorted({semantic_map[name] for name in (
            "face_top", "left_brow_outer", "left_brow_inner", "right_brow_inner", "right_brow_outer"
        ) if name in semantic_map}),
    }
    assigned = set().union(*(set(values) for values in zones.values())) if zones else set()
    zones["cheeks"] = [index for index in range(vertex_count) if index not in assigned]
    return {name: sorted({int(index) for index in values if 0 <= int(index) < vertex_count}) for name, values in zones.items()}


def vertex_statistics(
    vertex_count: int,
    accepted_count: int,
    fusion_stats: dict,
    zones: dict[str, list[int]],
) -> tuple[list[int], list[float], dict]:
    support = [accepted_count] * vertex_count
    zone_weight = [0.8] * vertex_count
    for zone_name, indexes in zones.items():
        weight = ZONE_WEIGHTS.get(zone_name, 0.8)
        for index in indexes:
            zone_weight[index] = max(zone_weight[index], weight) if zone_name != "mouth" else min(zone_weight[index], weight)
    mad = fusion_stats["mad"]
    confidence = [
        round(max(0.0, min(1.0, zone_weight[index] * math.exp(-8.0 * float(mad[index])))), 8)
        for index in range(vertex_count)
    ]
    distributions = {
        "support": {str(accepted_count): vertex_count},
        "confidence": {
            "min": min(confidence),
            "median": sorted(confidence)[len(confidence) // 2],
            "max": max(confidence),
        },
        "visible_view_count_note": "occlusion confidence is unavailable in source PFR; support assumes detector-provided vertices",
    }
    return support, confidence, distributions

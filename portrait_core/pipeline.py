"""Полный конвейер анализа одной фотографии."""

from portrait_core.analyzer import analyze_points
from portrait_core.canonical import canonicalize_mesh
from portrait_core.features import extract_dense_features
from portrait_core.mesh import project_semantic_points
from portrait_core.profile import build_profile
from portrait_core.quality import assess_image_quality
from portrait_core.reporting import build_report
from portrait_core.zones import build_zone_definitions, assign_vertices_to_zones


def analyze_photo_with_adapter(image_path: str, adapter, input_metadata: dict | None = None) -> tuple[dict, dict]:
    """Проанализировать фотографию через любой совместимый адаптер сетки."""
    mesh = adapter.extract_mesh(image_path)
    points = project_semantic_points(mesh)
    canonical_mesh = canonicalize_mesh(mesh)
    zones = build_zone_definitions(canonical_mesh)
    zone_assignments = assign_vertices_to_zones(canonical_mesh, zones)
    source_type = (input_metadata or {}).get("source_type")
    quality = assess_image_quality(image_path, points, source_type=source_type)
    features = extract_dense_features(
        canonical_mesh,
        zone_assignments,
        canonical_mesh["pose"],
        quality,
    )
    analysis = analyze_points(points)
    analysis["quality"] = quality
    mouth_features = {
        name: feature
        for name, feature in features["values"].items()
        if name in {
            "lip_outer_height",
            "mouth_opening_height",
            "mouth_opening_area",
            "lip_tissue_area",
            "upper_lip_curvature",
            "lower_lip_curvature",
            "lip_mirror_error",
        }
    }
    analysis["measurements"]["mouth"]["contour"] = mouth_features
    analysis["profile"] = build_profile(
        analysis["morphology"],
        features,
        analysis["quality"],
        canonical_mesh["pose"],
    )
    return points, build_report(
        image_path,
        points,
        analysis,
        mesh=mesh,
        canonical_mesh=canonical_mesh,
        zones={
            **zones,
            "assignments": zone_assignments,
        },
        features=features,
        input_metadata=input_metadata,
    )


def analyze_photo(image_path: str, model_path: str) -> tuple[dict, dict]:
    """Совместимый запуск через текущий MediaPipe backend."""
    from portrait_core.adapters.mediapipe_adapter import MediaPipeAdapter

    return analyze_photo_with_adapter(image_path, MediaPipeAdapter(model_path))

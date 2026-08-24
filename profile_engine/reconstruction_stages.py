"""Optional Profile Engine stages for canonical pseudo-3D research artifacts."""

from __future__ import annotations

from typing import Any

from portrait_core.archive.common import read_json, write_json
from portrait_core.reconstruction_3d import build_reconstruction
from portrait_core.reconstruction_3d.frame_selection import (
    SelectionConfig,
    load_source_frames,
    select_multi_view_frames,
)
from profile_engine.stages import BaseStage


def _selection_config(context) -> SelectionConfig:
    return SelectionConfig(
        min_frames=int(context.config.get("min_3d_frames", 3)),
        max_frames=int(context.config.get("max_3d_frames", 21)),
        max_frames_per_pose_bin=int(context.config.get("max_frames_per_pose_bin", 3)),
    )


def _ensure(context) -> dict[str, Any]:
    return build_reconstruction(
        context.dataset_path,
        scale_mode=context.config.get("scale_mode", "unit_ipd"),
        fusion_method=context.config.get("fusion_method", "median"),
        selection_config=_selection_config(context),
        generate_projections=True,
        force=bool(context.config.get("force_3d")),
    )


class Select3DFramesStage(BaseStage):
    name = "select_3d_frames"

    def run(self, context):
        frames = load_source_frames(context.dataset_path)
        selected, excluded, coverage = select_multi_view_frames(frames, _selection_config(context))
        output = context.dataset_path / "reconstruction_3d" / "selected_frames.json"
        if not context.config.get("dry_run"):
            write_json(output, {
                "schema": "orion.reconstruction_frame_selection.v1",
                "dataset_id": context.dataset_id,
                "profile": "multi_view_neutral",
                "selected": [frame.to_dict() for frame in selected],
                "excluded": excluded,
                "coverage": coverage,
            })
            context.add_artifact(self.name, output)
        return self.result("completed", actions=["multi-view neutral frames selected"], stats=coverage, artifacts=[context.relative_path(output)])


class BuildCanonical3DStage(BaseStage):
    name = "build_canonical_3d"

    def run(self, context):
        model = _ensure(context)
        output = context.dataset_path / "reconstruction_3d" / "canonical_face_3d.json"
        context.add_artifact("canonical_face_3d", output)
        return self.result("completed", actions=["canonical relative-depth pseudo-3D built"], stats={"vertices": len(model["vertices"]), "accepted_frames": len(model["source_frames"])}, artifacts=[context.relative_path(output)])


class ValidateCanonical3DStage(BaseStage):
    name = "validate_canonical_3d"

    def run(self, context):
        model = _ensure(context)
        validation = model.get("validation") or {}
        output = context.dataset_path / "reconstruction_3d" / "reconstruction_report.json"
        context.add_artifact("reconstruction_report", output)
        return self.result("completed" if validation.get("readiness") != "not_ready" else "warning", actions=["canonical reconstruction validated"], warnings=list(validation.get("readiness_reasons") or []), stats=validation, artifacts=[context.relative_path(output)])


class BuildStandardizedProjectionsStage(BaseStage):
    name = "build_standardized_projections"

    def run(self, context):
        _ensure(context)
        paths = sorted((context.dataset_path / "reconstruction_3d" / "projections").glob("*.json"))
        for path in paths: context.add_artifact("standardized_projection", path)
        return self.result("completed", actions=["standardized 2D projections built"], stats={"projections": len(paths)}, artifacts=[context.relative_path(path) for path in paths])


class Build3DMeasurementsStage(BaseStage):
    name = "build_3d_measurements"

    def run(self, context):
        model = _ensure(context)
        output = context.dataset_path / "reconstruction_3d" / "measurements_3d.json"
        if not context.config.get("dry_run"):
            write_json(output, {"schema": "orion.measurements_3d.v1", "source_reconstruction": model["reconstruction_id"], "scale_mode": model["scale_mode"], "measurements": model["measurements_3d"], "metric_units": False})
            context.add_artifact("measurements_3d", output)
        return self.result("completed", actions=["relative 3D measurements built"], stats={"measurements": len(model["measurements_3d"])}, artifacts=[context.relative_path(output)])


class Compare2D3DStage(BaseStage):
    name = "compare_2d_3d"

    def run(self, context):
        _ensure(context)
        output = context.dataset_path / "reconstruction_3d" / "3d_vs_2d_comparison.json"
        payload = read_json(output)
        context.add_artifact("3d_vs_2d_comparison", output)
        answer = payload.get("research_answer") or {}
        return self.result("completed", actions=["2D versus canonical geometry stability compared"], stats={"mean_relative_cv_improvement": payload.get("mean_relative_cv_improvement"), "supports_hypothesis": answer.get("supports_hypothesis")}, artifacts=[context.relative_path(output)])


def reconstruction_stages() -> list[BaseStage]:
    return [
        Select3DFramesStage(), BuildCanonical3DStage(), ValidateCanonical3DStage(),
        BuildStandardizedProjectionsStage(), Build3DMeasurementsStage(), Compare2D3DStage(),
    ]

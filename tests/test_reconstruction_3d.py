import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from portrait_core.analyzer import analyze_points
from portrait_core.reconstruction_3d.confidence import vertex_statistics
from portrait_core.reconstruction_3d.export import build_reconstruction
from portrait_core.reconstruction_3d.frame_selection import (
    SelectionConfig,
    load_source_frames,
    select_multi_view_frames,
)
from portrait_core.reconstruction_3d.mesh_alignment import align_meshes, rigid_align
from portrait_core.reconstruction_3d.mesh_fusion import fuse_meshes
from portrait_core.reconstruction_3d.models import SourceMeshFrame
from portrait_core.reconstruction_3d.pose_normalization import normalize_frame
from portrait_core.reconstruction_3d.projection import project_model, rotation_matrix
from portrait_core.reconstruction_3d.validation import ReconstructionError, validate_common_topology
from profile_engine.runner import run_profile_engine


NAMES = (
    "face_left", "face_right", "face_top", "chin",
    "left_eye_outer", "left_eye_inner", "right_eye_inner", "right_eye_outer",
    "nose_tip", "nose_bridge", "nose_left", "nose_right",
    "mouth_left", "mouth_right", "upper_lip", "lower_lip",
    "jaw_left", "jaw_right", "left_brow_outer", "left_brow_inner",
    "right_brow_inner", "right_brow_outer",
)
SEMANTIC_MAP = {name: index for index, name in enumerate(NAMES)}
BASE = np.asarray([
    [-1.00, 0.00, 0.00], [1.00, 0.00, 0.00], [0.00, -1.20, -0.05], [0.00, 1.35, 0.00],
    [-0.65, -0.35, 0.00], [-0.25, -0.35, -0.02], [0.25, -0.35, -0.02], [0.65, -0.35, 0.00],
    [0.00, 0.18, -0.32], [0.00, -0.42, -0.10], [-0.20, 0.16, -0.10], [0.20, 0.16, -0.10],
    [-0.42, 0.62, -0.02], [0.42, 0.62, -0.02], [0.00, 0.58, -0.08], [0.00, 0.63, -0.06],
    [-0.78, 0.85, 0.00], [0.78, 0.85, 0.00], [-0.68, -0.62, 0.00], [-0.24, -0.58, -0.03],
    [0.24, -0.58, -0.03], [0.68, -0.62, 0.00],
], dtype=float)


def source_frame(vertices=None, *, pfr_id="PFR-1", yaw=0.0, pose_available=True, topology="synthetic-22"):
    pose = {"yaw_deg": yaw, "pitch_deg": 0.0, "roll_deg": 0.0, "pose_available": pose_available}
    return SourceMeshFrame(
        dataset_id="DS-test", pfr_id=pfr_id, pfr_uuid=pfr_id, source_pfr_path=f"pfr/{pfr_id}.json",
        frame_index=0, timestamp_seconds=0.0, vertices=(BASE if vertices is None else vertices).tolist(),
        semantic_landmarks=dict(SEMANTIC_MAP), head_pose=pose,
        quality={"status": "passed", "metrics": {"face_width_px": 200.0, "sharpness_score": 1.0}},
        expression_metrics={"mouth_opening_ratio": 0.05}, selection_score=1.0,
        topology={"schema": "portrait-mesh", "schema_version": "1.0", "source_topology": topology},
    )


def write_synthetic_dataset(root: Path, yaws=(-25.0, -10.0, 0.0, 10.0, 25.0)) -> Path:
    dataset = root / "DS-synthetic"
    (dataset / "pfr").mkdir(parents=True)
    items = []
    for index, yaw in enumerate(yaws):
        rotated = (rotation_matrix(yaw, 0.0, 0.0) @ BASE.T).T
        observed = rotated * 100.0 + np.array([320.0, 240.0, 0.0])
        points = {name: observed[position, :2].tolist() for name, position in SEMANTIC_MAP.items()}
        pfr = {
            "id": f"PFR-{index}", "uuid": f"uuid-{index}", "dataset_id": "DS-synthetic",
            "input": {"frame": index, "timestamp": index / 10.0},
            "mesh": {
                "schema": "portrait-mesh", "schema_version": "1.0", "dimensions": 3,
                "vertices": observed.tolist(), "semantic_map": SEMANTIC_MAP,
                "source": {"topology": "synthetic-22"},
            },
            "canonical_mesh": {"pose": {"yaw_deg": yaw, "pitch_deg": 0.0, "roll_degrees": 0.0}},
            "quality": {"status": "passed", "metrics": {"face_width_px": 200.0, "sharpness_score": 1.0}},
            "measurements": analyze_points(points)["measurements"],
            "zones": {"assignments": {"mouth": [12, 13, 14, 15], "jaw": [16, 17], "nose": [8, 9, 10, 11]}},
        }
        relative = f"pfr/frame_{index}.json"
        (dataset / relative).write_text(json.dumps(pfr), encoding="utf-8")
        items.append({"pfr_path": relative, "status": "passed"})
    (dataset / "dataset.json").write_text(json.dumps({"id": "DS-synthetic", "items": items}), encoding="utf-8")
    return dataset


class Reconstruction3DTestCase(unittest.TestCase):
    def test_normalization_supports_ipd_and_face_width(self):
        transformed = BASE * 80.0 + np.array([300.0, 200.0, 4.0])
        frame = source_frame(transformed)
        ipd, report = normalize_frame(frame, "unit_ipd")
        face, _ = normalize_frame(frame, "unit_face_width")
        left = (ipd[4] + ipd[5]) / 2
        right = (ipd[6] + ipd[7]) / 2
        self.assertAlmostEqual(float(np.linalg.norm(right - left)), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(face[0] - face[1])), 1.0)
        self.assertEqual(len(report["transformation_matrix"]), 4)

    def test_rigid_alignment_recovers_known_transform(self):
        rotation = rotation_matrix(22.0, -7.0, 4.0)
        moved = (rotation @ BASE.T).T + np.array([2.0, -3.0, 0.5])
        aligned, report = rigid_align(moved, BASE, list(range(len(BASE))))
        self.assertLess(np.max(np.abs(aligned - BASE)), 1e-10)
        self.assertLess(report["alignment_error"], 1e-10)

    def test_robust_median_fusion_rejects_mouth_outlier(self):
        meshes = [BASE.copy() for _ in range(5)]
        meshes[-1][14:16, 1] += 4.0
        fused, stats = fuse_meshes(meshes, [1.0] * 5, "median")
        self.assertTrue(np.allclose(fused, BASE))
        self.assertEqual(stats["observation_count"], 5)

    def test_weighted_fusion_and_confidence(self):
        fused, stats = fuse_meshes([BASE, BASE + 1.0], [3.0, 1.0], "weighted_mean")
        self.assertTrue(np.allclose(fused, BASE + 0.25))
        support, confidence, _ = vertex_statistics(len(BASE), 2, stats, {"mouth": [14], "stable_core": [9]})
        self.assertEqual(support, [2] * len(BASE))
        self.assertGreaterEqual(confidence[9], confidence[14])

    def test_topology_mismatch_is_rejected(self):
        with self.assertRaises(ReconstructionError):
            validate_common_topology([source_frame(pfr_id="a"), source_frame(pfr_id="b", topology="other")])

    def test_missing_pose_is_excluded_and_insufficient_is_reported(self):
        frames = [source_frame(pfr_id="a", pose_available=False), source_frame(pfr_id="b", yaw=10.0)]
        with self.assertRaisesRegex(ValueError, "insufficient frames"):
            select_multi_view_frames(frames, SelectionConfig(min_frames=2))

    def test_missing_pose_bins_are_reported_without_fabrication(self):
        frames = [source_frame(pfr_id=f"f{i}", yaw=yaw) for i, yaw in enumerate((-10.0, 0.0, 10.0))]
        selected, _, coverage = select_multi_view_frames(frames, SelectionConfig(min_frames=3))
        self.assertEqual(len(selected), 3)
        self.assertEqual(coverage["pose_bins_available"], 3)
        self.assertLess(coverage["pose_coverage_score"], 1.0)

    def test_projection_is_deterministic_for_both_camera_models(self):
        model = {
            "reconstruction_id": "R3D-test", "vertices": BASE.tolist(),
            "topology": {"semantic_map": SEMANTIC_MAP}, "vertex_confidence": [1.0] * len(BASE),
        }
        first = project_model(model)
        second = project_model(model)
        perspective = project_model(model, projection_type="perspective")
        self.assertEqual(first, second)
        self.assertNotEqual(first["vertices_2d"], perspective["vertices_2d"])

    def test_scientific_sanity_fusion_reduces_random_noise(self):
        rng = np.random.default_rng(7)
        meshes = []
        for yaw in (-18.0, -9.0, 0.0, 9.0, 18.0, 27.0, -27.0):
            noisy = BASE + rng.normal(0.0, 0.025, BASE.shape)
            meshes.append((rotation_matrix(yaw, 0.0, 0.0) @ noisy.T).T + rng.normal(0.0, 0.2, 3))
        aligned, reports = align_meshes(meshes, SEMANTIC_MAP, max_error=1.0)
        fused, _ = fuse_meshes(aligned, [1.0] * len(aligned), "median")
        fused_to_base, _ = rigid_align(fused, BASE, list(range(len(BASE))))
        individual_errors = []
        for mesh in aligned:
            value, _ = rigid_align(mesh, BASE, list(range(len(BASE))))
            individual_errors.append(float(np.sqrt(np.mean((value - BASE) ** 2))))
        fused_error = float(np.sqrt(np.mean((fused_to_base - BASE) ** 2)))
        self.assertLess(fused_error, min(individual_errors))
        self.assertTrue(all(report["accepted"] for report in reports))

    def test_full_export_contract_and_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = write_synthetic_dataset(Path(directory))
            config = SelectionConfig(min_frames=3, max_frames=7, max_frames_per_pose_bin=2)
            model = build_reconstruction(dataset, selection_config=config)
            again = build_reconstruction(dataset, selection_config=config)
            output = dataset / "reconstruction_3d"
            self.assertEqual(model, again)
            self.assertEqual(model["schema"], "orion.canonical_face_3d.v1")
            self.assertEqual(model["alignment"]["fusion"]["visible_view_count"], [5] * len(BASE))
            self.assertIn("used_anchor_points", model["source_frames"][0])
            self.assertEqual(model["coordinate_system"], "canonical_right_handed_relative_depth")
            self.assertEqual(model["scale_mode"], "unit_ipd")
            self.assertEqual(len(model["source_frames"]), 5)
            self.assertTrue((output / "canonical_face_3d.obj").is_file())
            self.assertTrue((output / "reconstruction_report.json").is_file())
            self.assertTrue((output / "projection_measurements.json").is_file())
            self.assertTrue((output / "3d_vs_2d_comparison.json").is_file())
            self.assertEqual(len(list((output / "projections").glob("*.json"))), 7)
            comparison = json.loads((output / "3d_vs_2d_comparison.json").read_text(encoding="utf-8"))
            self.assertIn("research_answer", comparison)
            cached = json.loads((output / "canonical_face_3d.json").read_text(encoding="utf-8"))
            cached["sentinel"] = "must be replaced"
            (output / "canonical_face_3d.json").write_text(json.dumps(cached), encoding="utf-8")
            rebuilt = build_reconstruction(dataset, selection_config=config, force=True)
            self.assertNotIn("sentinel", rebuilt)

    def test_loader_marks_absent_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = write_synthetic_dataset(Path(directory), yaws=(0.0,))
            pfr_path = dataset / "pfr" / "frame_0.json"
            pfr = json.loads(pfr_path.read_text(encoding="utf-8"))
            del pfr["canonical_mesh"]
            pfr_path.write_text(json.dumps(pfr), encoding="utf-8")
            self.assertFalse(load_source_frames(dataset)[0].head_pose["pose_available"])

    def test_profile_engine_runs_explicit_reconstruction_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = write_synthetic_dataset(Path(directory))
            stages = (
                "select_3d_frames,build_canonical_3d,validate_canonical_3d,"
                "build_standardized_projections,build_3d_measurements,compare_2d_3d"
            )
            result = run_profile_engine(dataset, {"stages": stages})
            self.assertEqual(result["status"], "completed")
            self.assertEqual([stage["name"] for stage in result["stages"]], stages.split(","))
            self.assertTrue((dataset / "reconstruction_3d" / "measurements_3d.json").is_file())

if __name__ == "__main__":
    unittest.main()

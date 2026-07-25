import json
import tempfile
import unittest

from portrait_core.archive.common import new_uuid, write_json
from portrait_core.archive.dataset import create_dataset_archive, write_dataset_files
from profile_engine.runner import run_profile_engine


def minimal_pfr(pfr_id="PFR-one"):
    return {
        "id": pfr_id,
        "uuid": new_uuid(),
        "dataset_id": "DS-test",
        "image": "one.jpg",
        "points": {
            "left_eye_inner": [100.0, 100.0],
            "right_eye_inner": [180.0, 100.0],
            "nose_tip": [140.0, 150.0],
        },
        "lic_core": {
            "recommended_base": "ipd",
            "base_candidates": {
                "ipd": {"value": 80.0, "available": True},
            },
            "limitations": [],
        },
        "quality": {"status": "passed", "issues": []},
        "morphology": {"symmetry": "test"},
        "measurements": {
            "face": {"face_width": 200.0, "face_height": 300.0},
            "eyes": {"eye_distance": 80.0},
            "nose": {"nose_length": 60.0, "nose_width": 40.0},
            "mouth": {"mouth_width": 70.0},
            "jaw": {"jaw_width": 130.0, "chin_height": 40.0},
            "forehead": {"forehead_height": 50.0, "forehead_width": 120.0},
        },
    }


def write_dataset_with_pfr(root, *, status="passed"):
    dataset_dir, dataset = create_dataset_archive(root, dataset_id="DS-test")
    pfr_path = dataset_dir / "pfr" / "one_portrait.json"
    pfr = minimal_pfr()
    pfr["quality"]["status"] = status
    write_json(pfr_path, pfr)
    dataset["items"].append(
        {
            "pfr_id": pfr["id"],
            "pfr_uuid": pfr["uuid"],
            "image_path": "images/one.jpg",
            "pfr_path": "pfr/one_portrait.json",
            "status": status,
            "issues": [],
        }
    )
    write_dataset_files(dataset_dir, dataset)
    return dataset_dir


class ProfileEngineRunnerTestCase(unittest.TestCase):
    def test_runner_builds_artifacts_from_existing_pfr(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = write_dataset_with_pfr(directory)

            result = run_profile_engine(dataset_dir, config={"skip_pfr": True})
            manifest = json.loads((dataset_dir / "engine_run.json").read_text(encoding="utf-8"))
            updated_dataset = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
            stats = json.loads((dataset_dir / "invariant_stats.json").read_text(encoding="utf-8"))
            readiness = json.loads((dataset_dir / "invariant_readiness.json").read_text(encoding="utf-8"))

            self.assertEqual(result["status"], "completed")
            self.assertTrue((dataset_dir / "invariants" / "one_portrait_invariants.json").is_file())
            self.assertTrue((dataset_dir / "invariants" / "stats.json").is_file())
            self.assertTrue((dataset_dir / "invariant_stats.json").is_file())
            self.assertTrue((dataset_dir / "invariant_readiness.json").is_file())
            self.assertTrue((dataset_dir / "experiments" / "lic_stability.json").is_file())
            self.assertTrue((dataset_dir / "experiments" / "report_pack.json").is_file())
            self.assertEqual(updated_dataset["items"][0]["invariants_path"], "invariants/one_portrait_invariants.json")
            self.assertIn("all_valid_pfr", stats["samples"])
            self.assertEqual(readiness["research_readiness"]["status"], "not_ready")
            self.assertEqual(manifest["schema"], "profile.engine_run.v1")
            self.assertEqual(manifest["stages"][0]["name"], "validate_dataset")
            self.assertTrue(any(stage["name"] == "build_invariant_readiness" for stage in manifest["stages"]))

    def test_runner_is_idempotent_for_existing_invariants(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = write_dataset_with_pfr(directory)

            run_profile_engine(dataset_dir, config={"skip_pfr": True, "skip_lic": True, "skip_report_pack": True})
            invariants_path = dataset_dir / "invariants" / "one_portrait_invariants.json"
            first_payload = json.loads(invariants_path.read_text(encoding="utf-8"))
            first_payload["sentinel"] = "keep me"
            write_json(invariants_path, first_payload)

            run_profile_engine(dataset_dir, config={"skip_pfr": True, "skip_lic": True, "skip_report_pack": True})
            second_payload = json.loads(invariants_path.read_text(encoding="utf-8"))

        self.assertEqual(second_payload["sentinel"], "keep me")


if __name__ == "__main__":
    unittest.main()
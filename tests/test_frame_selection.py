import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apps.dataset_builder.builder import InputMediaCollection, build_dataset
from apps.dataset_builder.frame_selection import (
    FrontalNeutralThresholds,
    SelectionConfig,
    _evaluate_candidate,
    _select_candidates,
)


def _candidate(frame_index, score=0.9, timestamp=None, width=260, height=300):
    return {
        "frame_index": frame_index,
        "timestamp_seconds": frame_index / 25.0 if timestamp is None else timestamp,
        "track_id": "track-001",
        "track_stability": 1.0,
        "face_confidence": 1.0,
        "face_width_px": width,
        "face_height_px": height,
        "face_area_ratio": 0.08,
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "mouth_open_score": 0.0,
        "eyes_open_score": 1.0,
        "gaze_camera_score": 1.0,
        "sharpness_value": 100.0,
        "brightness_value": 120.0,
        "motion_score": 0.0,
        "occlusion_score": 0.0,
        "bbox": (100.0 + frame_index, 100.0, 260.0, 300.0),
        "candidate_score": score,
        "rejection_reasons": [],
    }


class FrameSelectionTestCase(unittest.TestCase):
    def test_rejects_profile_threshold_violations_with_reasons(self):
        candidate = _candidate(10)
        candidate["yaw_deg"] = 30.0
        candidate["mouth_open_score"] = 0.8

        _evaluate_candidate(candidate, SelectionConfig())

        self.assertIn("yaw_out_of_range", candidate["rejection_reasons"])
        self.assertIn("mouth_open", candidate["rejection_reasons"])
        self.assertLess(candidate["candidate_score"], 1.0)
        self.assertIn("pose", candidate["score_components"])

    def test_temporal_distance_prevents_neighbor_dominance(self):
        cfg = SelectionConfig(
            target_selected_frames=3,
            min_temporal_distance_seconds=0.5,
            max_frames_per_episode=10,
        )
        candidates = [
            _candidate(0, score=0.99),
            _candidate(2, score=0.98),
            _candidate(20, score=0.97),
            _candidate(40, score=0.96),
        ]

        selected = _select_candidates(candidates, cfg, fps=25.0)

        self.assertEqual([item["frame_index"] for item in selected], [0, 20, 40])

    @patch("apps.dataset_builder.builder.create_portrait_report")
    @patch("apps.dataset_builder.builder.collect_input_media")
    def test_quality_profile_metadata_and_pfr_only_for_selected_frames(self, collect_mock, report_mock):
        report_mock.return_value = {
            "schema_version": 3,
            "id": "PFR-test",
            "uuid": "12345678-1234-5678-1234-567812345678",
            "quality": {
                "status": "passed",
                "issues": [],
                "issue_codes": [],
                "metrics": {
                    "face_width_px": 260.0,
                    "face_height_px": 300.0,
                    "face_area_ratio": 0.08,
                },
            },
            "lic_core": {},
            "morphology": {},
            "measurements": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "selected_frame000010.jpg"
            selected.write_bytes(b"image")
            selection = {
                "mode": "quality_profile",
                "profile": "frontal_neutral",
                "candidate_frames": 10,
                "eligible_frames": 4,
                "selected_frames": 1,
                "target_frames": 100,
                "min_temporal_distance_seconds": 0.5,
                "thresholds": {},
                "score_version": "1.0",
            }
            collect_mock.return_value = InputMediaCollection(
                [selected],
                None,
                [],
                selection,
            )

            summary = build_dataset(
                "video.mp4",
                str(root / "dataset"),
                frame_selection_mode="quality_profile",
                target_selected_frames=100,
            )
            dataset = json.loads(
                (Path(summary["dataset_dir"]) / "dataset.json").read_text(encoding="utf-8")
            )

        self.assertEqual(report_mock.call_count, 1)
        self.assertEqual(dataset["selection"]["mode"], "quality_profile")
        self.assertEqual(summary["selection"]["selected_frames"], 1)


if __name__ == "__main__":
    unittest.main()

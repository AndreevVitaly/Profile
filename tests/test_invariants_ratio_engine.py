import math
import unittest

from portrait_core.invariants.ratio_engine import build_invariant_set_from_pfr


class InvariantsRatioEngineTestCase(unittest.TestCase):
    def test_calculates_ratio_from_existing_measurements(self):
        pfr = {
            "id": "PFR-test",
            "dataset_id": "DS-test",
            "measurements": {
                "face": {"face_width": 200.0, "face_height": 300.0},
                "eyes": {"eye_distance": 80.0},
            },
        }

        result = build_invariant_set_from_pfr(pfr)

        self.assertEqual(result.ratios["ipd_face_width"].value, 0.4)
        self.assertEqual(result.ratios["face_height_face_width"].value, 1.5)
        self.assertEqual(result.pfr_id, "PFR-test")
        self.assertEqual(result.dataset_id, "DS-test")

    def test_skips_missing_numerator_without_breaking_process(self):
        pfr = {
            "measurements": {
                "face": {"face_width": 200.0, "face_height": 300.0},
            },
        }

        result = build_invariant_set_from_pfr(pfr)

        self.assertIn("ipd_face_width", result.ratios)
        self.assertFalse(result.ratios["ipd_face_width"].valid)
        self.assertTrue(any("missing numerator interpupillary_distance" in item for item in result.warnings))

    def test_skips_zero_denominator(self):
        pfr = {
            "measurements": {
                "face": {"face_width": 0.0, "face_height": 300.0},
                "eyes": {"eye_distance": 80.0},
            },
        }

        result = build_invariant_set_from_pfr(pfr)

        self.assertIn("ipd_face_width", result.ratios)
        self.assertFalse(result.ratios["ipd_face_width"].valid)
        self.assertTrue(any("zero denominator face_width" in item for item in result.warnings))

    def test_records_alias_usage(self):
        pfr = {
            "measurements": {
                "face": {"face_width": 200.0},
                "eyes": {"ipd": 80.0},
            },
        }

        result = build_invariant_set_from_pfr(pfr)

        self.assertEqual(result.ratios["ipd_face_width"].value, 0.4)
        self.assertTrue(result.diagnostics["aliases"])

    def test_skips_non_finite_values(self):
        pfr = {
            "measurements": {
                "face": {"face_width": 200.0, "face_height": math.inf},
                "eyes": {"eye_distance": math.nan},
            },
        }

        result = build_invariant_set_from_pfr(pfr)

        self.assertFalse(result.ratios["ipd_face_width"].valid)
        self.assertFalse(result.ratios["face_height_face_width"].valid)

    def test_skips_incompatible_units(self):
        pfr = {
            "measurements": {
                "face": {"face_width": 200.0, "face_height": 300.0, "units": {"face_width": "px", "face_height": "mm"}},
            },
        }

        result = build_invariant_set_from_pfr(pfr)

        self.assertFalse(result.ratios["face_height_face_width"].valid)
        self.assertEqual(result.ratios["face_height_face_width"].skipped_reason, "incompatible units")


if __name__ == "__main__":
    unittest.main()
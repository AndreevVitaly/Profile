import unittest

from portrait_core.invariants.invariant_stats import build_invariant_stats


def invariant_payload(value):
    return {
        "schema": "profile.invariants.v1",
        "ratios": {
            "ipd_face_width": {
                "value": value,
                "numerator": "ipd",
                "denominator": "face_width",
                "category": "eyes",
                "quality": "ok",
            }
        },
    }


class InvariantStatsTestCase(unittest.TestCase):
    def test_cv_and_excellent_classification(self):
        result = build_invariant_stats(
            [invariant_payload(0.400), invariant_payload(0.402), invariant_payload(0.398)]
        )

        stat = result["stats"]["ipd_face_width"]
        self.assertEqual(stat["count"], 3)
        self.assertLess(stat["cv"], 0.05)
        self.assertEqual(stat["stability_score"], round(1.0 - stat["cv"], 6))
        self.assertEqual(stat["stability_class"], "excellent")

    def test_stability_classes(self):
        cases = [
            ([1.0, 1.08, 1.16], "stable"),
            ([1.0, 1.2, 1.4], "moderate"),
            ([1.0, 1.7, 2.4], "unstable"),
        ]

        for values, expected in cases:
            with self.subTest(expected=expected):
                result = build_invariant_stats([invariant_payload(value) for value in values])
                self.assertEqual(
                    result["stats"]["ipd_face_width"]["stability_class"],
                    expected,
                )

    def test_insufficient_data_with_missing_count(self):
        result = build_invariant_stats([invariant_payload(0.4), {"ratios": {}}])

        stat = result["stats"]["ipd_face_width"]
        self.assertEqual(stat["count"], 1)
        self.assertEqual(stat["missing_count"], 1)
        self.assertEqual(stat["stability_class"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()


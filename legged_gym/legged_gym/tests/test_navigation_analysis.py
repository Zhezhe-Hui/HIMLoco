import unittest

from legged_gym.scripts.analyze_navigation_benchmark import (
    analyze,
    exact_mcnemar_p,
    failure_category,
    wilson_interval,
)


class NavigationAnalysisTest(unittest.TestCase):
    def test_exact_mcnemar(self):
        self.assertAlmostEqual(exact_mcnemar_p(13, 0), 2.0 / (2 ** 13))
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(16, 30)
        self.assertLess(low, 16.0 / 30.0)
        self.assertGreater(high, 16.0 / 30.0)

    def test_failure_category_prioritizes_collision(self):
        record = {
            "success": False,
            "failure_reason": "碰撞(接触力20N)、卡住",
        }
        self.assertEqual(failure_category(record), "collision")

    def test_analysis_requires_paired_seeds(self):
        records = [
            {"method": "mppi", "seed": 1, "success": False,
             "failure_reason": "Timeout"},
            {"method": "fmm", "seed": 2, "success": False,
             "failure_reason": "Timeout"},
        ]
        with self.assertRaisesRegex(ValueError, "unpaired seed set"):
            analyze(records)

    def test_analysis_rejects_mismatched_layout_hash(self):
        records = [
            {"method": "mppi", "seed": 1, "success": False,
             "failure_reason": "Timeout", "obstacle_layout_hash": "aaa"},
            {"method": "fmm", "seed": 1, "success": False,
             "failure_reason": "Timeout", "obstacle_layout_hash": "bbb"},
        ]
        with self.assertRaisesRegex(ValueError, "obstacle layout mismatch"):
            analyze(records)


if __name__ == "__main__":
    unittest.main()

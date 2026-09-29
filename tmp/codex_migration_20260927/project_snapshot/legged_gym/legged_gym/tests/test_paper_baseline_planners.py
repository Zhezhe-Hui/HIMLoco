import unittest

import numpy as np

from legged_gym.envs.go2.paper_baseline_planners import (
    PatternSearchDWAController,
    SafeAPFController,
)


class SafeAPFControllerTest(unittest.TestCase):
    def make_controller(self):
        return SafeAPFController(
            bev_res=0.1,
            bev_x=6.0,
            bev_y=6.0,
            max_path_steps=80,
        )

    def test_open_space_points_toward_goal(self):
        planner = self.make_controller()
        planner.set_goal((4.0, 0.0))
        planner.update(np.zeros((planner.H, planner.W), dtype=np.uint8))

        self.assertAlmostEqual(planner.compute_wz(), 0.0, places=5)
        self.assertGreater(planner.compute_vx(), 0.9)
        self.assertGreater(len(planner.extract_path()), 5)

    def test_obstacle_creates_vortex_without_crossing_safety_region(self):
        planner = self.make_controller()
        planner.set_goal((4.0, 0.0))
        occupancy = np.zeros((planner.H, planner.W), dtype=np.uint8)
        occupancy[7:10, planner.robot_c - 1:planner.robot_c + 2] = 1
        planner.update(occupancy)

        self.assertGreater(abs(planner.compute_wz()), 0.05)
        self.assertTrue(planner.extract_path())
        self.assertTrue(all(
            not planner.hard_obstacle_map[r, c]
            for r, c in planner.extract_path()
        ))


class PatternSearchDWATest(unittest.TestCase):
    def test_pattern_search_respects_dynamic_window(self):
        planner = PatternSearchDWAController(
            bev_res=0.1,
            bev_x=6.0,
            bev_y=6.0,
            dt=0.1,
            max_evaluations=32,
        )
        planner.set_goal((4.0, 0.5))
        planner.update(np.zeros((planner.H, planner.W), dtype=np.uint8))

        self.assertGreaterEqual(planner.compute_vx(), 0.0)
        self.assertLessEqual(planner.compute_vx(), planner.max_acc_vx * planner.dt + 1e-9)
        self.assertLessEqual(abs(planner.compute_wz()), planner.max_acc_wz * planner.dt + 1e-9)
        self.assertLessEqual(planner.last_evaluation_count, 32)
        self.assertLess(planner.last_evaluation_count, 10 * 15)
        self.assertTrue(planner.is_reachable())

    def test_long_range_goal_is_clipped_to_local_bev(self):
        planner = PatternSearchDWAController(bev_res=0.1, bev_x=6.0, bev_y=6.0)
        planner.set_goal((24.0, 0.0))
        planner.update(np.zeros((planner.H, planner.W), dtype=np.uint8))

        self.assertEqual(planner.goal_rc[0], planner.H - 1)
        self.assertTrue(planner.is_reachable())
        self.assertGreater(planner.compute_vx(), 0.0)


if __name__ == "__main__":
    unittest.main()

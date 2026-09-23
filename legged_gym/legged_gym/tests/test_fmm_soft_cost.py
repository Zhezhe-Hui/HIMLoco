import unittest

import numpy as np
from scipy.ndimage import distance_transform_edt

from legged_gym.envs.go2.bev_height_mapper import FMMGradientController


class FMMSoftCostTest(unittest.TestCase):
    RES = 0.1
    SIZE_M = 6.0
    CELLS = 60

    def make_controller(self, *, soft, inflate=0.2, hard=0.2):
        return FMMGradientController(
            bev_res=self.RES,
            bev_x=self.SIZE_M,
            bev_y=self.SIZE_M,
            inflate_radius_m=inflate,
            use_soft_cost=soft,
            hard_radius_m=hard,
            soft_clearance_m=0.8,
            soft_cost_weight=6.0,
            soft_cost_power=2.0,
        )

    def plan(self, controller, occ):
        controller.set_goal((5.0, 0.0))
        controller.update(occ)
        return controller.extract_path(max_steps=200)

    def test_soft_cost_increases_clearance_around_block(self):
        occ = np.zeros((self.CELLS, self.CELLS), dtype=np.uint8)
        occ[12:30, 27:34] = 1

        fixed = self.make_controller(soft=False)
        soft = self.make_controller(soft=True)
        fixed_path = self.plan(fixed, occ)
        soft_path = self.plan(soft, occ)

        self.assertEqual(fixed_path[-1], fixed.goal_rc)
        self.assertEqual(soft_path[-1], soft.goal_rc)
        clearance = distance_transform_edt(occ == 0) * self.RES
        fixed_min = min(clearance[r, c] for r, c in fixed_path)
        soft_min = min(clearance[r, c] for r, c in soft_path)
        self.assertGreater(soft_min, fixed_min + 0.2)

    def test_dual_layer_keeps_narrow_gate_reachable(self):
        occ = np.zeros((self.CELLS, self.CELLS), dtype=np.uint8)
        occ[28:31, :27] = 1
        occ[28:31, 34:] = 1

        large_fixed = self.make_controller(soft=False, inflate=0.4)
        dual_layer = self.make_controller(soft=True, hard=0.2)
        fixed_path = self.plan(large_fixed, occ)
        dual_path = self.plan(dual_layer, occ)

        self.assertNotEqual(fixed_path[-1], large_fixed.goal_rc)
        self.assertGreaterEqual(large_fixed.fmm_dist[0, large_fixed.robot_c], large_fixed.large)
        self.assertEqual(dual_path[-1], dual_layer.goal_rc)
        self.assertLess(dual_layer.fmm_dist[0, dual_layer.robot_c], dual_layer.large)

    def test_extracted_path_is_strictly_descending(self):
        occ = np.zeros((self.CELLS, self.CELLS), dtype=np.uint8)
        occ[12:30, 27:34] = 1
        controller = self.make_controller(soft=True)
        path = self.plan(controller, occ)
        values = [controller.fmm_dist[r, c] for r, c in path]

        self.assertLess(len(path), 200)
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_unreachable_wall_is_detected_and_turns_in_place(self):
        occ = np.zeros((self.CELLS, self.CELLS), dtype=np.uint8)
        occ[15:18, :] = 1
        controller = FMMGradientController(
            bev_res=self.RES,
            bev_x=self.SIZE_M,
            bev_y=self.SIZE_M,
            use_soft_cost=True,
            hard_radius_m=0.2,
            soft_clearance_m=0.8,
            max_wz_accel=3.0,
            control_dt=0.02,
            unreachable_turn_wz=0.6,
        )
        self.plan(controller, occ)

        self.assertFalse(controller.is_reachable())
        self.assertAlmostEqual(abs(controller.compute_wz()), 0.06, places=6)

    def test_wz_rate_limit_prevents_instant_direction_flip(self):
        controller = FMMGradientController(
            max_wz=1.0, max_wz_accel=3.0, control_dt=0.02)

        first = controller._rate_limit_wz(1.0)
        second = controller._rate_limit_wz(-1.0)

        self.assertAlmostEqual(first, 0.06, places=6)
        self.assertAlmostEqual(second, 0.0, places=6)

    def test_path_lookahead_steers_toward_offset_narrow_gate(self):
        occ = np.zeros((self.CELLS, self.CELLS), dtype=np.uint8)
        occ[12:16, :36] = 1
        occ[12:16, 46:] = 1
        controller = self.make_controller(soft=True, hard=0.2)
        controller.set_goal((5.0, 1.2))
        controller.update(occ)

        self.assertTrue(controller.is_reachable())
        self.assertGreater(controller.compute_wz(), 0.0)
        self.assertLess(controller.path_min_clearance(2.0), 0.7)

    def test_open_space_small_goal_error_does_not_trigger_grid_turn(self):
        occ = np.zeros((self.CELLS, self.CELLS), dtype=np.uint8)
        controller = self.make_controller(soft=True, hard=0.2)
        controller.set_goal((24.0, -0.07))
        controller.update(occ)

        self.assertEqual(controller.compute_wz(), 0.0)
        self.assertEqual(controller._last_turn_sign, 0)

    def test_open_space_clears_stale_turn_latch(self):
        occ = np.zeros((self.CELLS, self.CELLS), dtype=np.uint8)
        controller = FMMGradientController(
            bev_res=self.RES,
            bev_x=self.SIZE_M,
            bev_y=self.SIZE_M,
            hysteresis_thresh=0.05,
            hysteresis_hold_steps=40,
            max_wz_accel=0.0,
        )
        controller._last_turn_sign = -1
        controller._turn_sign_age = 1
        controller.set_goal((5.0, 0.5))
        controller.update(occ)

        self.assertGreater(controller.compute_wz(), 0.0)
        self.assertEqual(controller._last_turn_sign, 0)


if __name__ == "__main__":
    unittest.main()

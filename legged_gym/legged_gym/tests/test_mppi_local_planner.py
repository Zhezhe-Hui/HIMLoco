import math
import unittest

import numpy as np

from legged_gym.envs.go2.mppi_local_planner import (
    FootprintMPPIController,
    WorldObstacleMemory,
    choose_safer_turn_sign,
)


class WorldObstacleMemoryTest(unittest.TestCase):
    def test_rotation_reprojects_obstacle_without_smearing(self):
        memory = WorldObstacleMemory(
            bev_res=0.1, bev_x=6.0, bev_y=6.0,
            max_age_steps=20, voxel_size_m=0.1)
        occ = np.zeros((60, 60), dtype=np.uint8)
        occ[20, 40] = 1  # approximately (2.05 m forward, 1.05 m left)
        memory.update(occ, (0.0, 0.0, 0.0), step=0)

        rotated = memory.project((0.0, 0.0, math.pi / 2.0))
        rows, cols = np.nonzero(rotated)

        self.assertEqual(memory.point_count, 1)
        self.assertEqual(rows.size, 1)
        self.assertAlmostEqual(rows[0] * 0.1, 1.0, delta=0.15)
        self.assertAlmostEqual((cols[0] - 30) * 0.1, -2.0, delta=0.15)

    def test_translation_and_expiry(self):
        memory = WorldObstacleMemory(
            bev_res=0.1, bev_x=6.0, bev_y=6.0,
            max_age_steps=3, voxel_size_m=0.1)
        occ = np.zeros((60, 60), dtype=np.uint8)
        occ[20, 30] = 1
        memory.update(occ, (0.0, 0.0, 0.0), step=0)
        shifted = memory.project((1.0, 0.0, 0.0))
        rows, _ = np.nonzero(shifted)
        self.assertAlmostEqual(rows[0] * 0.1, 1.0, delta=0.15)

        memory.update(np.zeros_like(occ), (1.0, 0.0, 0.0), step=4)
        self.assertEqual(memory.point_count, 0)

    def test_reobserved_free_space_clears_ghost_obstacle(self):
        memory = WorldObstacleMemory(
            bev_res=0.1, bev_x=6.0, bev_y=6.0,
            max_age_steps=20, voxel_size_m=0.1)
        occ = np.zeros((60, 60), dtype=np.uint8)
        occ[20, 30] = 1
        memory.update(occ, (0.0, 0.0, 0.0), step=0)

        observed = np.zeros_like(occ, dtype=bool)
        observed[20, 30] = True
        memory.update(
            np.zeros_like(occ), (0.0, 0.0, 0.0), step=1,
            observed_local=observed)

        self.assertEqual(memory.point_count, 0)

    def test_unobserved_space_keeps_obstacle_memory(self):
        memory = WorldObstacleMemory(
            bev_res=0.1, bev_x=6.0, bev_y=6.0,
            max_age_steps=20, voxel_size_m=0.1)
        occ = np.zeros((60, 60), dtype=np.uint8)
        occ[20, 30] = 1
        memory.update(occ, (0.0, 0.0, 0.0), step=0)
        memory.update(
            np.zeros_like(occ), (0.0, 0.0, 0.0), step=1,
            observed_local=np.zeros_like(occ, dtype=bool))

        self.assertEqual(memory.point_count, 1)


class EscapeDirectionTest(unittest.TestCase):
    def test_turns_toward_side_with_less_occupancy(self):
        occ = np.zeros((60, 60), dtype=np.uint8)
        occ[:30, 31:] = 1
        self.assertEqual(choose_safer_turn_sign(occ), -1)

        occ.fill(0)
        occ[:30, :30] = 1
        self.assertEqual(choose_safer_turn_sign(occ), 1)

    def test_equal_occupancy_uses_planner_turn(self):
        occ = np.zeros((60, 60), dtype=np.uint8)
        self.assertEqual(choose_safer_turn_sign(occ, preferred_wz=-0.4), -1)
        self.assertEqual(choose_safer_turn_sign(occ, preferred_wz=0.4), 1)


class FootprintMPPITest(unittest.TestCase):
    def make_controller(self, **kwargs):
        defaults = dict(
            bev_res=0.1, bev_x=6.0, bev_y=6.0,
            horizon=16, num_samples=512, rollout_dt=0.1,
            footprint_length_m=0.7, footprint_width_m=0.4,
            safety_margin_m=0.05, seed=3,
        )
        defaults.update(kwargs)
        return FootprintMPPIController(**defaults)

    def test_footprint_detects_side_swipe_missed_by_center_point(self):
        controller = self.make_controller()
        occ = np.zeros((60, 60), dtype=np.uint8)
        occ[10, 32] = 1  # y=0.2 m: center is free, body side overlaps
        states = np.array([[[1.0, 0.0, 0.0]]])

        self.assertEqual(occ[10, 30], 0)
        self.assertTrue(controller.footprint_collision_mask(states, occ)[0, 0])

    def test_open_space_command_is_fast_and_straight(self):
        controller = self.make_controller()
        occ = np.zeros((60, 60), dtype=np.uint8)
        path = np.column_stack((np.linspace(0.0, 5.0, 51), np.zeros(51)))

        vx, wz = controller.command(occ, path, goal_xy=(5.0, 0.0))

        self.assertGreater(vx, 0.6)
        self.assertLess(abs(wz), 0.08)
        self.assertEqual(controller.last_collision_fraction, 0.0)

    def test_offset_path_produces_bounded_smooth_turn(self):
        controller = self.make_controller()
        occ = np.zeros((60, 60), dtype=np.uint8)
        x = np.linspace(0.0, 5.0, 51)
        path = np.column_stack((x, 0.7 * (1.0 - np.exp(-x))))

        commands = [controller.command(occ, path, goal_xy=(5.0, 0.7))
                    for _ in range(4)]

        self.assertTrue(all(vx > 0.4 for vx, _ in commands))
        self.assertTrue(all(0.0 < wz <= controller.max_wz for _, wz in commands))
        changes = np.abs(np.diff([wz for _, wz in commands]))
        self.assertTrue(np.all(changes <= 0.080001))


if __name__ == "__main__":
    unittest.main()

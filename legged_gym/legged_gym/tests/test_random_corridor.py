import unittest

import numpy as np

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.go2 import viz_config
from legged_gym.envs.go2.go2_config import Go2RoughCfg


class RandomCorridorTest(unittest.TestCase):
    def test_active_scene_has_expected_geometry(self):
        self.assertEqual(viz_config.ACTIVE_SCENE_PRESET, "random_corridor")
        self.assertTrue(viz_config.NAV_RANDOM_STRAIGHT_ROUTE)
        self.assertEqual(viz_config.SPAWN_RANDOM_X_RANGE, (6.0, 6.0))
        self.assertEqual(viz_config.SPAWN_RANDOM_Y_RANGE, (3.0, 9.0))
        self.assertEqual(viz_config.SCENE_RANDOM_OBSTACLE_X, (12.0, 24.0))
        self.assertEqual(viz_config.SCENE_RANDOM_OBSTACLE_Y, (0.0, 12.0))
        self.assertEqual(
            viz_config.SCENE_NUM_BIG_STONES,
            viz_config.SCENE_RANDOM_CORRIDOR_BIG_STONE_COUNT)
        self.assertGreater(viz_config.SCENE_NUM_BIG_STONES, 0)
        self.assertEqual(viz_config.SCENE_TERRAIN_PRESET, "height")
        self.assertFalse(Go2RoughCfg.stone.randomize_each_reset)
        self.assertTrue(Go2RoughCfg.stone.randomize_at_creation)

    def test_obstacle_sampler_respects_bounds_and_separation(self):
        np.random.seed(7)
        count = viz_config.SCENE_NUM_BIG_STONES + 4
        min_separation = viz_config.SCENE_RANDOM_OBSTACLE_MIN_SEPARATION_M
        points = LeggedRobot._sample_separated_xy(
            count, (12.0, 24.0), (0.0, 12.0), min_separation)
        points = np.asarray(points)

        self.assertEqual(points.shape, (count, 2))
        self.assertTrue(np.all((points[:, 0] >= 12.0) & (points[:, 0] <= 24.0)))
        self.assertTrue(np.all((points[:, 1] >= 0.0) & (points[:, 1] <= 12.0)))
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                self.assertGreaterEqual(
                    np.linalg.norm(points[i] - points[j]), min_separation)

    def test_process_seed_is_independent_and_reproducible(self):
        scene_seed = viz_config.SCENE_RANDOM_SEED_BASE + 3
        first = LeggedRobot._sample_separated_xy(
            12, (12.0, 24.0), (0.0, 12.0), 1.4,
            rng=np.random.RandomState(scene_seed + 100000))
        repeated = LeggedRobot._sample_separated_xy(
            12, (12.0, 24.0), (0.0, 12.0), 1.4,
            rng=np.random.RandomState(scene_seed + 100000))
        next_scene = LeggedRobot._sample_separated_xy(
            12, (12.0, 24.0), (0.0, 12.0), 1.4,
            rng=np.random.RandomState(scene_seed + 1 + 100000))

        np.testing.assert_allclose(first, repeated)
        self.assertFalse(np.allclose(first, next_scene))


if __name__ == "__main__":
    unittest.main()

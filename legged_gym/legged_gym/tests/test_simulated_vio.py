import unittest

import numpy as np

from legged_gym.envs.go2.simulated_vio import SimulatedVIO


class SimulatedVIOTest(unittest.TestCase):
    def test_ideal_mode_matches_ground_truth(self):
        vio = SimulatedVIO(dt=0.02, seed=7)
        vio.reset((1.0, 2.0, 0.3))
        for pose in [(1.1, 2.0, 0.31), (1.2, 2.1, -3.0)]:
            np.testing.assert_allclose(vio.update(pose), pose)

    def test_seed_reproduces_noise_and_drift(self):
        kwargs = dict(dt=0.02, position_sigma_m=0.02,
                      yaw_sigma_rad=0.01,
                      position_drift_m_sqrt_s=0.01,
                      yaw_drift_rad_sqrt_s=0.005, seed=19)
        first = SimulatedVIO(**kwargs)
        second = SimulatedVIO(**kwargs)
        first.reset((0.0, 0.0, 0.0))
        second.reset((0.0, 0.0, 0.0))
        seq_a = [first.update((i * 0.1, 0.0, 0.0)) for i in range(10)]
        seq_b = [second.update((i * 0.1, 0.0, 0.0)) for i in range(10)]
        np.testing.assert_allclose(seq_a, seq_b)

    def test_latency_returns_older_pose(self):
        vio = SimulatedVIO(dt=0.02, latency_steps=2, seed=1)
        vio.reset((0.0, 0.0, 0.0))
        self.assertAlmostEqual(vio.update((1.0, 0.0, 0.0))[0], 0.0)
        self.assertAlmostEqual(vio.update((2.0, 0.0, 0.0))[0], 0.0)
        self.assertAlmostEqual(vio.update((3.0, 0.0, 0.0))[0], 1.0)

    def test_full_dropout_holds_last_pose(self):
        vio = SimulatedVIO(dt=0.02, dropout_probability=0.999999, seed=3)
        vio.reset((1.0, 2.0, 0.3))
        np.testing.assert_allclose(vio.update((5.0, 6.0, 0.9)), (1.0, 2.0, 0.3))


if __name__ == "__main__":
    unittest.main()

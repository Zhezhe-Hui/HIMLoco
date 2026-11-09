"""Offline geometry and command rejection tests; no robot/camera access."""
import io
import json
import time
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import run_ours as runtime
from estimate_floor import fit_ground


class RealRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapper, cls.mppi_module, _ = runtime.algorithms()

    def setUp(self):
        self.cal = json.loads((Path(__file__).parent/"calibration.example.json").read_text())
        self.bev = self.mapper.BEVMapper(bev_x=4, bev_y=4, bev_res=0.1,
                                         min_points=1, v_start_ratio=0)
        self.state = dict(x=0, y=0, z=0.31, yaw=0, roll=0, pitch=0,
                          mode=2, state_age=0, received=time.monotonic())

    def test_reject_reflection_and_nonfinite_calibration(self):
        self.cal["camera_rotation_optical_to_body"][1][0] = 1
        with self.assertRaises(ValueError):
            runtime.validate_calibration(self.cal)
        self.cal["camera_translation_body_m"][0] = float("nan")
        with self.assertRaises(ValueError):
            runtime.validate_calibration(self.cal)

    def test_real_intrinsics_and_left_right_geometry(self):
        # Two rays at identical depth, left image ray must map to body left.
        depth = np.array([[1.0, 0, 1.0]], dtype=np.float32)
        runtime.update_real_bev(self.bev, depth, (1, 1, 1, 0), self.state, self.cal)
        cells = np.argwhere(self.bev.count)
        self.assertEqual(len(cells), 2)
        self.assertEqual(set(cells[:, 0]), {13})
        self.assertEqual(set(cells[:, 1]), {10, 30})

    def test_invalid_depth_does_not_become_obstacle(self):
        depth = np.array([[0, np.nan, np.inf, -1, 10]], dtype=np.float32)
        occ = runtime.update_real_bev(self.bev, depth, (1, 1, 2, 0), self.state, self.cal)
        self.assertEqual(int(self.bev.count.sum()), 0)
        self.assertFalse(occ.any())

    def test_native_reduction_matches_numpy_with_duplicate_cells_and_height_edges(self):
        if runtime.bev_reducer() is None:
            self.skipTest('Compile bev_reduce.cpp to exercise native backend')
        rng = np.random.RandomState(104)
        rows = rng.randint(0, self.bev.H, 200000)
        cols = rng.randint(0, self.bev.W, 200000)
        heights = rng.uniform(-.5, 2., len(rows))
        heights[:6] = [.15, np.nextafter(.15, 1), np.nextafter(.15, 0),
                       1.2, np.nextafter(1.2, 2), np.nextafter(1.2, 0)]
        for n in (0, 6, len(rows)):
            self.bev.reset()
            native = runtime.aggregate_points(self.bev, rows[:n], cols[:n], heights[:n])
            expected = [x.copy() for x in (self.bev.min_z, self.bev.max_z, self.bev.count, native)]
            self.bev.reset()
            reference = runtime.aggregate_points(self.bev, rows[:n], cols[:n], heights[:n], use_native=False)
            for a,b in zip(expected, (self.bev.min_z, self.bev.max_z, self.bev.count, reference)):
                np.testing.assert_array_equal(a, b)

    def test_slow_planner_cannot_hide_original_input_age(self):
        from unittest.mock import patch
        clock = [100.]
        state = dict(self.state, received=100.)
        frame = dict(captured=100., received=100., clock_uncertainty=0)
        fmm = Mock(bev_res=.1, W=40)
        fmm.is_reachable.return_value = True
        fmm.extract_path.return_value = [(0,20),(10,20)]
        mppi = Mock(horizon=20)
        def slow(*args, **kwargs):
            clock[0] += .3
            return (.1, 0)
        mppi.command.side_effect = slow
        mppi.footprint_collision_mask.return_value = np.array([False])
        timing = {}
        with patch.object(runtime.time,'monotonic',side_effect=lambda:clock[0]):
            command,reason,*_ = runtime.select_command(self.bev,fmm,mppi,np.ones((8,8)),
                (10,10,4,4),state,frame,self.cal,np.array([.5,0]),True,diagnostics=timing)
        self.assertEqual(command,(0,0))
        self.assertEqual(reason,'stale_input')
        self.assertEqual(timing['stale_sources'],['camera','state'])
        self.assertAlmostEqual(timing['state_age_end_ms'],300)
        self.assertAlmostEqual(timing['mppi_ms'],300)

    def test_world_goal_transform(self):
        self.state.update(x=1, y=2, yaw=np.pi/2)
        np.testing.assert_allclose(runtime.local_goal((1, 4), self.state), (2, 0), atol=1e-9)

    def test_vertical_plane_is_not_free_space(self):
        depth = np.full((60, 80), 0.3, dtype=np.float32)
        occ = runtime.update_real_bev(self.bev, depth, (50, 50, 40, 30), self.state, self.cal)
        self.assertGreater(int(occ.sum()), 0)

    def test_ground_fit_recovers_height_with_depth_noise(self):
        v, u = np.indices((480, 640))
        depth = np.zeros((480, 640), dtype=np.float32)
        good = v > 250
        depth[good] = 0.43*386/(v[good]-240)
        rng = np.random.RandomState(3)
        depth[good] += rng.normal(0, 0.002, size=int(good.sum()))
        normal, height, fraction, rms = fit_ground(depth, (386, 386, 320, 240))
        self.assertAlmostEqual(height, 0.43, delta=0.01)
        self.assertGreater(-normal[1], 0.99)
        self.assertGreater(fraction, 0.9)
        self.assertLess(rms, 0.01)

    def test_vertical_wall_cannot_be_used_as_ground_calibration(self):
        with self.assertRaises(ValueError):
            fit_ground(np.ones((480, 640), dtype=np.float32), (386, 386, 320, 240))

    def plan(self, depth=None, **changes):
        self.state.update(changes)
        fmm = Mock(bev_res=0.1, W=40)
        fmm.is_reachable.return_value = True
        fmm.extract_path.return_value = [(0, 20), (10, 20)]
        mppi = Mock(horizon=20)
        mppi.command.return_value = (0.2, 0.1)
        mppi.footprint_collision_mask.return_value = np.array([False])
        if depth is None:
            depth = np.ones((8, 8))
        return runtime.select_command(self.bev, fmm, mppi, depth,
                       (10, 10, 4, 4), self.state, {"received": time.monotonic()},
                       self.cal, np.array([2, 0]), True)

    def test_stale_odometry_cannot_issue_motion(self):
        command, reason, *_ = self.plan(state_age=1)
        self.assertEqual(command, (0, 0))
        self.assertEqual(reason, "stale_input")

    def test_native_takeover_event_is_not_a_motion_state(self):
        latest=runtime.Latest()
        process=Mock(stdout=io.BytesIO(b'{"bridge_event":"remote_takeover"}\n'))
        runtime.read_state(process,latest)
        with self.assertRaisesRegex(RuntimeError,'bridge_event:remote_takeover'):
            latest.get()

    def test_near_cabinet_stops_even_with_reachable_path(self):
        command, reason, *_ = self.plan(depth=np.full((8, 8), 0.3))
        self.assertEqual(command, (0, 0))
        self.assertEqual(reason, "near_depth_stop")

    def test_depth_holes_in_front_are_not_treated_as_free_space(self):
        command, reason, *_ = self.plan(depth=np.zeros((8, 8)))
        self.assertEqual(command, (0, 0))
        self.assertEqual(reason, "insufficient_front_depth")

    def test_lying_mode_cannot_issue_motion(self):
        command, reason, *_ = self.plan(mode=0, z=0.05)
        self.assertEqual(command, (0, 0))
        self.assertEqual(reason, "robot_not_standing")

    def test_tilt_rejects_motion(self):
        command, reason, *_ = self.plan(roll=0.8)
        self.assertEqual(command, (0, 0))
        self.assertEqual(reason, "excessive_tilt")

    def test_incomplete_transport_frame_is_rejected(self):
        with self.assertRaises(EOFError):
            runtime.read_exact(io.BytesIO(b"abc"), 10)

    def test_existing_ours_plans_open_space(self):
        fmm = self.mapper.FMMGradientController(bev_x=4, bev_y=4, bev_res=0.1,
                     use_soft_cost=True, hard_radius_m=0.2)
        occ = np.zeros((40, 40), dtype=np.uint8)
        fmm.set_goal((2, 0)); fmm.update(occ)
        path = np.asarray([(r*0.1, (c-20)*0.1) for r, c in fmm.extract_path()])
        mppi = self.mppi_module.FootprintMPPIController(bev_x=4, bev_y=4, bev_res=0.1,
                          max_vx=0.2, min_vx=0, max_wz=0.35, num_samples=32)
        vx, wz = mppi.command(occ, path, (2, 0))
        self.assertTrue(fmm.is_reachable())
        self.assertTrue(0 <= vx <= 0.2)
        self.assertTrue(abs(wz) <= 0.35)

    def test_explicit_minimum_speed_stays_within_authorized_cap(self):
        mppi = self.mppi_module.FootprintMPPIController(bev_x=4, bev_y=4, bev_res=0.1,
                          max_vx=0.1, min_vx=0.1, max_wz=0.35, num_samples=32)
        vx, _ = mppi.command(np.zeros((40, 40), dtype=np.uint8), np.array([[0, 0], [0.5, 0]]), (0.5, 0))
        self.assertAlmostEqual(vx, 0.1)


if __name__ == "__main__":
    unittest.main()

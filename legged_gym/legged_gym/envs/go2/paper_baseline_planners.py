"""Paper-inspired local planner baselines adapted to the shared BEV interface.

These implementations follow the algorithms described in the cited papers but
are native Python adaptations for this Isaac Gym evaluation stack.  They are
not copies of the authors' MATLAB implementations.
"""

import numpy as np
import skimage.morphology as morph
from scipy.ndimage import distance_transform_edt

from legged_gym.envs.go2.a_star_dwa_planner import AStarDWAController


class SafeAPFController:
    """Safe artificial potential field with a rotated obstacle force.

    The controller implements the SAPF paper's central mechanism: the ordinary
    repulsive force is rotated by a distance-dependent angle, producing a
    vortex around an obstacle while recovering radial repulsion close to the
    safety boundary.  The occupancy-grid distance transform supplies the
    nearest obstacle distance and direction.
    """

    def __init__(
        self,
        bev_res=0.05,
        bev_x=6.0,
        bev_y=6.0,
        max_vx=1.0,
        max_wz=1.0,
        yaw_k=2.0,
        safe_distance_m=0.30,
        vortex_distance_m=0.70,
        influence_distance_m=1.20,
        attraction_gain=1.0,
        repulsion_gain=0.035,
        integration_step_m=0.10,
        max_path_steps=100,
    ):
        self.bev_res = float(bev_res)
        self.H = int(float(bev_x) / self.bev_res)
        self.W = int(float(bev_y) / self.bev_res)
        self.robot_r = 0
        self.robot_c = self.W // 2
        self.max_vx = float(max_vx)
        self.max_wz = float(max_wz)
        self.yaw_k = float(yaw_k)
        self.safe_distance_m = float(safe_distance_m)
        self.vortex_distance_m = float(vortex_distance_m)
        self.influence_distance_m = float(influence_distance_m)
        self.attraction_gain = float(attraction_gain)
        self.repulsion_gain = float(repulsion_gain)
        self.integration_step_cells = max(1.0, float(integration_step_m) / self.bev_res)
        self.max_path_steps = int(max_path_steps)

        if not 0.0 <= self.safe_distance_m < self.vortex_distance_m:
            raise ValueError("SAPF requires 0 <= safe_distance < vortex_distance")
        if self.influence_distance_m <= self.vortex_distance_m:
            raise ValueError("SAPF influence_distance must exceed vortex_distance")

        self.goal_robot_frame = None
        self.goal_rc = None
        self.hard_obstacle_map = np.zeros((self.H, self.W), dtype=bool)
        self.obstacle_clearance_map = np.full((self.H, self.W), np.inf, dtype=np.float32)
        self._nearest_indices = None
        self._path = []
        self._wz = 0.0
        self._vx = self.max_vx
        self._turn_sign = 1.0
        self._inside_obstacle_field = False

    def set_goal(self, goal_robot_frame):
        gx = max(0.05, float(goal_robot_frame[0]))
        gy = float(goal_robot_frame[1])
        self.goal_robot_frame = (gx, gy)
        self.goal_rc = (
            int(np.clip(round(gx / self.bev_res), 0, self.H - 1)),
            int(np.clip(round(gy / self.bev_res) + self.robot_c, 0, self.W - 1)),
        )

    @staticmethod
    def _unit(vector):
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-9 else np.zeros(2, dtype=np.float64)

    def _force_at(self, position_rc):
        attraction = self.attraction_gain * self._unit(
            np.asarray(self.goal_rc, dtype=np.float64) - position_rc)
        if self._nearest_indices is None:
            return attraction

        r = int(np.clip(round(position_rc[0]), 0, self.H - 1))
        c = int(np.clip(round(position_rc[1]), 0, self.W - 1))
        distance = float(self.obstacle_clearance_map[r, c])
        if not np.isfinite(distance) or distance >= self.influence_distance_m:
            return attraction

        nearest = self._nearest_indices[:, r, c].astype(np.float64)
        repulsive_direction = self._unit(position_rc - nearest)
        effective = max(distance - self.safe_distance_m, 0.02)
        influence = max(self.influence_distance_m - self.safe_distance_m, 0.02)
        magnitude = self.repulsion_gain * max(0.0, 1.0 / effective - 1.0 / influence) / (effective ** 2)
        magnitude = min(magnitude, 4.0)

        span = 2.0 * (self.vortex_distance_m - self.safe_distance_m)
        d_rel = np.clip((distance - self.safe_distance_m) / max(span, 1e-6), 0.0, 1.0)
        gamma_abs = np.pi * (d_rel if d_rel <= 0.5 else 1.0 - d_rel)

        gamma = self._turn_sign * gamma_abs
        cos_g, sin_g = np.cos(gamma), np.sin(gamma)
        rotated_repulsion = magnitude * np.array([
            cos_g * repulsive_direction[0] - sin_g * repulsive_direction[1],
            sin_g * repulsive_direction[0] + cos_g * repulsive_direction[1],
        ])
        return attraction + rotated_repulsion

    def update(self, occ_bev):
        occ = np.asarray(occ_bev, dtype=bool)
        if occ.shape != (self.H, self.W):
            raise ValueError("occ_bev shape %s does not match (%d, %d)" % (occ.shape, self.H, self.W))

        if np.any(occ):
            clearance_cells, indices = distance_transform_edt(~occ, return_indices=True)
            self.obstacle_clearance_map = (clearance_cells * self.bev_res).astype(np.float32)
            self._nearest_indices = indices
            radius_cells = max(1, int(round(self.safe_distance_m / self.bev_res)))
            self.hard_obstacle_map = morph.binary_dilation(occ, morph.disk(radius_cells))
        else:
            self.obstacle_clearance_map.fill(np.inf)
            self._nearest_indices = None
            self.hard_obstacle_map.fill(False)

        origin = np.array([self.robot_r, self.robot_c], dtype=np.float64)
        origin_clearance = float(self.obstacle_clearance_map[self.robot_r, self.robot_c])
        inside_field = origin_clearance < self.influence_distance_m
        # Select a vortex side only when entering an obstacle's influence
        # region.  Holding that side until the robot exits prevents the local
        # grid from choosing alternating left/right commands around symmetry.
        if inside_field and not self._inside_obstacle_field and self._nearest_indices is not None:
            nearest = self._nearest_indices[:, self.robot_r, self.robot_c].astype(np.float64)
            repulsion = self._unit(origin - nearest)
            attraction = self._unit(np.asarray(self.goal_rc, dtype=np.float64) - origin)
            cross = attraction[0] * repulsion[1] - attraction[1] * repulsion[0]
            if abs(cross) > 1e-4:
                self._turn_sign = 1.0 if cross >= 0.0 else -1.0
        self._inside_obstacle_field = inside_field
        force = self._force_at(origin)
        desired_yaw = float(np.arctan2(force[1], max(force[0], 1e-6)))
        self._wz = float(np.clip(self.yaw_k * desired_yaw, -self.max_wz, self.max_wz))
        heading_scale = max(0.15, np.cos(min(abs(desired_yaw), np.pi / 2.0)))
        clearance = origin_clearance
        clearance_scale = np.clip(
            (clearance - self.safe_distance_m) /
            max(self.vortex_distance_m - self.safe_distance_m, 1e-6), 0.15, 1.0)
        self._vx = float(self.max_vx * min(heading_scale, clearance_scale))
        self._path = self._integrate_path(origin)

    def _integrate_path(self, start):
        if self.goal_rc is None:
            return []
        position = start.copy()
        path = [(int(round(position[0])), int(round(position[1])))]
        for _ in range(self.max_path_steps):
            if np.linalg.norm(position - np.asarray(self.goal_rc)) <= 2.0:
                break
            force = self._force_at(position)
            direction = self._unit(force)
            if np.linalg.norm(direction) < 1e-6:
                break
            candidate = position + direction * self.integration_step_cells
            r, c = int(round(candidate[0])), int(round(candidate[1]))
            if not (0 <= r < self.H and 0 <= c < self.W):
                break
            if self.hard_obstacle_map[r, c]:
                # Preserve a drawable path while refusing to integrate through
                # the paper's hard safety region.
                break
            position = candidate
            if (r, c) != path[-1]:
                path.append((r, c))
        return path

    def compute_wz(self):
        return self._wz

    def compute_vx(self):
        return self._vx

    def extract_path(self):
        return self._path

    def is_reachable(self):
        return bool(self._path)

    def path_min_clearance(self, distance_m=1.5):
        if not self._path:
            return 0.0
        count = max(1, int(float(distance_m) / self.bev_res))
        values = [self.obstacle_clearance_map[r, c] for r, c in self._path[:count]]
        return float(np.min(values)) if values else 0.0

    def reset_hysteresis(self):
        self._turn_sign = 1.0
        self._inside_obstacle_field = False
        self._wz = 0.0


class PatternSearchDWAController(AStarDWAController):
    """DWA whose velocity optimization uses bounded pattern search."""

    def __init__(self, *args, max_evaluations=40, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_evaluations = int(max_evaluations)
        self.last_evaluation_count = 0

    def set_goal(self, goal_robot_frame):
        """Keep long-range goals on the local BEV boundary for A* search."""
        self.goal_robot_frame = tuple(map(float, goal_robot_frame))
        dx, dy = self.goal_robot_frame
        self.goal_rc = (
            int(np.clip(round(dx / self.bev_res), 0, self.H - 1)),
            int(np.clip(round(dy / self.bev_res) + self.W // 2, 0, self.W - 1)),
        )

    def _dwa_compute(self, grid):
        lower = np.array([
            max(0.0, self.current_vel[0] - self.max_acc_vx * self.dt),
            max(-self.max_wz, self.current_vel[1] - self.max_acc_wz * self.dt),
        ])
        upper = np.array([
            min(self.max_vx, self.current_vel[0] + self.max_acc_vx * self.dt),
            min(self.max_wz, self.current_vel[1] + self.max_acc_wz * self.dt),
        ])
        if len(self.global_path) > 5:
            idx = min(len(self.global_path) - 1, int(0.8 / self.bev_res))
            node = self.global_path[idx]
            ref_xy = np.array([node[0] * self.bev_res, (node[1] - self.W // 2) * self.bev_res])
        elif self.goal_robot_frame is not None:
            ref_xy = np.asarray(self.goal_robot_frame, dtype=np.float64)
        else:
            return 0.0, 0.0

        point = np.clip(self.current_vel.copy(), lower, upper)
        step = np.maximum((upper - lower) * 0.5, np.array([0.02, 0.04]))
        best_score = self._evaluate_traj(self._predict_trajectory(*point), grid, ref_xy)
        evaluations = 1

        while evaluations < self.max_evaluations and np.any(step > np.array([0.01, 0.02])):
            improved = False
            for axis in range(2):
                for sign in (-1.0, 1.0):
                    candidate = point.copy()
                    candidate[axis] = np.clip(candidate[axis] + sign * step[axis], lower[axis], upper[axis])
                    if np.allclose(candidate, point):
                        continue
                    score = self._evaluate_traj(self._predict_trajectory(*candidate), grid, ref_xy)
                    evaluations += 1
                    if score > best_score:
                        point, best_score, improved = candidate, score, True
                    if evaluations >= self.max_evaluations:
                        break
                if evaluations >= self.max_evaluations:
                    break
            if not improved:
                step *= 0.5

        self.last_evaluation_count = evaluations
        if best_score < -1e5:
            return 0.0, 0.5 if self.goal_robot_frame[1] > 0 else -0.5
        return float(point[0]), float(point[1])

    def compute_vx(self):
        return float(self.current_vel[0])

    def is_reachable(self):
        return bool(self.global_path)

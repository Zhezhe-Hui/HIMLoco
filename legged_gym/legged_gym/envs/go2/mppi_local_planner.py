import math

import numpy as np
from scipy.ndimage import distance_transform_edt


def _wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def choose_safer_turn_sign(occ_local, preferred_wz=0.0):
    """Choose +1 (left) or -1 (right) from near-field occupancy."""
    occ = np.asarray(occ_local, dtype=np.uint8)
    if occ.ndim != 2 or occ.size == 0:
        return 1 if float(preferred_wz) >= 0.0 else -1
    h, w = occ.shape
    center = w // 2
    near = occ[:max(1, h // 2)]
    right_count = int(np.count_nonzero(near[:, :center]))
    left_count = int(np.count_nonzero(near[:, center + 1:]))
    if left_count < right_count:
        return 1
    if right_count < left_count:
        return -1
    if abs(float(preferred_wz)) > 1e-3:
        return 1 if float(preferred_wz) > 0.0 else -1
    return 1


class WorldObstacleMemory:
    """Sparse world-aligned obstacle memory projected into the current BEV."""

    def __init__(self, bev_res, bev_x, bev_y, max_age_steps=75,
                 voxel_size_m=0.10, max_points=12000):
        self.bev_res = float(bev_res)
        self.bev_x = float(bev_x)
        self.bev_y = float(bev_y)
        self.H = int(self.bev_x / self.bev_res)
        self.W = int(self.bev_y / self.bev_res)
        self.max_age_steps = int(max_age_steps)
        self.voxel_size_m = float(voxel_size_m)
        self.max_points = int(max_points)
        self._voxels = {}

    def reset(self):
        self._voxels.clear()

    def _key(self, x_w, y_w):
        return (int(np.round(x_w / self.voxel_size_m)),
                int(np.round(y_w / self.voxel_size_m)))

    def update(self, occ_local, pose_xy_yaw, step, observed_local=None):
        occ = np.asarray(occ_local, dtype=np.uint8)
        if occ.shape != (self.H, self.W):
            raise ValueError("occ_local shape does not match memory BEV")

        # Clear stale voxels only where the current depth frame explicitly observed
        # free space. Cells outside the camera frustum remain unknown and keep memory.
        if observed_local is not None and self._voxels:
            observed = np.asarray(observed_local, dtype=bool)
            if observed.shape != (self.H, self.W):
                raise ValueError("observed_local shape does not match memory BEV")
            free_observed = observed & (occ == 0)
            keys = np.asarray(list(self._voxels.keys()), dtype=np.float64)
            x_w = keys[:, 0] * self.voxel_size_m
            y_w = keys[:, 1] * self.voxel_size_m
            x, y, yaw = map(float, pose_xy_yaw)
            cy, sy = math.cos(yaw), math.sin(yaw)
            dx, dy = x_w - x, y_w - y
            rows = np.floor((cy * dx + sy * dy) / self.bev_res).astype(np.int32)
            cols = np.floor(
                (-sy * dx + cy * dy) / self.bev_res + self.W * 0.5
            ).astype(np.int32)
            inside = ((rows >= 0) & (rows < self.H)
                      & (cols >= 0) & (cols < self.W))
            clear = np.zeros(keys.shape[0], dtype=bool)
            clear[inside] = free_observed[rows[inside], cols[inside]]
            for key in map(tuple, keys[clear].astype(np.int64)):
                self._voxels.pop(key, None)

        rows, cols = np.nonzero(occ)
        if rows.size:
            x_r = (rows.astype(np.float64) + 0.5) * self.bev_res
            y_r = ((cols.astype(np.float64) + 0.5) - self.W * 0.5) * self.bev_res
            x, y, yaw = map(float, pose_xy_yaw)
            cy, sy = math.cos(yaw), math.sin(yaw)
            x_w = x + cy * x_r - sy * y_r
            y_w = y + sy * x_r + cy * y_r
            for px, py in zip(x_w, y_w):
                self._voxels[self._key(px, py)] = int(step)

        cutoff = int(step) - self.max_age_steps
        if cutoff > 0:
            self._voxels = {
                key: seen for key, seen in self._voxels.items() if seen >= cutoff
            }
        if len(self._voxels) > self.max_points:
            newest = sorted(self._voxels.items(), key=lambda item: item[1], reverse=True)
            self._voxels = dict(newest[:self.max_points])

    def project(self, pose_xy_yaw):
        occ = np.zeros((self.H, self.W), dtype=np.uint8)
        if not self._voxels:
            return occ

        keys = np.asarray(list(self._voxels.keys()), dtype=np.float64)
        x_w = keys[:, 0] * self.voxel_size_m
        y_w = keys[:, 1] * self.voxel_size_m
        x, y, yaw = map(float, pose_xy_yaw)
        cy, sy = math.cos(yaw), math.sin(yaw)
        dx, dy = x_w - x, y_w - y
        x_r = cy * dx + sy * dy
        y_r = -sy * dx + cy * dy
        rows = np.floor(x_r / self.bev_res).astype(np.int32)
        cols = np.floor(y_r / self.bev_res + self.W * 0.5).astype(np.int32)
        valid = ((rows >= 0) & (rows < self.H)
                 & (cols >= 0) & (cols < self.W))
        occ[rows[valid], cols[valid]] = 1
        return occ

    @property
    def point_count(self):
        return len(self._voxels)


class FootprintMPPIController:
    """Sampling-based MPPI tracker for a path in a robot-centric BEV.

    State is a unicycle ``(x, y, yaw)`` and controls are ``(vx, wz)``. Candidate
    trajectories are evaluated with the full rectangular body footprint, path
    tracking, obstacle clearance and control smoothness costs.
    """

    def __init__(self, bev_res=0.05, bev_x=6.0, bev_y=6.0,
                 horizon=20, num_samples=384, rollout_dt=0.10,
                 max_vx=1.0, max_wz=1.0, min_vx=0.10,
                 footprint_length_m=0.72, footprint_width_m=0.42,
                 safety_margin_m=0.05, temperature=8.0,
                 noise_vx=0.22, noise_wz=0.55, seed=7):
        self.bev_res = float(bev_res)
        self.bev_x = float(bev_x)
        self.bev_y = float(bev_y)
        self.H = int(self.bev_x / self.bev_res)
        self.W = int(self.bev_y / self.bev_res)
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.rollout_dt = float(rollout_dt)
        self.max_vx = float(max_vx)
        self.max_wz = float(max_wz)
        self.min_vx = float(min_vx)
        self.temperature = float(temperature)
        self.noise_vx = float(noise_vx)
        self.noise_wz = float(noise_wz)
        self.rng = np.random.RandomState(int(seed))

        length = float(footprint_length_m) + 2.0 * float(safety_margin_m)
        width = float(footprint_width_m) + 2.0 * float(safety_margin_m)
        self.footprint_length_m = length
        self.footprint_width_m = width
        self._footprint_points = self._make_footprint_points(length, width)
        self._u = np.zeros((self.horizon, 2), dtype=np.float64)
        self._u[:, 0] = min(0.95, self.max_vx)
        self._last_cmd = np.array([min(0.95, self.max_vx), 0.0], dtype=np.float64)
        self.last_best_cost = float("inf")
        self.last_collision_fraction = 0.0
        self.last_plan = np.zeros((0, 3), dtype=np.float64)

    def _make_footprint_points(self, length, width):
        # Dense perimeter plus center/front-center catches both side-swipe and head-on contact.
        nx = max(3, int(np.ceil(length / self.bev_res)) + 1)
        ny = max(3, int(np.ceil(width / self.bev_res)) + 1)
        xs = np.linspace(-length * 0.5, length * 0.5, nx)
        ys = np.linspace(-width * 0.5, width * 0.5, ny)
        points = []
        points.extend((x, -width * 0.5) for x in xs)
        points.extend((x, width * 0.5) for x in xs)
        points.extend((-length * 0.5, y) for y in ys[1:-1])
        points.extend((length * 0.5, y) for y in ys[1:-1])
        points.extend([(0.0, 0.0), (length * 0.25, 0.0), (length * 0.5, 0.0)])
        return np.asarray(points, dtype=np.float64)

    def reset(self):
        self._u.fill(0.0)
        self._u[:, 0] = min(0.95, self.max_vx)
        self._last_cmd[:] = (min(0.95, self.max_vx), 0.0)
        self.last_best_cost = float("inf")
        self.last_collision_fraction = 0.0
        self.last_plan = np.zeros((0, 3), dtype=np.float64)

    def _path_targets(self, path_xy):
        path = np.asarray(path_xy, dtype=np.float64)
        if path.ndim != 2 or path.shape[0] == 0:
            return np.zeros((self.horizon, 2), dtype=np.float64)
        if path.shape[0] == 1:
            return np.repeat(path[:, :2], self.horizon, axis=0)
        path = path[:, :2]
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(seg)))
        travel = 0.20 + np.arange(1, self.horizon + 1) * self.rollout_dt * 0.90
        targets = np.empty((self.horizon, 2), dtype=np.float64)
        targets[:, 0] = np.interp(travel, cumulative, path[:, 0])
        targets[:, 1] = np.interp(travel, cumulative, path[:, 1])
        return targets

    def _sample_controls(self, desired_heading):
        noise = self.rng.normal(size=(self.num_samples, self.horizon, 2))
        noise[:, :, 0] *= self.noise_vx
        noise[:, :, 1] *= self.noise_wz
        # Correlated noise produces executable control sequences rather than frame-wise chatter.
        for t in range(1, self.horizon):
            noise[:, t, :] = 0.65 * noise[:, t - 1, :] + 0.35 * noise[:, t, :]
        controls = self._u[None, :, :] + noise
        controls[0] = self._u
        controls[:, :, 0] = np.clip(controls[:, :, 0], self.min_vx, self.max_vx)
        controls[:, :, 1] = np.clip(controls[:, :, 1], -self.max_wz, self.max_wz)

        # Seed one deterministic path-following candidate so symmetric scenes do not depend on noise.
        controls[1, :, 0] = min(0.90, self.max_vx)
        controls[1, :, 1] = np.clip(2.0 * desired_heading, -self.max_wz, self.max_wz)
        return controls

    def _rollout(self, controls):
        states = np.zeros((controls.shape[0], self.horizon, 3), dtype=np.float64)
        x = np.zeros(controls.shape[0], dtype=np.float64)
        y = np.zeros_like(x)
        yaw = np.zeros_like(x)
        for t in range(self.horizon):
            v, w = controls[:, t, 0], controls[:, t, 1]
            x += v * np.cos(yaw) * self.rollout_dt
            y += v * np.sin(yaw) * self.rollout_dt
            yaw = _wrap_angle(yaw + w * self.rollout_dt)
            states[:, t, 0] = x
            states[:, t, 1] = y
            states[:, t, 2] = yaw
        return states

    def footprint_collision_mask(self, states, occ_bev):
        states = np.asarray(states, dtype=np.float64)
        original_shape = states.shape[:-1]
        flat = states.reshape(-1, 3)
        yaw = flat[:, 2, None]
        px = self._footprint_points[None, :, 0]
        py = self._footprint_points[None, :, 1]
        wx = flat[:, 0, None] + np.cos(yaw) * px - np.sin(yaw) * py
        wy = flat[:, 1, None] + np.sin(yaw) * px + np.cos(yaw) * py
        rows = np.floor(wx / self.bev_res).astype(np.int32)
        cols = np.floor(wy / self.bev_res + self.W * 0.5).astype(np.int32)
        visible = (rows >= 0) & (rows < self.H) & (cols >= 0) & (cols < self.W)
        lateral_out = (wy <= -self.bev_y * 0.5) | (wy >= self.bev_y * 0.5)
        hit = lateral_out.copy()
        if np.any(visible):
            hit[visible] |= np.asarray(occ_bev, dtype=bool)[rows[visible], cols[visible]]
        return np.any(hit, axis=1).reshape(original_shape)

    def _trajectory_cost(self, states, controls, targets, occ_bev):
        error = states[:, :, :2] - targets[None, :, :]
        track_cost = 5.0 * np.sum(error * error, axis=(1, 2))
        terminal_error = states[:, -1, :2] - targets[-1]
        terminal_cost = 15.0 * np.sum(terminal_error * terminal_error, axis=1)

        target_heading = np.arctan2(targets[-1, 1] - states[:, -1, 1],
                                    targets[-1, 0] - states[:, -1, 0])
        heading_cost = 3.0 * _wrap_angle(target_heading - states[:, -1, 2]) ** 2

        collision = self.footprint_collision_mask(states, occ_bev)
        collision_cost = 2500.0 * np.sum(collision, axis=1)

        clearance = distance_transform_edt(np.asarray(occ_bev) == 0) * self.bev_res
        rows = np.clip(np.floor(states[:, :, 0] / self.bev_res).astype(np.int32), 0, self.H - 1)
        cols = np.clip(
            np.floor(states[:, :, 1] / self.bev_res + self.W * 0.5).astype(np.int32),
            0, self.W - 1)
        center_clearance = clearance[rows, cols]
        proximity = np.clip(0.65 - center_clearance, 0.0, 0.65) / 0.65
        proximity_cost = 18.0 * np.sum(proximity * proximity, axis=1)

        du = np.diff(controls, axis=1)
        smooth_cost = (1.5 * np.sum(du[:, :, 0] ** 2, axis=1)
                       + 4.0 * np.sum(du[:, :, 1] ** 2, axis=1))
        effort_cost = (0.08 * np.sum((self.max_vx - controls[:, :, 0]) ** 2, axis=1)
                       + 0.35 * np.sum(controls[:, :, 1] ** 2, axis=1))
        return (track_cost + terminal_cost + heading_cost + collision_cost
                + proximity_cost + smooth_cost + effort_cost), collision

    def command(self, occ_bev, path_xy, goal_xy=None):
        occ = np.asarray(occ_bev, dtype=np.uint8)
        if occ.shape != (self.H, self.W):
            raise ValueError("occ_bev shape does not match MPPI BEV")
        targets = self._path_targets(path_xy)
        if not np.any(targets) and goal_xy is not None:
            targets[:] = np.asarray(goal_xy, dtype=np.float64)[:2]
        desired_heading = math.atan2(targets[min(3, self.horizon - 1), 1],
                                     max(0.05, targets[min(3, self.horizon - 1), 0]))
        controls = self._sample_controls(desired_heading)
        states = self._rollout(controls)
        costs, collision = self._trajectory_cost(states, controls, targets, occ)
        finite = np.isfinite(costs)
        if not np.any(finite):
            return 0.0, 0.0
        best = float(np.min(costs[finite]))
        logits = -(costs - best) / max(self.temperature, 1e-6)
        logits = np.clip(logits, -60.0, 0.0)
        weights = np.exp(logits)
        weights[~finite] = 0.0
        weights /= max(float(np.sum(weights)), 1e-12)
        self._u = np.sum(weights[:, None, None] * controls, axis=0)

        best_idx = int(np.argmin(costs))
        self.last_best_cost = float(costs[best_idx])
        self.last_collision_fraction = float(np.mean(collision[best_idx]))
        self.last_plan = states[best_idx].copy()

        cmd = self._u[0].copy()
        cmd[1] = 0.65 * self._last_cmd[1] + 0.35 * cmd[1]
        if abs(cmd[1]) < 0.035:
            cmd[1] = 0.0
        # Final command slew limits run at the 50 Hz control rate.
        cmd[0] = np.clip(cmd[0], self._last_cmd[0] - 0.04, self._last_cmd[0] + 0.04)
        cmd[1] = np.clip(cmd[1], self._last_cmd[1] - 0.08, self._last_cmd[1] + 0.08)
        cmd[0] = np.clip(cmd[0], 0.0, self.max_vx)
        cmd[1] = np.clip(cmd[1], -self.max_wz, self.max_wz)
        self._last_cmd = cmd

        self._u[:-1] = self._u[1:]
        self._u[-1] = self._u[-2]
        self._u[0] = cmd
        return float(cmd[0]), float(cmd[1])

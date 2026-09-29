"""Adapter from the official iPlanner network to HIMLoco's local planner API."""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as functional


class IPlannerController:
    """Run the official RSS 2023 iPlanner checkpoint on Isaac Gym depth.

    iPlanner predicts five 3-D keypoints and a collision-fear probability from
    a metric depth image plus a robot-frame goal.  This adapter preserves the
    official network and cubic interpolation, then applies a proportional path
    follower to expose the same ``vx/wz`` interface as the other planners.
    """

    def __init__(
        self,
        model_path,
        runtime_dir,
        bev_res=0.05,
        bev_x=6.0,
        bev_y=6.0,
        max_goal_distance_m=6.0,
        max_vx=1.0,
        max_wz=1.0,
        yaw_k=2.0,
        lookahead_m=0.8,
        fear_threshold=0.5,
        fear_buffer_size=10,
        fear_speed_m_s=0.15,
        device="cuda",
    ):
        self.model_path = os.path.abspath(os.path.expanduser(model_path))
        self.runtime_dir = os.path.abspath(os.path.expanduser(runtime_dir))
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError("iPlanner checkpoint not found: %s" % self.model_path)
        if not os.path.isdir(self.runtime_dir):
            raise FileNotFoundError("iPlanner runtime not found: %s" % self.runtime_dir)

        if self.runtime_dir not in sys.path:
            sys.path.insert(0, self.runtime_dir)
        # torch.load needs these module names available to unpickle the
        # official full-model checkpoint.
        __import__("percept_net")
        __import__("planner_net")
        traj_opt = __import__("traj_opt")

        requested = str(device).lower()
        self.device = torch.device(
            "cuda" if requested.startswith("cuda") and torch.cuda.is_available() else "cpu")
        loaded = torch.load(self.model_path, map_location=self.device)
        self.net = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
        self.net = self.net.to(self.device).eval()
        self.trajectory_generator = traj_opt.TrajOpt()

        self.bev_res = float(bev_res)
        self.H = int(float(bev_x) / self.bev_res)
        self.W = int(float(bev_y) / self.bev_res)
        self.robot_r = 0
        self.robot_c = self.W // 2
        self.max_goal_distance_m = float(max_goal_distance_m)
        self.max_vx = float(max_vx)
        self.max_wz = float(max_wz)
        self.yaw_k = float(yaw_k)
        self.lookahead_m = float(lookahead_m)
        self.fear_threshold = float(fear_threshold)
        self.fear_buffer_size = int(fear_buffer_size)
        self.fear_speed_m_s = float(fear_speed_m_s)

        self.goal_robot_frame = None
        self._depth = None
        self._path = []
        self._wz = 0.0
        self._vx = 0.0
        self._fear = 0.0
        self._fear_buffer = 0
        self._fear_active = False
        self._keypoints = None

    def set_goal(self, goal_robot_frame):
        goal = np.asarray(goal_robot_frame, dtype=np.float32)
        norm = float(np.linalg.norm(goal))
        if norm > self.max_goal_distance_m:
            goal *= self.max_goal_distance_m / norm
        self.goal_robot_frame = (float(goal[0]), float(goal[1]))

    def set_depth(self, depth_m):
        if depth_m is None:
            return
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim != 2:
            depth = np.squeeze(depth)
        if depth.ndim != 2:
            raise ValueError("iPlanner depth must be HxW, got %s" % (depth.shape,))
        self._depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    def update(self, _occ_bev=None):
        if self.goal_robot_frame is None or self._depth is None:
            self._path = []
            self._vx = 0.0
            self._wz = 0.0
            return

        depth = torch.from_numpy(self._depth).to(self.device)[None, None]
        depth = functional.interpolate(
            depth, size=(360, 640), mode="bilinear", align_corners=False)
        depth = depth.expand(-1, 3, -1, -1)
        goal = torch.tensor(
            [[self.goal_robot_frame[0], self.goal_robot_frame[1], 0.0]],
            dtype=torch.float32, device=self.device)

        with torch.no_grad():
            keypoints, fear = self.net(depth, goal)
            trajectory = self.trajectory_generator.TrajGeneratorFromPFreeRot(
                keypoints, step=0.1)

        self._fear = float(fear.reshape(-1)[0].item())
        trajectory_xy = trajectory[0, :, :2].detach().cpu().numpy()
        self._keypoints = keypoints[0].detach().cpu().numpy()
        self._path = self._trajectory_to_grid(trajectory_xy)

        target = self._lookahead_point(trajectory_xy)
        desired_yaw = float(np.arctan2(target[1], max(target[0], 1e-4)))
        self._wz = float(np.clip(self.yaw_k * desired_yaw, -self.max_wz, self.max_wz))

        forward_tracking = target[0] > 0.0 and abs(desired_yaw) < np.deg2rad(45.0)
        if self._fear > self.fear_threshold and forward_tracking:
            self._fear_buffer += 1
        elif self._fear_active:
            self._fear_buffer -= 1
        self._fear_buffer = int(np.clip(self._fear_buffer, 0, self.fear_buffer_size + 1))
        self._fear_active = self._fear_buffer > self.fear_buffer_size

        heading_scale = max(0.2, np.cos(min(abs(desired_yaw), np.pi / 2.0)))
        self._vx = self.max_vx * heading_scale
        if self._fear_active:
            self._vx = min(self._vx, self.fear_speed_m_s)

    def _lookahead_point(self, trajectory_xy):
        if trajectory_xy.size == 0:
            return np.asarray(self.goal_robot_frame, dtype=np.float32)
        distances = np.linalg.norm(trajectory_xy, axis=1)
        candidates = np.flatnonzero(distances >= self.lookahead_m)
        index = int(candidates[0]) if candidates.size else len(trajectory_xy) - 1
        return trajectory_xy[index]

    def _trajectory_to_grid(self, trajectory_xy):
        path = [(self.robot_r, self.robot_c)]
        for x, y in trajectory_xy:
            r = int(round(float(x) / self.bev_res))
            c = int(round(float(y) / self.bev_res)) + self.robot_c
            if 0 <= r < self.H and 0 <= c < self.W and (r, c) != path[-1]:
                path.append((r, c))
        return path

    def compute_wz(self):
        return self._wz

    def compute_vx(self):
        return self._vx

    def extract_path(self):
        return self._path

    def is_reachable(self):
        return len(self._path) > 1

    @property
    def fear_probability(self):
        return self._fear

    def reset_hysteresis(self):
        self._fear_buffer = 0
        self._fear_active = False
        self._wz = 0.0
        self._vx = 0.0


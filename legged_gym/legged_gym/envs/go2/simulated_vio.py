import math
from collections import deque

import numpy as np


def wrap_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class SimulatedVIO:
    """Deterministic pose sensor model for ideal and degraded VIO experiments."""

    def __init__(self, dt, position_sigma_m=0.0, yaw_sigma_rad=0.0,
                 position_drift_m_sqrt_s=0.0, yaw_drift_rad_sqrt_s=0.0,
                 latency_steps=0, dropout_probability=0.0, seed=0):
        self.dt = float(dt)
        self.position_sigma_m = float(position_sigma_m)
        self.yaw_sigma_rad = float(yaw_sigma_rad)
        self.position_drift_m_sqrt_s = float(position_drift_m_sqrt_s)
        self.yaw_drift_rad_sqrt_s = float(yaw_drift_rad_sqrt_s)
        self.latency_steps = max(0, int(latency_steps))
        self.dropout_probability = float(dropout_probability)
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("dropout_probability must be in [0, 1)")
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.position_drift = np.zeros(2, dtype=np.float64)
        self.yaw_drift = 0.0
        self._queue = deque()
        self._last_output = None

    @property
    def is_ideal(self):
        return (self.position_sigma_m == 0.0
                and self.yaw_sigma_rad == 0.0
                and self.position_drift_m_sqrt_s == 0.0
                and self.yaw_drift_rad_sqrt_s == 0.0
                and self.latency_steps == 0
                and self.dropout_probability == 0.0)

    def reset(self, initial_pose, seed=None):
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.position_drift.fill(0.0)
        self.yaw_drift = 0.0
        pose = np.asarray(initial_pose, dtype=np.float64).reshape(3).copy()
        pose[2] = wrap_angle(pose[2])
        self._queue = deque([pose.copy() for _ in range(self.latency_steps)])
        self._last_output = pose.copy()
        return pose.copy()

    def update(self, true_pose):
        true_pose = np.asarray(true_pose, dtype=np.float64).reshape(3)
        sqrt_dt = math.sqrt(max(self.dt, 0.0))
        self.position_drift += self.rng.normal(
            scale=self.position_drift_m_sqrt_s * sqrt_dt, size=2)
        self.yaw_drift += float(self.rng.normal(
            scale=self.yaw_drift_rad_sqrt_s * sqrt_dt))

        measured = true_pose.copy()
        measured[:2] += self.position_drift
        measured[:2] += self.rng.normal(scale=self.position_sigma_m, size=2)
        measured[2] = wrap_angle(
            measured[2] + self.yaw_drift
            + float(self.rng.normal(scale=self.yaw_sigma_rad)))

        if (self._last_output is not None
                and self.rng.rand() < self.dropout_probability):
            return self._last_output.copy()

        self._queue.append(measured)
        output = self._queue.popleft()
        self._last_output = output.copy()
        return output.copy()

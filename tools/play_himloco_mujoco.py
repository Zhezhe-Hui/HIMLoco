"""Run an original HIMLoco checkpoint in a local MuJoCo viewer.

This simulation-only entry point has no DDS or robot connection. Parameters
mirror the current Go2 training source, not a recovered historical run config.
"""

import argparse
import contextlib
import io
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))
from rsl_rl.modules import HIMActorCritic

# Verified using Isaac Gym get_asset_dof_names on the training Go2 URDF.
JOINT_NAMES = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
               for joint in ("hip", "thigh", "calf")]
DEFAULT_Q = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                      0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
TORQUE_LIMITS = np.tile([23.7, 23.7, 35.55], 4)
PHYSICS_DT = 0.005
DECIMATION = 4


class Simulation:
    def __init__(self, args):
        self.args = args
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        weights = checkpoint["model_state_dict"]
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.policy = HIMActorCritic(270, weights["critic.0.weight"].shape[1], 45, 12)
        self.policy.load_state_dict(weights, strict=True)
        self.policy.eval()
        self.model = mujoco.MjModel.from_xml_path(str(args.scene))
        self.model.opt.timestep = PHYSICS_DT
        if args.friction is not None:
            if args.friction <= 0:
                raise ValueError("--friction must be positive")
            # Feet have higher contact priority than the floor in this scene;
            # changing just the floor would leave foot contact friction at 0.4.
            for name in ("floor", "FL", "FR", "RL", "RR"):
                self.model.geom_friction[self.model.geom(name).id, 0] = args.friction
        self.data = mujoco.MjData(self.model)
        self.joints = np.array([self.model.joint(name).id for name in JOINT_NAMES])
        self.q_indices = self.model.jnt_qposadr[self.joints]
        self.v_indices = self.model.jnt_dofadr[self.joints]
        self.actuators = []
        for joint_id in self.joints:
            indices = np.flatnonzero(
                (self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                & (self.model.actuator_trnid[:, 0] == joint_id))
            if len(indices) != 1:
                raise ValueError(f"Expected one motor for joint {joint_id}")
            self.actuators.append(int(indices[0]))
        self.base_id = self.model.body("base_link").id
        self.commands = np.array([args.vx, args.vy, args.wz], dtype=float)
        self.reset_requested = False
        self.min_height = float("inf")
        self.reset()
        print(f"Checkpoint: {args.checkpoint} (iteration {checkpoint['iter']})", flush=True)
        print("Policy 50 Hz; PD 200 Hz; joint names mapped to MuJoCo motors.", flush=True)
        print("W/S: forward/back; A/D: lateral; Q/E: turn; X: zero command; R: reset.", flush=True)
        print("Commands are velocities in m/s, m/s, rad/s; all control is local simulation.", flush=True)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0, 0, 0.37]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[self.q_indices] = DEFAULT_Q
        self.history = np.zeros(270, dtype=np.float32)
        self.action = np.zeros(12)
        self.target = DEFAULT_Q.copy()
        self.steps = 0
        self.min_height = float("inf")
        self.reset_requested = False
        mujoco.mj_forward(self.model, self.data)

    def key(self, keycode):
        changes = {ord("W"): (0, 0.1), ord("S"): (0, -0.1),
                   ord("A"): (1, 0.1), ord("D"): (1, -0.1),
                   ord("Q"): (2, 0.1), ord("E"): (2, -0.1)}
        if keycode in changes:
            index, delta = changes[keycode]
            self.commands[index] += delta
            self.commands[:] = np.clip(self.commands, [-0.5, -0.5, -1], [1, 0.5, 1])
        elif keycode == ord("X"):
            self.commands[:] = 0
        elif keycode == ord("R"):
            self.commands[:] = 0
            self.reset_requested = True
        else:
            return
        print(f"Command [vx,vy,wz]: {self.commands.round(2).tolist()}", flush=True)

    def step(self):
        if self.reset_requested:
            self.reset()
        q = self.data.qpos[self.q_indices]
        dq = self.data.qvel[self.v_indices]
        if self.steps % DECIMATION == 0:
            rotation = self.data.xmat[self.base_id].reshape(3, 3)
            gravity = rotation.T @ np.array([0., 0., -1.])
            body_velocity = np.zeros(6)
            # XBODY uses the regular body frame, matching training. BODY would
            # rotate into the principal-inertia frame (different on this Go2).
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                    self.base_id, body_velocity, 1)
            gyro = body_velocity[:3]
            current = np.concatenate((self.commands.copy() * [2., 2., 0.25],
                                      gyro * 0.25, gravity, q - DEFAULT_Q,
                                      dq * 0.05, self.action))
            # Current frame first, then the five preceding policy frames.
            self.history[45:] = self.history[:-45].copy()
            self.history[:45] = current
            np.clip(self.history, -100, 100, out=self.history)
            with torch.inference_mode():
                output = self.policy.act_inference(torch.from_numpy(self.history.copy())[None])
            self.action = output[0].numpy().clip(-100, 100).copy()
            if not np.isfinite(self.action).all():
                raise RuntimeError("Nonfinite action")
            self.target = DEFAULT_Q + self.action * 0.25
            # Preserve the additional hip clamp in current _compute_torques.
            self.target[[0, 3, 6, 9]] = self.target[[0, 3, 6, 9]].clip(-0.1, 0.1)
        torques = (40. * (self.target - q) - dq).clip(-TORQUE_LIMITS, TORQUE_LIMITS)
        self.data.ctrl[self.actuators] = torques
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        height = float(self.data.qpos[2])
        self.min_height = min(self.min_height, height)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("Nonfinite simulation state")

    def status(self):
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        return (f"t={self.data.time:.1f}s height={self.data.qpos[2]:.3f}m "
                f"min_height={self.min_height:.3f}m upright={rotation[2, 2]:.3f} "
                f"position={self.data.qpos[:2].round(3).tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "legged_gym/logs/go2_new_reward/Dec10_20-00-07_/model_1200.pt")
    parser.add_argument("--scene", type=Path, default=Path("/home/lenovo/unitree_rl_mjlab/src/assets/robots/unitree_go2/xmls/scene_go2.xml"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0, help="Simulation seconds; 0 means unlimited")
    parser.add_argument("--vx", type=float, default=0.)
    parser.add_argument("--vy", type=float, default=0.)
    parser.add_argument("--wz", type=float, default=0.)
    parser.add_argument("--friction", type=float, default=None,
                        help="Override sliding friction on feet and floor; default keeps XML")
    args = parser.parse_args()
    sim = Simulation(args)
    if args.headless:
        if args.duration <= 0:
            parser.error("Headless mode requires a positive --duration")
        for _ in range(round(args.duration / PHYSICS_DT)):
            sim.step()
        print(sim.status(), flush=True)
        if sim.data.xmat[sim.base_id].reshape(3, 3)[2, 2] < 0.5 or sim.data.qpos[2] < 0.15:
            raise RuntimeError("Robot ended the smoke run fallen")
        return
    import mujoco.viewer
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=sim.key) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = sim.base_id
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
        wall_start = time.monotonic()
        frame = 0
        last_status = 0.
        while viewer.is_running():
            if args.duration > 0 and sim.data.time >= args.duration:
                break
            with viewer.lock():
                sim.step()
            frame += 1
            if frame % 3 == 0:
                viewer.sync()
            if time.monotonic() - last_status > 5:
                print(sim.status(), flush=True)
                last_status = time.monotonic()
            deadline = wall_start + frame * PHYSICS_DT
            time.sleep(max(0., deadline - time.monotonic()))
            # Avoid a long catch-up burst after dragging the window or a stall.
            if time.monotonic() - deadline > 0.5:
                wall_start = time.monotonic() - frame * PHYSICS_DT


if __name__ == "__main__":
    main()

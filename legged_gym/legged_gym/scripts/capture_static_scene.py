"""Capture one static Isaac Gym scene without running navigation or showing a live viewer."""

import os
from pathlib import Path

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch.
from isaacgym import gymapi
import numpy as np
from PIL import Image
import torch

from legged_gym.envs import *  # noqa: F401,F403 - registers tasks.
from legged_gym.utils import get_args, task_registry


def _env_vec3(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return tuple(float(value) for value in default)
    values = [part.strip() for part in raw.split(",")]
    if len(values) != 3:
        raise ValueError("%s must contain x,y,z" % name)
    return tuple(float(value) for value in values)


def _assert_middle_strip(points, label):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return
    if not np.all((points[:, 0] >= 12.0) & (points[:, 0] <= 24.0)
                  & (points[:, 1] >= 0.0) & (points[:, 1] <= 12.0)):
        raise RuntimeError("%s are not fully inside the middle terrain strip" % label)


def _assert_endpoint_clear(points, label):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return
    endpoints = np.asarray([(12.0, 6.0), (24.0, 6.0)], dtype=np.float32)
    required = float(os.environ.get("HIMLOCO_ROUTE_ENDPOINT_CLEARANCE", "1.5"))
    distances = np.linalg.norm(points[:, None, :] - endpoints[None, :, :], axis=2)
    if np.any(distances < required - 1e-5):
        raise RuntimeError("%s violate endpoint clearance %.2f m" % (label, required))


def main(args):
    if args.headless:
        raise ValueError("Camera rendering needs a graphics device; do not use --headless")

    output = Path(os.environ.get(
        "HIMLOCO_SCENE_CAPTURE_PATH",
        "../assets/runs/navigation_benchmark/static_scene_easy.png")).resolve()
    camera_position = _env_vec3("HIMLOCO_SCENE_VIEW_POS", (18.0, 5.0, 18.0))
    camera_target = _env_vec3("HIMLOCO_SCENE_VIEW_LOOKAT", (18.0, 6.0, 0.0))

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.spawn.init_velocity_range = 0.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.camera.enable = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    try:
        env.reset_idx(torch.arange(env.num_envs, device=env.device))

        stones = np.asarray(
            getattr(env, "_creation_random_stone_positions", []),
            dtype=np.float32).reshape(-1, 2)
        trees = np.asarray(
            getattr(env, "_creation_random_tree_positions", []),
            dtype=np.float32).reshape(-1, 2)
        all_stones = np.asarray(env.saved_stone_start_poses[0], dtype=np.float32)[:, :2]
        _assert_middle_strip(all_stones, "stones")
        _assert_middle_strip(trees, "trees")
        _assert_endpoint_clear(all_stones, "stones")
        _assert_endpoint_clear(trees, "trees")

        # One zero-action physics synchronization makes newly created static
        # actors visible to the graphics pipeline. No navigation loop follows.
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)

        if env.cam_handle is None or env.camera_color_tensor is None:
            raise RuntimeError("Isaac Gym camera sensor was not created")
        env.gym.set_camera_location(
            env.cam_handle,
            env.cam_env_handle,
            gymapi.Vec3(*camera_position),
            gymapi.Vec3(*camera_target))
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        env.gym.start_access_image_tensors(env.sim)
        try:
            rgb = env.camera_color_tensor.detach().cpu().numpy()[..., :3].copy()
        finally:
            env.gym.end_access_image_tensors(env.sim)

        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(str(output))
        print("[capture_static_scene] saved: %s" % output)
        print("[capture_static_scene] big stones=%d, small stones=%d, trees=%d"
              % (len(stones), len(all_stones) - len(stones), len(trees)))
        print("[capture_static_scene] camera position=%s lookat=%s"
              % (camera_position, camera_target))
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
            env.viewer = None
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main(get_args())

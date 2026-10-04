"""Open a static Isaac Gym scene for interactive viewer-camera adjustment."""

import os
import time

# 静态论文场景的默认终点与端点净空要和截图脚本一致。
if (os.environ.get("HIMLOCO_ROUTE_START_X", "").strip()
        or os.environ.get("HIMLOCO_ROUTE_START_Y", "").strip()):
    os.environ.setdefault("HIMLOCO_ROUTE_GOAL_X", "24")
    os.environ.setdefault("HIMLOCO_ROUTE_ENDPOINT_CLEARANCE", "2.0")

import isaacgym  # noqa: F401 - Isaac Gym must be imported before torch.
from isaacgym import gymapi
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


def main(args):
    if args.headless:
        raise ValueError("scene_view_test.py requires a viewer; do not use --headless")

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.spawn.init_velocity_range = 0.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.camera.enable = False
    env_cfg.viewer.horizontal_fov = float(
        os.environ.get("HIMLOCO_SCENE_VIEW_FOV", "90"))
    env_cfg.viewer.width = int(os.environ.get("HIMLOCO_SCENE_VIEW_WIDTH", "640"))
    env_cfg.viewer.height = int(os.environ.get("HIMLOCO_SCENE_VIEW_HEIGHT", "360"))

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))

    # 与静态截图脚本保持相同：只同步一次PhysX/图形状态，
    # 让新创建的石头、树和机器人进入viewer，之后不再推进仿真。
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)

    topdown_z = os.environ.get("HIMLOCO_SCENE_VIEW_Z", "").strip()
    if topdown_z:
        # Obstacle strip center is (18, 6). A 0.2 m lateral offset avoids the
        # undefined up-vector of an exactly vertical look-at while remaining
        # visually top-down. World x consequently runs left-to-right on screen.
        camera_position = (18.0, 5.8, float(topdown_z))
        camera_target = (18.0, 6.0, 0.0)
    else:
        camera_position = _env_vec3("HIMLOCO_SCENE_VIEW_POS", (18.0, 16.0, 18.0))
        camera_target = _env_vec3("HIMLOCO_SCENE_VIEW_LOOKAT", (18.0, 6.0, 0.0))
    env.set_camera(camera_position, camera_target)

    print("[scene_view_test] Static scene viewer is ready.")
    print("[scene_view_test] camera position=%s lookat=%s"
          % (camera_position, camera_target))
    print("[scene_view_test] No policy, navigation, simulation stepping, or file output.")
    print("[scene_view_test] Close the viewer window or press Esc to exit.")

    try:
        while not env.gym.query_viewer_has_closed(env.viewer):
            should_quit = False
            for event in env.gym.query_viewer_action_events(env.viewer):
                if event.action == "QUIT" and event.value > 0:
                    should_quit = True
                    break
            if should_quit:
                break
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, False)
            env.gym.poll_viewer_events(env.viewer)
            time.sleep(1.0 / 60.0)
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
            env.viewer = None
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main(get_args())

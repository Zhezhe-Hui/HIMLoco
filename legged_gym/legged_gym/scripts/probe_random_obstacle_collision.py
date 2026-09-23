"""Verify that a creation-randomized static obstacle produces physical contact."""

import isaacgym  # noqa: F401 - must be imported before torch

import torch
from isaacgym import gymtorch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def probe(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    env.gym.refresh_actor_root_state_tensor(env.sim)
    stone = env.stone_root_states[0, 0].clone()
    robot = env.root_states[0].clone()
    robot[0:2] = stone[0:2]
    robot[2] = stone[2] + 0.15
    robot[7:13] = 0.0
    env.all_root_states[0] = robot

    actor_index = torch.tensor([0], dtype=torch.int32, device=env.device)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.all_root_states),
        gymtorch.unwrap_tensor(actor_index),
        1,
    )

    max_force = 0.0
    for _ in range(10):
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.gym.refresh_net_contact_force_tensor(env.sim)
        force = torch.norm(
            env.contact_forces[0, env.termination_contact_indices, :], dim=-1)
        max_force = max(max_force, float(force.max().item()))

    print(
        "COLLISION_PROBE stone_xy=(%.3f, %.3f) base_contact_max=%.3fN"
        % (stone[0].item(), stone[1].item(), max_force)
    )
    if max_force <= 1.0:
        raise RuntimeError("随机石头位置未检测到有效机身接触力")


if __name__ == "__main__":
    probe(get_args())

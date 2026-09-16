# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

# =====================================================================
#  截图/出图开关已统一到 viz_config.py
#  改开关请编辑: legged_gym/legged_gym/envs/go2/viz_config.py
#  （ACTIVE_PRESET = "off" 可一键关掉所有截图）
# =====================================================================
from legged_gym.envs.go2.viz_config import (
    LEG_CLOSEUP_CAPTURE, LEG_CLOSEUP_DIR, LEG_CLOSEUP_START_STEP,
    LEG_CLOSEUP_FRAME_INTERVAL, LEG_CLOSEUP_MAX_FRAMES,
    EXPORT_POLICY, RECORD_FRAMES, MOVE_CAMERA, NUM_ENVS,
    PLAY_VEL_X, PLAY_VEL_Y, PLAY_VEL_YAW,
    LEG_CAMERA_FORWARD_M, LEG_CAMERA_LEFT_M, LEG_CAMERA_HEIGHT_M,
    LEG_CAMERA_LOOKAT_FORWARD_M, LEG_CAMERA_LOOKAT_HEIGHT_M,
)
from legged_gym.envs.go2.viz_config import ensure_dirs as _ensure_viz_dirs, summary as _viz_summary
from legged_gym.envs.go2.viz_config import (
    EVAL_OVERRIDES, POLICY_LOAD_RUN, POLICY_CHECKPOINT,
)
from legged_gym.envs.go2.viz_config import (
    PLAY_LOG_ROBOT_INDEX, PLAY_LOG_JOINT_INDEX, PLAY_LOG_STATE_STEPS,
    PLAY_TOTAL_STEPS_MULTIPLIER, PLAY_PLOT_STATES,
)

_ensure_viz_dirs()
print("[viz_config] " + _viz_summary())


def _get_robot_yaw(env, robot_index=0):
    quat = env.base_quat[robot_index].detach().cpu().numpy()
    x, y, z, w = quat
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _set_leg_closeup_camera(env, robot_index=0):
    base_pos = env.root_states[robot_index, :3].detach().cpu().numpy()
    yaw = _get_robot_yaw(env, robot_index)

    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])

    # Front-right overhead view: full robot in frame, legs visible, obstacle ahead visible.
    # 机位参数见 viz_config.py 第十五节
    camera_position = (base_pos + LEG_CAMERA_FORWARD_M * forward
                       + LEG_CAMERA_LEFT_M * left
                       + np.array([0.0, 0.0, LEG_CAMERA_HEIGHT_M]))
    camera_target = (base_pos + LEG_CAMERA_LOOKAT_FORWARD_M * forward
                     + np.array([0.0, 0.0, LEG_CAMERA_LOOKAT_HEIGHT_M]))
    env.set_camera(camera_position, camera_target)


def _save_leg_closeup_frame(env, frame_id):
    if env.viewer is None:
        return False
    os.makedirs(LEG_CLOSEUP_DIR, exist_ok=True)
    _set_leg_closeup_camera(env, robot_index=0)
    env.render(sync_frame_time=False)
    filename = os.path.join(LEG_CLOSEUP_DIR, f"leg_closeup_{frame_id:03d}.png")
    env.gym.write_viewer_image_to_file(env.viewer, filename)
    return True


def play(args, x_vel=PLAY_VEL_X, y_vel=PLAY_VEL_Y, yaw_vel=PLAY_VEL_YAW):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = NUM_ENVS
    # 噪声/域随机化覆盖项统一走 viz_config.EVAL_OVERRIDES（默认全关，与原行为一致）
    for (section, key), value in EVAL_OVERRIDES:
        setattr(getattr(env_cfg, section), key, value)
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    # 显式指定 run / checkpoint，避免 get_load_path 按目录名字符串排序误选模型
    train_cfg.runner.load_run = POLICY_LOAD_RUN
    train_cfg.runner.checkpoint = POLICY_CHECKPOINT
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    # 日志索引/步数统一在 viz_config.py 第十五节配置
    robot_index = PLAY_LOG_ROBOT_INDEX   # which robot is used for logging
    joint_index = PLAY_LOG_JOINT_INDEX   # which joint is used for logging
    stop_state_log = PLAY_LOG_STATE_STEPS  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # 基于episode_length_s计算的步数阈值
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # === 初始化性能指标跟踪 ===
    metrics = {
        'vel_tracking_mae': 0.0,
        'energy': 0.0,
        'orientation_error': 0.0,
        'angular_vel_error': 0.0,
        'step_count': 0
    }
    metrics_printed = False  # 标记是否已打印性能指标
    leg_closeup_frame_id = 0
    if LEG_CLOSEUP_CAPTURE:
        os.makedirs(LEG_CLOSEUP_DIR, exist_ok=True)
        print(f"Leg close-up frames will be saved to: {LEG_CLOSEUP_DIR}")

    # 运行到episode结束（基于episode_length_s配置）
    total_steps = int(env.max_episode_length * PLAY_TOTAL_STEPS_MULTIPLIER)
    
    for i in range(total_steps):
        actions = policy(obs.detach())
        env.commands[:, 0] = x_vel
        env.commands[:, 1] = y_vel
        env.commands[:, 2] = yaw_vel
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())

        # === 持续累加性能指标 ===
        # 1. 速度跟踪MAE (平均绝对误差)
        cmd_vel = env.commands[robot_index, :2]
        actual_vel = env.base_lin_vel[robot_index, :2]
        vel_error = torch.mean(torch.abs(cmd_vel - actual_vel)).item()
        metrics['vel_tracking_mae'] += vel_error
        
        # 2. 平均扭矩成本 (能量消耗)
        torque_magnitude = torch.mean(torch.abs(env.torques[robot_index])).item()
        metrics['energy'] += torque_magnitude
        
        # 3. 姿态误差 (投影重力的xy分量的范数)
        orientation_error = torch.norm(env.projected_gravity[robot_index, :2]).item()
        metrics['orientation_error'] += orientation_error
        
        # 4. 角速度误差 (偏航角速度误差的绝对值)
        cmd_ang_vel = env.commands[robot_index, 2]
        actual_ang_vel = env.base_ang_vel[robot_index, 2]
        ang_vel_error = torch.abs(cmd_ang_vel - actual_ang_vel).item()
        metrics['angular_vel_error'] += ang_vel_error
        
        # 更新步数计数
        metrics['step_count'] += 1

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if LEG_CLOSEUP_CAPTURE and leg_closeup_frame_id < LEG_CLOSEUP_MAX_FRAMES:
            should_capture = i >= LEG_CLOSEUP_START_STEP and (i - LEG_CLOSEUP_START_STEP) % LEG_CLOSEUP_FRAME_INTERVAL == 0
            if should_capture and _save_leg_closeup_frame(env, leg_closeup_frame_id):
                leg_closeup_frame_id += 1

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i == stop_state_log:
            if PLAY_PLOT_STATES:
                logger.plot_states()
        
        # 奖励日志打印逻辑（原逻辑）
        if 0 < i < stop_rew_log:
            if infos.get("episode", None):
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            # 打印奖励日志
            logger.print_rewards()
            
            # 立即打印性能指标（与奖励日志一起输出）
            if metrics['step_count'] > 0 and not metrics_printed:
                # 计算平均值
                avg_vel_mae = metrics['vel_tracking_mae'] / metrics['step_count']
                avg_energy = metrics['energy'] / metrics['step_count']
                avg_ori_error = metrics['orientation_error'] / metrics['step_count']
                avg_ang_vel_error = metrics['angular_vel_error'] / metrics['step_count']
                
                # 按照要求的格式打印性能指标
                print("\nPerformance Metrics:")
                print(f" - Velocity Tracking MAE : {avg_vel_mae:.3f} m/s")
                print(f" - Avg Torque Cost      : {avg_energy:.3f} N·m")
                print(f" - Orientation Error     : {avg_ori_error:.3f} rad")
                print(f" - Angular Vel Error     : {avg_ang_vel_error:.3f} rad/s")
                print(f"Total number of steps: {metrics['step_count']}")
                metrics_printed = True
    
    # 兜底：如果到循环结束还没打印，强制打印
    if not metrics_printed and metrics['step_count'] > 0:
        avg_vel_mae = metrics['vel_tracking_mae'] / metrics['step_count']
        avg_energy = metrics['energy'] / metrics['step_count']
        avg_ori_error = metrics['orientation_error'] / metrics['step_count']
        avg_ang_vel_error = metrics['angular_vel_error'] / metrics['step_count']
        
        print("\nPerformance Metrics:")
        print(f" - Velocity Tracking MAE : {avg_vel_mae:.3f} m/s")
        print(f" - Avg Torque Cost      : {avg_energy:.3f} N·m")
        print(f" - Orientation Error     : {avg_ori_error:.3f} rad")
        print(f" - Angular Vel Error     : {avg_ang_vel_error:.3f} rad/s")
        print(f"Total number of steps: {metrics['step_count']}")

if __name__ == '__main__':
    args = get_args()
    play(args, x_vel=PLAY_VEL_X, y_vel=PLAY_VEL_Y, yaw_vel=PLAY_VEL_YAW)

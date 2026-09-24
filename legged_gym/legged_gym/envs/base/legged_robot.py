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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs
        self.history_length = int(self.num_obs / self.num_one_step_obs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
    def compute_tracking_error(self, target_position):
        """
        计算目标点与当前机器人的位置误差（欧几里得误差）
        Args:
            target_position (torch.Tensor): 目标点的坐标 (x, y, z)
        Returns:
            tracking_error (torch.Tensor): 当前误差
        """
        # 从 root_states 中获取当前机器人的基座位置 (x, y, z)
        current_position = self.root_states[:, :3]  # 假设 root_states 中前3个元素是位置(x, y, z)
        
        # 计算欧几里得误差
        tracking_error = torch.norm(current_position - target_position, dim=1)
        
        return tracking_error

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.delayed_actions = self.actions.clone().view(self.num_envs, 1, self.num_actions).repeat(1, self.cfg.control.decimation, 1)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[:, i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.delayed_actions[:, _]).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.total_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.total_bodies, 13)[:, self.feet_indices, 7:10]

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)


        self.disturbance[:, :, :] = 0.0
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, termination_privileged_obs

    def check_termination(self):
        """
        Check if environments need to be reset

        各项终止条件的开关与阈值都来自 cfg.termination（集中配置，勿在此硬编码）。
        """
        term_cfg = getattr(self.cfg, "termination", None)
        # 1️碰撞终止
        if term_cfg is None or getattr(term_cfg, "enable_collision", True):
            threshold = float(getattr(term_cfg, "collision_force_threshold", 1.0)) \
                if term_cfg is not None else 1.0
            collision = torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > threshold,
                dim=1
            )
        else:
            collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 2️超时终止
        if term_cfg is None or getattr(term_cfg, "enable_timeout", True):
            self.time_out_buf = self.episode_length_buf > self.max_episode_length
        else:
            self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 3️悬崖 / 掉落
        if term_cfg is not None and getattr(term_cfg, "enable_cliff_fall", True):
            cliff_fall = self.root_states[:, 2] < float(getattr(term_cfg, "cliff_fall_height", -0.5))
        else:
            cliff_fall = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 4️被卡住终止
        if term_cfg is not None and getattr(term_cfg, "enable_stuck", True):
            cmd_speed = torch.norm(self.commands[:, :2], dim=1)
            actual_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
            # 判断是否“应该在动但没动”
            stuck_mask = (cmd_speed > float(getattr(term_cfg, "stuck_cmd_speed_min", 0.3))) & \
                         (actual_speed < float(getattr(term_cfg, "stuck_actual_speed_max", 0.05)))
            self.stuck_counter[stuck_mask] += 1
            self.stuck_counter[~stuck_mask] = 0
            stuck = self.stuck_counter > self.stuck_steps_threshold
        else:
            stuck = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 汇总所有终止条件
        self.reset_buf = collision | self.time_out_buf | cliff_fall | stuck

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.reset_buf[env_ids] = 1

        # update height measurements
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        
         #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), 1), device=self.device)
        self.refresh_actor_rigid_shape_props(env_ids)
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        self.obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs], self.obs_buf[:, :-self.num_one_step_obs]), dim=-1)
        self.privileged_obs_buf = torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)

    def get_current_obs(self):
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        return current_obs
        
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        return torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)[env_ids]
        
            
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -2., 2.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]

        self.commands[high_vel_env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(high_vel_env_ids), 1), device=self.device).squeeze(1)

        # set y commands of high vel envs to zero
        self.commands[high_vel_env_ids, 1:2] *= (torch.norm(self.commands[high_vel_env_ids, 0:1], dim=1) < 1.0).unsqueeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *=self.cfg.control.hip_reduction
        self.joint_pos_target = self.default_dof_pos + actions_scaled
        # === Hip Roll Joint Clamp (ADD THIS) ===
        self.joint_pos_target[:, [0, 3, 6, 9]] = torch.clamp(
            self.joint_pos_target[:, [0, 3, 6, 9]], -0.1, 0.1
        )
        # ========================================
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
            """ Resets DOF position and velocities of selected environmments
            Positions are randomly selected within 0.5:1.5 x default positions.
            Velocities are set to zero.

            Args:
                env_ids (List[int]): Environemnt ids
            """
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
            self.dof_vel[env_ids] = 0.

            # 计算机器人的真实 Actor 索引
            # robot_indices = env_id * num_actors_per_env
            robot_indices = (env_ids * self.num_actors_per_env).to(dtype=torch.int32)
            
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(robot_indices), len(robot_indices))

    @staticmethod
    def _sample_separated_xy(count, x_range, y_range, min_separation, rng=None):
            """Uniformly sample obstacle centers with a minimum pairwise distance."""
            rng = np.random if rng is None else rng
            points = []
            attempts = 0
            max_attempts = max(1000, int(count) * 300)
            min_sep_sq = float(min_separation) ** 2
            while len(points) < int(count) and attempts < max_attempts:
                attempts += 1
                candidate = np.array([
                    rng.uniform(float(x_range[0]), float(x_range[1])),
                    rng.uniform(float(y_range[0]), float(y_range[1])),
                ], dtype=np.float32)
                if all(float(np.sum((candidate - point) ** 2)) >= min_sep_sq
                       for point in points):
                    points.append(candidate)
            if len(points) != int(count):
                raise RuntimeError(
                    "无法在区域 x=%s y=%s 内放置 %d 个最小间距 %.2fm 的障碍物"
                    % (x_range, y_range, count, min_separation))
            return points

    def _randomize_navigation_obstacles(self, env_ids):
            """Reject unsafe reset-time relocation of fixed GPU PhysX actors."""
            stone_cfg = getattr(self.cfg, "stone", None)
            tree_cfg = getattr(self.cfg, "static_obstacles", None)
            randomize_stones = bool(getattr(stone_cfg, "randomize_each_reset", False))
            randomize_trees = bool(getattr(tree_cfg, "randomize_each_reset", False))
            if randomize_stones or randomize_trees:
                raise RuntimeError(
                    "GPU PhysX 不支持仿真启动后移动 fix_base_link 静态障碍。"
                    "请使用 randomize_at_creation，并通过 HIMLOCO_SCENE_SEED "
                    "为每个进程生成新布局。")
            return []

    def _reset_root_states(self, env_ids):
            """ Resets ROOT states position and velocities of selected environmments
                Sets base position based on the curriculum
                Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
            Args:
                env_ids (List[int]): Environemnt ids
            """
            if len(env_ids) == 0:
                return

            # === 1. 重置机器人 (Robot) ===
            robot_indices = (env_ids * self.num_actors_per_env).to(dtype=torch.long)
            num_resets = len(env_ids)

            robot_states_update = self.base_init_state.unsqueeze(0).repeat(num_resets, 1)

            # 出生点开关见 cfg.spawn（go2 由 viz_config.SPAWN_* 提供）
            spawn_cfg = getattr(self.cfg, "spawn", None)
            use_origins = bool(getattr(spawn_cfg, "use_env_origins", False)) \
                if spawn_cfg is not None else False
            if use_origins:
                robot_states_update[:, :3] += self.env_origins[env_ids]
                jitter = float(getattr(spawn_cfg, "origin_xy_jitter", 1.0))
                if jitter > 0:
                    xy_offset = ((torch.rand(num_resets, 2, device=self.device) * 2.0)
                                 - 1.0) * jitter
                    robot_states_update[:, :2] += xy_offset
            else:
                # 固定世界坐标模式也可按范围逐 trial 随机，供随机直线走廊评测。
                fixed = tuple(getattr(spawn_cfg, "fixed_position", (6.0, 9.0, 0.4))
                              if spawn_cfg is not None else (6.0, 9.0, 0.4))
                randomize_spawn = bool(getattr(spawn_cfg, "randomize_each_reset", False)) \
                    if spawn_cfg is not None else False
                if randomize_spawn:
                    x_range = tuple(getattr(spawn_cfg, "random_x_range", (fixed[0], fixed[0])))
                    y_range = tuple(getattr(spawn_cfg, "random_y_range", (fixed[1], fixed[1])))
                    trial_seed = getattr(self, "navigation_trial_seed", None)
                    if trial_seed is not None:
                        xy_values = []
                        for env_id in env_ids.detach().cpu().tolist():
                            rng = np.random.RandomState(int(trial_seed) + int(env_id))
                            xy_values.append([
                                rng.uniform(float(x_range[0]), float(x_range[1])),
                                rng.uniform(float(y_range[0]), float(y_range[1])),
                            ])
                        xy_tensor = torch.tensor(
                            xy_values, dtype=robot_states_update.dtype, device=self.device)
                        robot_states_update[:, 0:2] = xy_tensor
                    else:
                        robot_states_update[:, 0] = (
                            torch.rand(num_resets, device=self.device)
                            * (float(x_range[1]) - float(x_range[0])) + float(x_range[0]))
                        robot_states_update[:, 1] = (
                            torch.rand(num_resets, device=self.device)
                            * (float(y_range[1]) - float(y_range[0])) + float(y_range[0]))
                else:
                    robot_states_update[:, 0] = float(fixed[0])
                    robot_states_update[:, 1] = float(fixed[1])
                robot_states_update[:, 2] = float(fixed[2])

            vel_range = float(getattr(spawn_cfg, "init_velocity_range", 1.0)) \
                if spawn_cfg is not None else 1.0
            if vel_range > 0:
                robot_states_update[:, 7:13] = \
                    (torch.rand(num_resets, 6, device=self.device) - 0.5) * vel_range
            self.all_root_states[robot_indices] = robot_states_update

            # 静态障碍已在 actor 创建前确定位置，reset 时绝不移动；这里只提交机器人。
            obstacle_indices = self._randomize_navigation_obstacles(env_ids)

            # === 2. 更新物理引擎 ===
            # 关键修改：只将机器人的索引传给物理引擎
            # 石头的索引不传进去，物理引擎就不会去更新它们的位置
            if obstacle_indices:
                obstacle_indices_tensor = torch.tensor(
                    obstacle_indices, dtype=torch.long, device=self.device)
                actor_indices = torch.cat((robot_indices, obstacle_indices_tensor))
            else:
                actor_indices = robot_indices
            actor_indices_int32 = actor_indices.to(dtype=torch.int32)

            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.all_root_states),
                gymtorch.unwrap_tensor(actor_indices_int32),
                len(actor_indices_int32)
            )

    def _push_robots(self):
            """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
            """
            max_vel = self.cfg.domain_rand.max_push_vel_xy
            
            # 1. 修改速度 (在 View 上操作，会自动同步到 all_root_states)
            # 使用 torch.rand 替代 torch_rand_float 以保持稳健性
            # range [-max, max] -> (rand * 2 - 1) * max
            self.root_states[:, 7:9] = (torch.rand(self.num_envs, 2, device=self.device) * 2.0 - 1.0) * max_vel
            
            # 2. 更新物理引擎
            # 关键修复：不能直接传 self.root_states (因为它不连续)
            # 我们使用 set_actor_root_state_tensor_indexed，传入完整的 all_root_states
            
            # 计算所有机器人的索引: [0, 1*N, 2*N, ...]
            robot_indices = torch.arange(self.num_envs, dtype=torch.int32, device=self.device) * self.num_actors_per_env
            
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.all_root_states), # 传入连续的大张量
                gymtorch.unwrap_tensor(robot_indices),        # 只更新机器人
                len(robot_indices)
            )

    def _disturbance_robots(self):
        """ Random add disturbance force to the robots.
        """
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, 0, :] = disturbance
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        low_vel_env_ids = (env_ids > (self.num_envs * 0.2))
        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        low_vel_env_ids = env_ids[low_vel_env_ids.nonzero(as_tuple=True)]
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_lin_vel"][low_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]) and (torch.mean(self.episode_sums["tracking_lin_vel"][high_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])\
        if self.cfg.terrain.measure_heights:
            noise_vec = torch.zeros(9 + 3*self.num_actions + 187, device=self.device)
        else:
            noise_vec = torch.zeros(9 + 3*self.num_actions, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = 0. # commands
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:(9 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(9 + self.num_actions):(9 + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(9 + 2 * self.num_actions):(9 + 3 * self.num_actions)] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions + 187)] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        #noise_vec[232:] = 0
        return noise_vec

    def _get_num_stones(self):
            """石头总数 = 大石头 + 小碎石。

            优先用解耦后的 num_big_stones / num_small_stones；
            若旧配置只给了 num_stones，则退回按 big_stone_indices 推断大石头数。
            """
            stone_cfg = getattr(self.cfg, "stone", None)
            if stone_cfg is None or not getattr(stone_cfg, "enable", False):
                return 0
            num_big = getattr(stone_cfg, "num_big_stones", None)
            num_small = getattr(stone_cfg, "num_small_stones", None)
            if num_big is not None and num_small is not None:
                return max(0, int(num_big)) + max(0, int(num_small))
            return int(getattr(stone_cfg, "num_stones", 0))

    def _stone_cfg_get(self, name, default=None):
            """读取 cfg.stone 上的可选字段，缺失时返回 default。"""
            stone_cfg = getattr(self.cfg, "stone", None)
            if stone_cfg is None:
                return default
            return getattr(stone_cfg, name, default)

    def _get_num_big_stones(self):
            """固定大石头的数量。

            大石头占 num_stones 序列的前 N 个索引，且必须存在对应位置，
            因此取 big_stone_indices 与 big_stone_positions 的交集长度，避免越界。
            """
            if not self._stone_cfg_get("enable", False):
                return 0
            # 解耦后的大石头数量优先；否则退回 big_stone_indices 与位置的交集
            num_big = self._stone_cfg_get("num_big_stones", None)
            positions = self._stone_cfg_get("big_stone_positions", []) or []
            if num_big is not None:
                return max(0, min(int(num_big), len(positions)))
            indices = self._stone_cfg_get("big_stone_indices", []) or []
            return min(len(indices), len(positions))

    def _apply_stone_collision_options(self, options, kind):
            """按配置调整石头碰撞体开销（mesh 碰撞非常贵，可用 VHACD 凸分解替代）。

            kind: "small" 或 "big"，分别对应 cfg.stone.small_collision / big_collision。
            取值:
              "mesh"  直接用 URDF 里的三角面网格做碰撞（最贵，最贴近外形）
              "vhacd" 凸分解（折中， PhysX 对凸体有专门的快速通道）
              其它值  不做额外设置，沿用 IsaacGym 默认行为
            """
            mode = str(self._stone_cfg_get(f"{kind}_collision",
                                           "mesh" if kind == "big" else "vhacd")).lower()
            if mode == "vhacd" and getattr(options, "vhacd_enabled", None) is not None:
                options.vhacd_enabled = True
                params = getattr(options, "vhacd_params", None)
                cfg_params = self._stone_cfg_get(f"{kind}_vhacd_params", {}) or {}
                if params is not None:
                    for key, value in cfg_params.items():
                        if hasattr(params, key):
                            setattr(params, key, value)
            return mode

    def _quantize_stone_scale(self, scale, kind):
            """把连续随机缩放量化到有限档位。

            IsaacGym/PhysX 为每种 actor scale 生成并缓存独立碰撞几何，
            连续 scale 会导致每个石头都新建一份，显存与构建时间线性膨胀。
            量化成 N 档后可大量复用缓存（N=0 表示不量化，保持原行为）。
            """
            steps = int(self._stone_cfg_get(f"{kind}_scale_steps", 0) or 0)
            if steps <= 0:
                return float(scale)
            s_min = float(self._stone_cfg_get("scale_min", 0.1))
            s_max = float(self._stone_cfg_get("scale_max", 0.3))
            if kind == "big":
                return float(scale)
            if s_max <= s_min:
                return float(scale)
            idx = int(round((float(scale) - s_min) / (s_max - s_min) * steps))
            idx = min(max(idx, 0), steps)
            return s_min + (s_max - s_min) * idx / steps

    #----------------------------------------
    def _init_buffers(self):
            """ Initialize torch tensors which will contain simulation states and processed quantities
            """
            # get gym GPU state tensors
            actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
            dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
            net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
            rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            # ===== 卡住判断 =====
            self.stuck_counter = torch.zeros(self.num_envs,device=self.device,dtype=torch.int32)

            # === 1. 计算总 Actor 数 (用于 Root States) ===
            num_stones = self._get_num_stones()
            num_static_obstacles = getattr(self, "num_static_obstacle_actors", 0)
            self.num_actors_per_env = 1 + num_stones + num_static_obstacles # 1个机器人 + N个石头 + M个树障碍物

            # === 2. 计算总 Body 数 (用于 Rigid Body States) ===
            stone_bodies_count = getattr(self, 'num_stone_bodies', 1)
            stone_bodies_total = getattr(self, "num_stone_bodies_total", num_stones * stone_bodies_count)
            self.total_bodies = self.num_bodies + stone_bodies_total + getattr(self, "num_static_obstacle_bodies", 0)

            # --- 处理 Root States ---
            self.all_root_states = gymtorch.wrap_tensor(actor_root_state)
            self.root_states_reshaped = self.all_root_states.view(self.num_envs, self.num_actors_per_env, 13)
            self.root_states = self.root_states_reshaped[:, 0, :] 
            if num_stones > 0:
                self.stone_root_states = self.root_states_reshaped[:, 1:1 + num_stones, :]
            if num_static_obstacles > 0:
                self.static_obstacle_root_states = self.root_states_reshaped[:, 1 + num_stones:, :]

            # 定义 base_quat
            self.base_quat = self.root_states[:, 3:7]

            # --- 处理 Dof States ---
            self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
            self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
            self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

            # --- 处理 Rigid Body States ---
            self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
            temp_rigid_body = self.rigid_body_states.view(self.num_envs, self.total_bodies, 13)
            self.feet_pos = temp_rigid_body[:, self.feet_indices, 0:3]
            self.feet_vel = temp_rigid_body[:, self.feet_indices, 7:10]

            # --- 处理 Contact Forces ---
            self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) 

            # initialize some data used later on
            self.common_step_counter = 0
            self.extras = {}
            self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
            self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
            self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
            self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.last_dof_vel = torch.zeros_like(self.dof_vel)
            self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
            self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) 
            self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)
            self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
            self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
            
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
            
            if self.cfg.terrain.measure_heights:
                self.height_points = self._init_height_points()
            self.measured_heights = self._get_heights()
            self.base_height_points = self._init_base_height_points()

            # joint positions offsets and PD gains
            self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(self.num_dofs):
                name = self.dof_names[i]
                angle = self.cfg.init_state.default_joint_angles[name]
                self.default_dof_pos[i] = angle
                found = False
                for dof_name in self.cfg.control.stiffness.keys():
                    if dof_name in name:
                        self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                        self.d_gains[i] = self.cfg.control.damping[dof_name]
                        found = True
                if not found:
                    self.p_gains[i] = 0.
                    self.d_gains[i] = 0.
                    if self.cfg.control.control_type in ["P", "V"]:
                        print(f"PD gain of joint {name} were not defined, setting them to zero")
            self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
            
            # randomize prop
            self.Kp_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            self.Kd_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            self.motor_strength_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
            # 关键修复： disturbance 必须使用 self.total_bodies
            self.disturbance = torch.zeros(self.num_envs, self.total_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
                
            if self.cfg.domain_rand.randomize_kp:
                self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, 1), device=self.device)
            if self.cfg.domain_rand.randomize_kd:
                self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, 1), device=self.device)
            if self.cfg.domain_rand.randomize_motor_strength:
                self.motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, 1), device=self.device)
            if self.cfg.domain_rand.randomize_payload_mass:
                self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
            if self.cfg.domain_rand.randomize_com_displacement:
                self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
                
            self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

            # === 新增：将石头初始位置转换为 Tensor 以支持向量化重置 ===
            if num_stones > 0:
                self.stone_start_poses_tensor = torch.tensor(
                    self.saved_stone_start_poses, 
                    device=self.device, 
                    dtype=torch.float
                )
            else:
                self.stone_start_poses_tensor = None
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _cfg_get(self, cfg, key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _quat_from_euler_xyz(self, rpy):
        roll, pitch, yaw = [float(v) for v in rpy]
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        return gymapi.Quat(
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

    def _get_static_obstacle_asset_options(self, spec):
        cfg = self.cfg.static_obstacles
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = 0
        for attr in [
            "collapse_fixed_joints",
            "replace_cylinder_with_capsule",
            "flip_visual_attachments",
            "fix_base_link",
            "density",
            "angular_damping",
            "linear_damping",
            "max_angular_velocity",
            "max_linear_velocity",
            "armature",
            "thickness",
            "disable_gravity",
            "vhacd_enabled",
            "convex_decomposition_from_submeshes",
        ]:
            value = self._cfg_get(spec, attr, self._cfg_get(cfg, attr))
            if value is None or not hasattr(asset_options, attr):
                continue
            setattr(asset_options, attr, value)

        # 树的 collision mesh 动辄几十万三角面，PhysX cooking 极慢（实测 4 棵树 ~38s）。
        # 需要时可在这里透传 vhacd_params 做凸分解，或降低 vhacd 分辨率。
        vhacd_params = self._cfg_get(spec, "vhacd_params", self._cfg_get(cfg, "vhacd_params"))
        if vhacd_params and getattr(asset_options, "vhacd_enabled", False):
            params = getattr(asset_options, "vhacd_params", None)
            if params is not None:
                for key, value in dict(vhacd_params).items():
                    if hasattr(params, key):
                        setattr(params, key, value)
        return asset_options

    def _load_static_obstacle_assets(self):
        self.static_obstacle_specs = []
        obstacle_cfg = getattr(self.cfg, "static_obstacles", None)
        if obstacle_cfg is None or not getattr(obstacle_cfg, "enable", False):
            return

        # 同一个 URDF 常被多个 spec 引用（例如两棵 snow_tree）。高模 mesh 的
        # PhysX cooking 每次都要重新做一遍，实测单棵树就要 7~13s，因此按
        # (文件, asset options, 摩擦/弹性) 缓存已加载的 asset 句柄复用。
        asset_cache = {}
        if not bool(getattr(obstacle_cfg, "reuse_duplicate_assets", True)):
            asset_cache = None

        for spec in getattr(obstacle_cfg, "assets", []):
            positions = self._cfg_get(spec, "positions", [])
            if len(positions) == 0:
                continue

            asset_file_cfg = self._cfg_get(spec, "file")
            if asset_file_cfg is None:
                raise ValueError("Each static obstacle asset must define a 'file' path")

            asset_path = asset_file_cfg.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            asset_root = os.path.dirname(asset_path)
            asset_file = os.path.basename(asset_path)
            asset_options = self._get_static_obstacle_asset_options(spec)
            friction = self._cfg_get(spec, "static_friction", getattr(obstacle_cfg, "static_friction", self.cfg.terrain.static_friction))
            restitution = self._cfg_get(spec, "restitution", getattr(obstacle_cfg, "restitution", self.cfg.terrain.restitution))

            cache_key = None
            if asset_cache is not None:
                # friction/restitution 通过 set_asset_rigid_shape_properties 作用在整个
                # asset 上，所以必须参与缓存键，否则复用会串改上一个 spec 的材质。
                cache_key = (asset_path, friction, restitution,
                             tuple(getattr(asset_options, attr, None) for attr in (
                                 "collapse_fixed_joints", "replace_cylinder_with_capsule",
                                 "flip_visual_attachments", "fix_base_link", "density",
                                 "armature", "thickness", "disable_gravity",
                                 "vhacd_enabled", "convex_decomposition_from_submeshes")))
                if cache_key in asset_cache:
                    asset, num_bodies = asset_cache[cache_key]
                else:
                    asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
                    num_bodies = self._apply_static_obstacle_material(
                        asset, obstacle_cfg, friction, restitution)
                    asset_cache[cache_key] = (asset, num_bodies)
            else:
                asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
                num_bodies = self._apply_static_obstacle_material(
                    asset, obstacle_cfg, friction, restitution)

            self.static_obstacle_specs.append({
                "asset": asset,
                "name": self._cfg_get(spec, "name", os.path.splitext(asset_file)[0]),
                "positions": positions,
                "rpy": self._cfg_get(spec, "rpy", [0.0, 0.0, 0.0]),
                "scale": float(self._cfg_get(spec, "scale", 1.0)),
                "place_on_terrain": self._cfg_get(spec, "place_on_terrain", getattr(obstacle_cfg, "place_on_terrain", True)),
                "use_env_origin": self._cfg_get(spec, "use_env_origin", getattr(obstacle_cfg, "use_env_origin", False)),
                "collision_filter": int(self._cfg_get(spec, "collision_filter", 0)),
                "segmentation_id": int(self._cfg_get(spec, "segmentation_id", 0)),
                "num_bodies": num_bodies,
            })

    def _apply_static_obstacle_material(self, asset, obstacle_cfg, friction, restitution):
        """设置摩擦力/弹性，返回刚体数量。抽出来是为了 asset 缓存复用。"""
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(asset)
        for shape_prop in rigid_shape_props:
            shape_prop.friction = friction
            shape_prop.restitution = restitution
        self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)
        return self.gym.get_asset_rigid_body_count(asset)

    def _get_terrain_height_at(self, x, y):
        if not hasattr(self, "terrain") or self.height_samples is None:
            return 0.0

        px = int((x + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale)
        py = int((y + self.terrain.cfg.border_size) / self.terrain.cfg.horizontal_scale)
        px = np.clip(px, 0, self.terrain.tot_rows - 1)
        py = np.clip(py, 0, self.terrain.tot_cols - 1)
        return float(self.terrain.height_field_raw[px, py]) * self.terrain.cfg.vertical_scale

    def _get_static_obstacle_position(self, position_cfg, env_id, spec):
        position_cfg = list(position_cfg)
        if len(position_cfg) == 2:
            position_cfg.append(0.0)
        if len(position_cfg) != 3:
            raise ValueError("Static obstacle positions must be [x, y] or [x, y, z_offset]")

        position = np.array(position_cfg, dtype=np.float32)
        if spec["use_env_origin"]:
            position += self.env_origins[env_id].detach().cpu().numpy()
        if spec["place_on_terrain"]:
            position[2] = self._get_terrain_height_at(position[0], position[1]) + float(position_cfg[2])
        return gymapi.Vec3(float(position[0]), float(position[1]), float(position[2]))

    def _create_static_obstacles(self, env_handle, env_id):
        handles = []
        random_positions = getattr(self, "_creation_random_tree_positions", None)
        random_cursor = 0
        for spec in self.static_obstacle_specs:
            obstacle_rotation = self._quat_from_euler_xyz(spec["rpy"])
            for obstacle_id, position_cfg in enumerate(spec["positions"]):
                if random_positions is not None:
                    position_cfg = [
                        float(random_positions[random_cursor][0]),
                        float(random_positions[random_cursor][1]),
                        float(getattr(self.cfg.static_obstacles, "random_z_offset", 0.02)),
                    ]
                    random_cursor += 1
                obstacle_pose = gymapi.Transform()
                obstacle_pose.p = self._get_static_obstacle_position(position_cfg, env_id, spec)
                obstacle_pose.r = obstacle_rotation
                actor_handle = self.gym.create_actor(
                    env_handle,
                    spec["asset"],
                    obstacle_pose,
                    f"{spec['name']}_{obstacle_id}",
                    0,
                    spec["collision_filter"],
                    spec["segmentation_id"],
                )
                if spec["scale"] != 1.0:
                    self.gym.set_actor_scale(env_handle, actor_handle, spec["scale"])
                handles.append(actor_handle)
        return handles

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # ===== Camera (only for env0) =====
        self.cam_handle = None
        self.cam_env_handle = None
        self.camera_depth_tensor = None

        num_stones = self._get_num_stones()
        num_big_stones = self._get_num_big_stones()
        num_small_stones = max(0, num_stones - num_big_stones)
        self.num_big_stones = num_big_stones
        self.num_small_stones = num_small_stones
        stone_asset = None
        big_stone_asset = None
        if num_stones > 0:
            stone_root = os.path.dirname(
                os.path.join(LEGGED_GYM_ROOT_DIR, self._stone_cfg_get("small_urdf",
                                                                       "resources/stone/stone.urdf")))

            # 小碎石。默认静态化（fix_base_link + disable_gravity）：
            # 动态 mesh 刚体是 PhysX 里最贵的组合，985 个/env 时显存与 step 时间都会爆炸。
            small_static = self._stone_cfg_get("small_static", True)
            small_options = gymapi.AssetOptions()
            small_options.fix_base_link = bool(small_static)
            small_options.disable_gravity = bool(small_static)
            small_options.density = float(self._stone_cfg_get("density", 1000.0))
            # 连续随机 scale 会让每个石头生成独立碰撞几何；量化成有限档位可复用缓存
            self._apply_stone_collision_options(small_options, "small")
            if num_small_stones > 0:
                stone_asset = self.gym.load_asset(self.sim, stone_root, "stone.urdf", small_options)

            # 大石头（静态 + 贴图），只在需要时加载
            big_options = gymapi.AssetOptions()
            big_options.fix_base_link = True
            big_options.disable_gravity = True
            big_options.density = float(self._stone_cfg_get("density", 1000.0))
            self._apply_stone_collision_options(big_options, "big")
            if num_big_stones > 0:
                big_stone_asset = self.gym.load_asset(
                    self.sim, stone_root, "stone_big.urdf", big_options)

            small_body_count = (self.gym.get_asset_rigid_body_count(stone_asset)
                                if stone_asset is not None else 0)
            big_body_count = (self.gym.get_asset_rigid_body_count(big_stone_asset)
                              if big_stone_asset is not None else 0)
            self.num_stone_bodies = small_body_count or big_body_count
            self.num_stone_bodies_total = (num_big_stones * big_body_count
                                           + num_small_stones * small_body_count)
        else:
            self.num_stone_bodies = 0
            self.num_stone_bodies_total = 0

        self._load_static_obstacle_assets()
        self.num_static_obstacle_actors = sum(len(spec["positions"]) for spec in self.static_obstacle_specs)
        self.num_static_obstacle_bodies = sum(spec["num_bodies"] * len(spec["positions"]) for spec in self.static_obstacle_specs)

        # Static PhysX actors cannot be moved after GPU simulation starts.  Sample
        # the whole stone/tree layout before actor creation so visual and collision
        # geometry are born at the same pose.  A new process/seed gives a new layout.
        random_at_creation = bool(
            getattr(getattr(self.cfg, "stone", None), "randomize_at_creation", False)
            or getattr(getattr(self.cfg, "static_obstacles", None),
                       "randomize_at_creation", False))
        self._creation_random_stone_positions = None
        self._creation_random_tree_positions = None
        if random_at_creation:
            source_cfg = (self.cfg.stone if num_big_stones > 0
                          else self.cfg.static_obstacles)
            x_range = tuple(getattr(source_cfg, "random_spawn_x", (12.0, 24.0)))
            y_range = tuple(getattr(source_cfg, "random_spawn_y", (0.0, 12.0)))
            min_sep = float(getattr(source_cfg, "random_min_separation", 1.4))
            seed = int(getattr(source_cfg, "random_seed", 0))
            positions = self._sample_separated_xy(
                num_big_stones + self.num_static_obstacle_actors,
                x_range, y_range, min_sep,
                rng=np.random.RandomState(seed + 100000))
            self._creation_random_stone_positions = positions[:num_big_stones]
            self._creation_random_tree_positions = positions[num_big_stones:]
            print("[legged_robot] 静态障碍在创建期随机化: seed=%d, 大石头=%d, 树=%d"
                  % (seed, num_big_stones, self.num_static_obstacle_actors))

        self.stone_handles = [[] for _ in range(self.num_envs)]
        self.static_obstacle_handles = [[] for _ in range(self.num_envs)]
        self.saved_stone_start_poses = [] 
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        hip_joint_names = [
            "FL_hip_joint",  # 左前髋关节
            "FR_hip_joint",  # 右前髋关节
            "RL_hip_joint",  # 左后髋关节
            "RR_hip_joint"   # 右后髋关节
        ]
        # 3. 根据关节名称查找对应的索引（即 hip_indices）
        self.hip_indices = [self.dof_names.index(name) for name in hip_joint_names]
        # 转换为张量（方便后续在GPU上计算）
        self.hip_indices = torch.tensor(self.hip_indices, device=self.device, dtype=torch.long)
        # 在 __init__ 中添加打印
        # 定义大腿关节（thigh）的名称和索引
        thigh_joint_names = [
            "FL_thigh_joint",  # 左前大腿
            "FR_thigh_joint",  # 右前大腿
            "RL_thigh_joint",  # 左后大腿
            "RR_thigh_joint"   # 右后大腿
        ]
        self.thigh_indices = [self.dof_names.index(name) for name in thigh_joint_names]
        self.thigh_indices = torch.tensor(self.thigh_indices, device=self.device, dtype=torch.long)

        # 定义小腿关节（calf）的名称和索引
        calf_joint_names = [
            "FL_calf_joint",   # 左前小腿
            "FR_calf_joint",   # 右前小腿
            "RL_calf_joint",   # 左后小腿
            "RR_calf_joint"    # 右后小腿
        ]
        self.calf_indices = [self.dof_names.index(name) for name in calf_joint_names]
        self.calf_indices = torch.tensor(self.calf_indices, device=self.device, dtype=torch.long)
        
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
            
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            # 这里 collision_group 设为 i，filter 设为 0 (0表示与所有组碰撞)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            if i == 0:
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                    
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            # === 新增：创建石头 Actor ===
            env_stone_poses = []
            if num_stones > 0:
                s_min = getattr(self.cfg.stone, "scale_min", 0.1)
                s_max = getattr(self.cfg.stone, "scale_max", 0.1)
                h_offset = getattr(self.cfg.stone, "spawn_height_offset", 0.3)
                spawn_x_min, spawn_x_max = self.cfg.stone.stone_spawn_x
                spawn_y_min, spawn_y_max = self.cfg.stone.stone_spawn_y
                big_scale = float(getattr(self.cfg.stone, "big_stone_scale", 1.0))
                big_positions = list(getattr(self.cfg.stone, "big_stone_positions", []) or [])
                if self._creation_random_stone_positions is not None:
                    big_positions = self._creation_random_stone_positions
                small_static = bool(self._stone_cfg_get("small_static", True))
                # 静态石头不会自由落体，直接贴地放置；动态时仍从 h_offset 抛下
                small_z_offset = float(self._stone_cfg_get("small_static_z_offset", 0.01)
                                       if small_static else h_offset)
                # Bind rubble generation to the scene seed. Otherwise paired
                # planner runs would silently receive different small stones.
                small_seed = int(self._stone_cfg_get("random_seed", 0)) + 200000 + i
                small_rng = np.random.RandomState(small_seed)

                # --- 1) 固定大石头（静态、带贴图）：actor 索引 1..num_big ---
                #     保持“大石头在前、小碎石在后”的既有 actor 顺序，
                #     root_states_reshaped 的切片与 pathplanner 的隐藏逻辑都依赖它。
                for k in range(min(num_big_stones, len(big_positions))):
                    rx, ry = big_positions[k]
                    terrain_z = self._get_terrain_height_at(rx, ry)
                    z_height = terrain_z + h_offset
                    stone_pose = gymapi.Transform()
                    stone_pose.p = gymapi.Vec3(rx, ry, z_height)
                    stone_pose.r = gymapi.Quat(0, 0, 0, 1)
                    stone_handle = self.gym.create_actor(
                        env_handle, big_stone_asset, stone_pose, f"stone_{k}", 0, 0, 0)
                    self.gym.set_actor_scale(env_handle, stone_handle, big_scale)
                    self.stone_handles[i].append(stone_handle)
                    env_stone_poses.append([rx, ry, z_height])

                # --- 2) 随机小碎石：actor 索引 num_big+1..num_stones ---
                for s in range(num_small_stones):
                    actor_id = num_big_stones + s
                    rx = small_rng.uniform(spawn_x_min, spawn_x_max)
                    ry = small_rng.uniform(spawn_y_min, spawn_y_max)
                    scale = self._quantize_stone_scale(
                        small_rng.uniform(s_min, s_max), "small")
                    terrain_z = self._get_terrain_height_at(rx, ry)
                    z_height = terrain_z + small_z_offset
                    stone_pose = gymapi.Transform()
                    stone_pose.p = gymapi.Vec3(rx, ry, z_height)
                    rand = small_rng.randn(4)
                    rand /= np.linalg.norm(rand)
                    stone_pose.r = gymapi.Quat(rand[0], rand[1], rand[2], rand[3])
                    stone_handle = self.gym.create_actor(
                        env_handle, stone_asset, stone_pose, f"stone_{actor_id}", 0, 0, 0)
                    self.gym.set_actor_scale(env_handle, stone_handle, scale)
                    self.stone_handles[i].append(stone_handle)
                    env_stone_poses.append([rx, ry, z_height])
            self.saved_stone_start_poses.append(env_stone_poses)
            self.static_obstacle_handles[i] = self._create_static_obstacles(env_handle, i)
            cam_cfg = getattr(self.cfg, "camera", None)
            if i == 0 and (cam_cfg is None or getattr(cam_cfg, "enable", True)):
                camera_props = gymapi.CameraProperties()
                camera_props.width = int(getattr(cam_cfg, "width", 1280))
                camera_props.height = int(getattr(cam_cfg, "height", 720))
                camera_props.horizontal_fov = float(getattr(cam_cfg, "horizontal_fov", 90.0))
                camera_props.enable_tensors = True
                # 记下 fov，供感知模块校验（BEV 反投影必须用同一个 fov）
                self.camera_horizontal_fov = camera_props.horizontal_fov
                self.camera_resolution = (camera_props.width, camera_props.height)

                cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
                # headless（graphics_device_id == -1）下 IsaacGym 建不出相机，返回 -1。
                # 若不拦下，后面 set_camera_location 会报
                # "could not find camera with handle -1"，get_camera_image_gpu_tensor
                # 还会打出 "*** Can't create empty tensor"。
                # 这里按“无相机”处理：train --headless 日志干净，
                # 而 pathplanner 读深度前已有 cam_handle is None 的守卫，不会崩。
                if cam_handle is None or int(cam_handle) < 0:
                    cam_handle = None
                    self.camera_horizontal_fov = None
                    self.camera_resolution = None
                    if i == 0:
                        print("[legged_robot] headless 下无法创建相机（cam_handle=-1），"
                              "已跳过；需要深度图请勿加 --headless。")

                if cam_handle is not None:
                    # 初始机位（运行时会被感知模块按 base 朝向重写）
                    _fwd = float(getattr(cam_cfg, "forward_offset", 0.3))
                    _look = float(getattr(cam_cfg, "lookat_forward_offset", 1.3))
                    _init_h = float(getattr(cam_cfg, "init_height", 0.6))
                    cam_pos = gymapi.Vec3(pos[0].item() + _fwd, pos[1].item(), _init_h)
                    cam_target = gymapi.Vec3(pos[0].item() + _look, pos[1].item(), _init_h)

                    self.gym.set_camera_location(
                        cam_handle,
                        env_handle,
                        cam_pos,
                        cam_target
                    )

                    # 只保存一个
                    self.cam_handle = cam_handle
                    self.cam_env_handle = env_handle

        # ================= Get depth tensor =================
        if self.cam_handle is not None:
            depth_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim,
                self.cam_env_handle,
                self.cam_handle,
                gymapi.IMAGE_DEPTH
            )
            self.camera_depth_tensor = gymtorch.wrap_tensor(depth_tensor)

            # rgb
            color_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim,
                self.cam_env_handle,
                self.cam_handle,
                gymapi.IMAGE_COLOR
            )
            self.camera_color_tensor = gymtorch.wrap_tensor(color_tensor)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

            
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        _term_cfg = getattr(self.cfg, "termination", None)
        _stuck_time_s = float(getattr(_term_cfg, "stuck_time_s",
                                      getattr(self.cfg.env, "stuck_time_s", 1.0))) \
            if _term_cfg is not None else float(getattr(self.cfg.env, "stuck_time_s", 1.0))
        self.stuck_steps_threshold = max(1, int(_stuck_time_s / self.dt))

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _init_base_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _get_base_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        # heights = (heights1 + heights2 + heights3) / 3

        base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

        return base_height
    
    def _get_feet_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.feet_pos[:, :, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = self.feet_pos[env_ids].clone()
        else:
            points = self.feet_pos.clone()

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        # heights = torch.min(heights1, heights2)
        # heights = torch.min(heights, heights3)
        heights = (heights1 + heights2 + heights3) / 3

        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        feet_height =  self.feet_pos[:, :, 2] - heights

        return feet_height

    #------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_joint_power(self):
        #Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error
    
    def _reward_symmetry(self):
        # Penalize FR nad RL not symmetry

        reward=torch.square(self.dof_pos[:, self.hip_indices[0]]-self.dof_pos[:, self.hip_indices[3]])
        reward+=torch.square(self.dof_pos[:, self.hip_indices[1]]-self.dof_pos[:, self.hip_indices[2]])
        reward+=torch.square(self.dof_pos[:, self.thigh_indices[0]]-self.dof_pos[:, self.thigh_indices[3]])
        reward+=torch.square(self.dof_pos[:, self.thigh_indices[1]]-self.dof_pos[:, self.thigh_indices[2]])
        reward+=torch.square(self.dof_pos[:, self.calf_indices[0]]-self.dof_pos[:, self.calf_indices[3]])
        reward+=torch.square(self.dof_pos[:, self.calf_indices[1]]-self.dof_pos[:, self.calf_indices[2]])
        return reward

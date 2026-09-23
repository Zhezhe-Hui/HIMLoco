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
import numpy as np
from isaacgym.torch_utils import quat_apply
from isaacgym import gymutil, gymapi
from legged_gym.envs.go2.pathplanner import SimplePlanner,Evaluator

# =====================================================================
#  所有开关（截图/实时显示/出图/评测参数）统一在 viz_config.py 配置
#  改 ACTIVE_PRESET 一行即可切换 off / viewer_only / paper / all
# =====================================================================
from legged_gym.envs.go2.viz_config import (
    NUM_TRIALS, RENDER, PLANNER_MODE, WAYPOINTS, EXPORT_POLICY, NUM_ENVS,
    NAV_EPISODE_LENGTH_S,
)
from legged_gym.envs.go2.viz_config import ensure_dirs as _ensure_viz_dirs, summary as _viz_summary
from legged_gym.envs.go2.viz_config import (
    EVAL_OVERRIDES, EVAL_TERRAIN_OVERRIDE_ENABLE, EVAL_TERRAIN_NUM_ROWS,
    EVAL_TERRAIN_NUM_COLS, EVAL_TERRAIN_CURRICULUM, EVAL_TERRAIN_MAX_INIT_LEVEL,
    EVAL_TERRAIN_MESH_TYPE, POLICY_LOAD_RUN, POLICY_CHECKPOINT,
)

# 启动时打印当前开关与场景（石头/树/地形），便于一眼确认跑的是哪套配置
_ensure_viz_dirs()
print("[viz_config] " + _viz_summary())

def play(args):
    EXP_CONFIG = dict(
        num_trials=NUM_TRIALS,
        render=RENDER,
        waypoints=WAYPOINTS,
        mode=PLANNER_MODE,
    )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.episode_length_s = float(NAV_EPISODE_LENGTH_S)
    # override some parameters for testing
    # 原来硬编码 min(cfg, 50)，50 个 env × 1000 石头会 PhysX OOM，改由 viz_config.NUM_ENVS 控制
    env_cfg.env.num_envs = NUM_ENVS
    # 评测覆盖项（噪声/域随机化/地形）全部走 viz_config，脚本里不再写死
    if EVAL_TERRAIN_OVERRIDE_ENABLE:
        env_cfg.terrain.num_rows = EVAL_TERRAIN_NUM_ROWS
        env_cfg.terrain.num_cols = EVAL_TERRAIN_NUM_COLS
        env_cfg.terrain.curriculum = EVAL_TERRAIN_CURRICULUM
        env_cfg.terrain.max_init_terrain_level = EVAL_TERRAIN_MAX_INIT_LEVEL
        env_cfg.terrain.mesh_type = EVAL_TERRAIN_MESH_TYPE
    for (section, key), value in EVAL_OVERRIDES:
        setattr(getattr(env_cfg, section), key, value)
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    # 显式指定加载哪个 run / 哪个 checkpoint，避免 helpers.get_load_path
    # 用“目录名字符串排序”选最新 run 而误加载冒烟测试留下的垃圾模型
    train_cfg.runner.load_run = POLICY_LOAD_RUN
    train_cfg.runner.checkpoint = POLICY_CHECKPOINT
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # 初始化Planner
    planner = SimplePlanner(
        waypoints=EXP_CONFIG["waypoints"],
    )

    max_steps = int(env.max_episode_length)
    evaluator = Evaluator(
        env=env,
        policy=policy,
        planner=planner,
        max_steps=max_steps,
        render=EXP_CONFIG["render"],
        mode=EXP_CONFIG["mode"]
    )
    evaluator.run(EXP_CONFIG["num_trials"])
if __name__ == '__main__':
    args = get_args()
    play(args)

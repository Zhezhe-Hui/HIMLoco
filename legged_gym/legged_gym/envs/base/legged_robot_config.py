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

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_one_step_observations = 45
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = 45 + 3 + 3 + 187 # additional: base_lin_vel, external_forces, scan_dots
        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
                                
    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m] 地形边缘空白区域
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

        # ---- 地形形状参数（原写死在 utils/terrain.py:make_terrain_by_id 里）----
        #: sequence_terrain 使用的固定难度（影响坡度/台阶高度/障碍高度）
        sequence_difficulty = 0.5
        #: difficulty -> 各尺寸的换算系数
        slope_scale = 0.4              # 坡度 = difficulty * slope_scale
        amplitude_min = 0.01           # 粗糙起伏 = amplitude_min + amplitude_scale*difficulty
        amplitude_scale = 0.07
        step_height_scale = 0.18       # 台阶高度 = step_height_scale * difficulty
        step_width = 0.50              # 台阶宽度 [m]
        discrete_obstacle_base = 0.05  # 离散障碍高度 = base + discrete_obstacle_scale*difficulty
        discrete_obstacle_scale = 0.1
        #: 各地形的平台尺寸 [m]
        platform_size_default = 3.0
        platform_size_rough_slope = 0.5
        #: 粗糙斜坡(id=1)的随机起伏
        rough_slope_min_height = -0.05
        rough_slope_max_height = 0.05
        rough_slope_step = 0.005
        rough_slope_downsample = 0.2
        #: 离散障碍(id=4)
        discrete_obstacle_height_fixed = 0.01
        discrete_rect_min_size = 2
        discrete_rect_max_size = 3
        discrete_num_rectangles = 50
        #: 离散高度(id=6)
        height_field_min = 0.0
        height_field_max = 0.08
        height_field_step = 0.005
        height_field_downsample = 0.2
        #: 地形边界墙（utils/terrain.py:add_all_walls）
        wall_enable = True
        wall_thickness_m = 1.0
        wall_height_m = 4.0
        wall_margin_m = 1.0        # 与出生点的安全距离
        #: sequence_terrain 列数不足时的兜底地形 id（5 = 平地）
        sequence_fallback_terrain_id = 5

    class commands:
        curriculum = True
        max_curriculum = 3.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_payload_mass = True
        payload_mass_range = [-1, 2]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_link_mass = False
        link_mass_range = [0.9, 1.1]
        
        randomize_friction = True
        friction_range = [0.2, 1.25]
        
        randomize_restitution = False
        restitution_range = [0., 1.0]
        
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        
        randomize_kp = True
        kp_range = [0.9, 1.1]
        
        randomize_kd = True
        kd_range = [0.9, 1.1]
        
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]
        
        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8
        
        push_robots = True
        push_interval_s = 16
        max_push_vel_xy = 1.

        delay = True

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = 0.09
        
    class static_obstacles:
        enable = False
        place_on_terrain = True
        fix_base_link = True
        disable_gravity = True
        collapse_fixed_joints = True
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        assets = []
        #: 同一个 URDF 被多个 spec 引用时复用已加载的 asset（避免重复 PhysX cooking）
        reuse_duplicate_assets = True
        #: 树的 collision mesh 面数极高时可开 vhacd 凸分解；实测默认参数更慢，慎用
        vhacd_enabled = None
        convex_decomposition_from_submeshes = None
        vhacd_params = None

    class camera:
        """env0 上的深度/RGB 相机传感器。高层 BEV 建图依赖它，
        ⚠ horizontal_fov 必须与感知模块用的视场角一致，否则深度图反投影会错位。
        """
        enable = True
        width = 1280
        height = 720
        horizontal_fov = 90.0
        #: 初始机位（仅第一帧生效，之后由感知模块按 base 朝向每步重写）
        forward_offset = 0.3
        lookat_forward_offset = 1.3
        init_height = 0.6
        #: 运行时相机相对 base 的安装高度（感知模块跟随用，见 viz_config.BEV_CAM_HEIGHT）
        height_offset = 0.25

    class spawn:
        """机器人出生点与初速度。

        ⚠ use_env_origins=False 时，所有 env 的机器人都被放到同一个固定世界坐标，
          这是导航评测可复现的前提（起点一致），但训练时会让所有机器人挤在同一点。
        """
        #: False = 用下面的固定坐标；True = 用课程/地形算出的 env_origins
        use_env_origins = False
        fixed_position = (6.0, 9.0, 0.4)
        #: use_env_origins=True 时，xy 方向的随机抖动幅度 [m]
        origin_xy_jitter = 1.0
        #: 固定世界坐标模式下，也可在每次 reset 从给定范围随机出生。
        randomize_each_reset = False
        random_x_range = (6.0, 6.0)
        random_y_range = (3.0, 9.0)
        #: 重置时给机器人的随机初速度范围（线性速度 + 角速度，共 6 维）
        #: 实现为 rand()-0.5 再乘该幅度，0 = 不给初速度
        init_velocity_range = 1.0

    class termination:
        """终止判定阈值。基类给出与原 main 分支一致的行为，具体数值由
        各机器人的 config（如 go2 -> viz_config）覆盖。
        """
        #: 碰撞终止：终止部位接触力超过该值即判摔倒
        enable_collision = True
        collision_force_threshold = 1.0
        #: 超时终止
        enable_timeout = True
        #: 悬崖/掉落终止：base 高度低于该值
        enable_cliff_fall = True
        cliff_fall_height = -0.5
        #: 卡住终止：命令速度够大但实际速度几乎为 0，持续 stuck_time_s 秒
        enable_stuck = True
        stuck_cmd_speed_min = 0.3
        stuck_actual_speed_max = 0.05
        stuck_time_s = 1.0

    class stone:
        """碎石/大石头场景。基类默认全关，go2 在自身 config 中覆盖。

        大小石头数量是**两个独立开关**：
          num_big_stones   固定位置的大石头（静态、带贴图）
          num_small_stones 随机撒布的小碎石
        num_stones 是二者之和，仅供底层创建循环使用，不要单独改。
        """
        enable = False
        num_big_stones = 0
        num_small_stones = 0
        num_stones = 0
        stone_spawn_x = (0.0, 1.0)
        stone_spawn_y = (0.0, 1.0)
        scale_min = 0.1
        scale_max = 0.3
        spawn_height_offset = 0.3
        big_stone_positions = []
        big_stone_indices = []
        big_stone_scale = 1.0
        density = 1000.0
        small_urdf = "resources/stone/stone.urdf"
        big_urdf = "resources/stone/stone_big.urdf"
        #: 小碎石是否静态化（fix_base_link + disable_gravity）
        small_static = True
        #: 静态化时石头中心离地形的高度（动态时改用 spawn_height_offset 抛下落体）
        small_static_z_offset = 0.01
        #: 碰撞体模式: "mesh" | "vhacd" | None(用 IsaacGym 默认)
        small_collision = "mesh"
        big_collision = "mesh"
        small_vhacd_params = {}
        big_vhacd_params = {}
        #: 把连续随机缩放量化成 N 档，复用 PhysX 碰撞几何缓存；0 = 不量化
        small_scale_steps = 0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'HIMOnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        num_steps_per_env = 100 # per iteration
        max_iterations = 200000 # number of policy updates

        # logging
        save_interval = 20 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR
class Go2RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_one_step_observations = 45
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = 45 + 3 + 3 + 187# additional: base_lin_vel, external_forces, scan_dots
        num_privileged_obs = num_one_step_privileged_obs * 1 
        episode_length_s = 30
        stuck_time_s = 1.0
             
    class static_obstacles(LeggedRobotCfg.static_obstacles):
        enable = True
        place_on_terrain = True
        use_env_origin = False
        assets = [
            {
                "name": "deadwood",
                "file": "{LEGGED_GYM_ROOT_DIR}/resources/tree/deadwood/deadwood.urdf",
                # HIMLOC uses global terrain coordinates, the same coordinate system as big_stone_positions.
                "positions": [[13.0, 3.0, 0.02]],
                "rpy": [1.57079632679, 0.0, 0.0],
                "scale": 0.5,
            },
            {
                "name": "oak_tree",
                "file": "{LEGGED_GYM_ROOT_DIR}/resources/tree/oak_tree/oak_tree.urdf",
                "positions": [[18.0, 2.0, 0.02]],
                "rpy": [1.57079632679, 0.0, 0.0],
                "scale": 0.3,
            },
            {
                "name": "snow_tree",
                "file": "{LEGGED_GYM_ROOT_DIR}/resources/tree/snow_tree/snow_tree.urdf",
                "positions": [[12.0, 9.0, 0.02]],
                "rpy": [0, 0.0, 0.0],
                "scale": 0.15,
            },
            {
                "name": "sno_tree",
                "file": "{LEGGED_GYM_ROOT_DIR}/resources/tree/snow_tree/snow_tree.urdf",
                "positions": [[20.0, 6.0, 0.02]],
                "rpy": [0, 0.0, 0.0],
                "scale": 0.15,
            },
        ]

    class stone:
        enable = True
        stone_spawn_x = (12.0, 36.0)    #x方向：整个地形的边长
        stone_spawn_y = (0.0, 12.0)
        num_stones = 1000
        scale_min = 0.1     #石头大小
        scale_max = 0.3 
        # ===== 新增：固定大石头 =====

        big_stone_scale = 2          # URDF 等比放大

        big_stone_positions = [
            # x = 8
            (8, 1),
            (8, 5),
            # (8, 7),
            (8, 9.4),
            
            # # x = 12
            (12, 2),
            # (12, 4),
            # (12, 5),
            # (12, 8),
            (12, 10),
            
            # x = 16
            (16, 1),
            (16, 5),
            (16, 7),
            (16, 9.4),
            
            # x = 20
            (20, 2),
            (20, 4),
            # (20, 5),
            (20, 8),
            (20, 10),
            
            # x = 24
            # (24, 1),
            # (24, 5),
            (24, 7),
            (24, 9.4),
        ]
        big_stone_indices = list(range(len(big_stone_positions))) # 第 0/1/2 个 stone 变大

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.10 # [m] 小砖块地形需要更细的 heightfield 分辨率
        vertical_scale = 0.005 # [m]
        border_size = 0 # [m]
        curriculum = False
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
        terrain_length = 12.
        terrain_width = 12.
        num_rows = 3  # number of terrain rows (levels)
        num_cols = 1# number of terrain cols (types)
        # 地形类型索引：
        # 0 = 光滑斜坡   1 = 粗糙斜坡   2 = 下楼梯   3 = 上楼梯
        # 4 = 离散障碍   5 = 平地       6 = 离散高度 7 = 砖块长条障碍
        terrain_sequence = [4,1,4,1,1,1]
        # 砖块长条障碍参数，单位都是米。高度建议先从 0.18~0.25 训练。
        brick_obstacle_height = 0.1
        brick_length = 1.0
        brick_width = 0.35
        num_bricks = 45
        brick_platform_size = 2.0
        brick_max_yaw = 0.35
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.37] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            # lin_vel_x = [0.5, 2.0] # min max [m/s]
            # lin_vel_y = [0., 0.]   # min max [m/s]
            # ang_vel_yaw = [0., 0.]    # min max [rad/s]
            # heading = [0, 0]
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 50.}  # [N*m/rad]
        stiffness = {'joint': 40.0}  # [N*m/rad]
        # damping = {'joint': 2}     # [N*m*s/rad]
        damping = {'joint': 1}
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        #新加的
        hip_reduction = 1.0

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        # penalize_contacts_on = ["thigh", "calf"]
        # terminate_after_contacts_on = ["base"]
        penalize_contacts_on = ["thigh", "calf", "base"] # 碰撞惩罚部位（大腿、小腿、机身碰撞地面会扣分）。
        terminate_after_contacts_on = ["base"]           # 机身（base）碰撞地面直接终止训练（视为摔倒）。
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -1.0
            foot_clearance = -0.01
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time =  0.0
            collision = -1.0
            feet_stumble = -0.0
            stand_still = -0.
            torques = -0.0
            dof_vel = -0.0
            # dof_pos_limits = -0.5
            dof_vel_limits = -0.0
            torque_limits = -0.0
            # hip_neutral = 0.3
            hip_pos = -0.2
            dof_error = -0.001
            # symmetry = -0.5

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.35
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.20


class Go2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01            # 熵系数（鼓励动作多样性，避免模型过早 “僵化”）。
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'

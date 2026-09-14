from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class BigDogRoughCfg( LeggedRobotCfg ):

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.50] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_back':   -0.,   # [rad]
            'left_hip_front':   0.,    # [rad]
            'right_hip_back':   0.,     # [rad]
            'right_hip_front': -0.,   # [rad]

            'left_thigh_back': 1.7,   # [rad]
            'left_thigh_front':  1.8,  # [rad]
            'right_thigh_back': 1.7,   # [rad]
            'right_thigh_front': 1.8,     # [rad]

            'left_calf_back':   -1.2,  # [rad]
            'left_calf_front':  -1.2,    # [rad]
            'right_calf_back':  -1.2,   # [rad]
            'right_calf_front': -1.2,   # [rad]

        }
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        #这里两个是PD控制的参数用通配的方式配置的，以后还是建议这么弄
        #stiffness = {'joint': 20.}  # [N*m/rad]
        #damping = {'joint': 0.5}     # [N*m*s/rad]
        stiffness = {
            'left_hip_back':    100.,'left_hip_front':   100.,'right_hip_front':  100.,'right_hip_back':   100.,
            'left_thigh_back':  100.,'left_thigh_front': 100.,'right_thigh_front':100.,'right_thigh_back': 100.,
            'left_calf_back':   100.,'left_calf_front':  100.,'right_calf_front': 100.,'right_calf_back':  100.,
        }
        damping = {
            'left_hip_back':    4,'left_hip_front':   4,'right_hip_front':  4,'right_hip_back':   4,
            'left_thigh_back':  4,'left_thigh_front': 4,'right_thigh_front':4,'right_thigh_back': 4,
            'left_calf_back':   4,'left_calf_front':  4,'right_calf_front': 4,'right_calf_back':  4,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 0.4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bigdog/urdf/bigdog.urdf'
        name = "bigdog"
        foot_name = "calf"
        penalize_contacts_on = ["thigh"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter理论上来说有可能发生自碰撞，所以要打开
        flip_visual_attachments = False

    class commands( LeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            # lin_vel_x = [0.5, 2.0] # min max [m/s]
            # lin_vel_y = [0., 0.]   # min max [m/s]
            # ang_vel_yaw = [0., 0.]    # min max [rad/s]
            # heading = [0, 0]
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1, 1]   # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s]
            heading = [-3.14, 3.14]
  
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.25 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
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
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10# number of terrain cols (types)
        # 地形类型索引：
        # 0 = 光滑斜坡   1 = 粗糙斜坡#   2 = 下楼梯    3 = 上楼梯    4 = 离散障碍#   5 = 平地   6 = 离散高度#
        terrain_sequence = [5,1,2,3,4,1,3,4]
        # trimesh only:
        slope_treshold = 2.0 # slopes above this threshold will be corrected to vertical surfaces

    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5
            lin_vel_z = -0.3
            ang_vel_xy = -0.2
            orientation = -0.3
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -1.0
            foot_clearance = -0.01
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time =  0.0
            collision = -10.0
            feet_stumble = -0.0
            stand_still = -0.
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = -0.5
            dof_vel_limits = -0.0
            torque_limits = -0.0
            # hip_neutral = 0.3
            hip_pos = -0.6
            dof_error = -0.001
            symmetry = -0.5
 
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.45
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.3

class BigDogRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.02
        learning_rate = 3e-4
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_bigdog'
        max_iterations = 500000  # 增加训练迭代次数
        save_interval = 50     # 每5000步保存模型
        log_interval = 100       # 每100步打印训练日志

  

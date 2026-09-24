from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import json
import hashlib
import time
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import numpy as np
from isaacgym.torch_utils import quat_apply
from isaacgym import gymutil, gymapi, gymtorch
import matplotlib.pyplot as plt
from legged_gym.envs.go2.bev_height_mapper import BEVMapper,FMMGradientController
from legged_gym.envs.go2.bev_height_mapper import plot_all_trials,plot_all_saved_trajectories
from legged_gym.envs.go2.rrt_planner import RRTController
from legged_gym.envs.go2.a_star_dwa_planner import AStarDWAController
from legged_gym.envs.go2.mppi_local_planner import (
    FootprintMPPIController,
    WorldObstacleMemory,
    choose_safer_turn_sign,
)
from legged_gym.envs.go2.simulated_vio import SimulatedVIO, wrap_angle

# =====================================================================
#  所有可视化 / 截图 / 出图开关已统一迁移到 viz_config.py
#  改开关请编辑: legged_gym/legged_gym/envs/go2/viz_config.py
#  （改 ACTIVE_PRESET 一行即可切换 off / viewer_only / paper / all）
# =====================================================================
from legged_gym.envs.go2.viz_config import *  # noqa: F401,F403
from legged_gym.envs.go2.viz_config import ensure_dirs as _ensure_viz_dirs, summary as _viz_summary

_ensure_viz_dirs()
print("[viz_config] " + _viz_summary())

class MemoryGuardExceeded(RuntimeError):
    """进程 RSS 超过 SAFETY_MEM_LIMIT_GB，主动中止评测以免拖死整机。"""


def _diagnose_termination(env, cmd_vx=None, cmd_wz=None):
    """把 reset_buf 的合并结果反推成可读的终止原因（碰撞/卡住/掉落/超时）。

    check_termination 只把四项 or 进 reset_buf，上层拿到 dones=True 无法区分。
    这里按同样的判据重新算一遍（只针对 env0），用于诊断成功率下降的真正原因。

    ⚠ cmd_vx / cmd_wz 必须由调用方【显式传入】（2026-09-17 修复的测量缺陷）：
      本函数在 env.step() 【返回之后】才被调用，而 step() 内部顺序是
        post_physics_step -> check_termination -> reset_idx(env_ids)
                                            -> _resample_commands(env_ids)  ← 无条件
      reset_idx 会把 commands 重采样成随机值（lin_vel_x/y ∈ [-1,1]，
      ‖·‖ 最大 1.41），所以此处直接读 env.commands 拿到的是【重置后的随机指令】，
      不是规划器当步真正下发的那个值。
      实测铁证：诊断打印“指令 1.16 / 1.19 / 1.25”，而 NAV_FORWARD_VX=1.0，
      规划器根本不可能下发这么大的值 —— 全部是重采样来的噪声，
      导致“卡住”判据长期在拿错误指令做诊断，误导了此前所有调参结论。
      （base_lin_vel 与 contact_forces 在 reset 前已算好并各自缓存，读它们仍有效。）
      不传这两个参数时退回旧行为（读 env.commands），仅用于离线复现。
    """
    try:
        import torch
        term_cfg = getattr(env.cfg, "termination", None)
        reasons = []
        # 碰撞：终止部位接触力超阈值
        thresh = float(getattr(term_cfg, "collision_force_threshold", 1.0)) \
            if term_cfg is not None else 1.0
        force = torch.norm(
            env.contact_forces[0, env.termination_contact_indices, :], dim=-1)
        if bool(torch.any(force > thresh)):
            reasons.append("碰撞(接触力%.1fN>%.1fN)" % (force.max().item(), thresh))
        # 掉落/悬崖
        if term_cfg is None or getattr(term_cfg, "enable_cliff_fall", True):
            limit = float(getattr(term_cfg, "cliff_fall_height", -0.5)) \
                if term_cfg is not None else -0.5
            if float(env.root_states[0, 2].item()) < limit:
                reasons.append("掉落(z=%.2f<%.2f)" % (env.root_states[0, 2].item(), limit))
        # 卡住
        if term_cfg is not None and getattr(term_cfg, "enable_stuck", True):
            if cmd_vx is None:
                cmd_sp = float(torch.norm(env.commands[0, :2]).item())
            else:
                # 用规划器当步真实下发的指令（只有 vx 参与 ‖·‖，wz 是角速度）
                cmd_sp = abs(float(cmd_vx))
            act_sp = float(torch.norm(env.base_lin_vel[0, :2]).item())
            # 角速度与朝向变化率：区分“真卡住”和“原地转向”（假卡住）。
            #   base_lin_vel 是【机体坐标系】线速度：机器狗原地打转时它≈0，
            #   但 yaw rate 很大、位置仍在变化，此时 stuck 判据会误杀。
            #   stuck_steps_threshold 只有 1s，一次 waypoint 急转就可能触发。
            yaw_rate = float(env.base_ang_vel[0, 2].item())
            if (cmd_sp > float(getattr(term_cfg, "stuck_cmd_speed_min", 0.3))
                    and act_sp < float(getattr(term_cfg, "stuck_actual_speed_max", 0.05))):
                reasons.append("卡住(指令%.2f实际%.3f yawrate%+.2f)" % (cmd_sp, act_sp, yaw_rate))
        # 姿态：摔倒的直接证据（roll/pitch 过大）
        quat = env.base_quat[0]
        w, x, y, z = quat[3].item(), quat[0].item(), quat[1].item(), quat[2].item()
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        if abs(roll) > 0.6 or abs(pitch) > 0.6:
            reasons.append("姿态失衡(roll=%.2f pitch=%.2f)" % (roll, pitch))
        return "、".join(reasons) if reasons else "未命中任一判据(可能是上一步累积的 stuck 计数)"
    except Exception as err:   # 诊断失败绝不该影响主流程
        return "诊断异常: %r" % (err,)


def _read_self_rss_gb():
    """读本进程 VmRSS（GB）。非 Linux 或读不到时返回 None（不触发看门狗）。"""
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        return None
    return None


class SimplePlanner:
    def __init__(self,
                 waypoints,
                 ):  
        self.waypoints = [np.array(wp[:2], dtype=np.float32) for wp in waypoints]
        self.reset()

    def reset(self):
        self.current_id = 0
        self.finished = False
        self.visited = [False] * len(self.waypoints)
        self.last_update_time = 0.0

    def set_waypoints(self, waypoints):
        self.waypoints = [np.array(wp[:2], dtype=np.float32) for wp in waypoints]
        self.reset()

class Evaluator:
    def __init__(self, env, policy, planner,
                 max_steps=2000,
                 render=True, mode='fmm'):
        self.env = env
        self.policy = policy
        self.planner = planner
        self.max_steps = max_steps
        self.render = render
        # viewer 逐帧重绘开关：关掉后 env.step() 内部的 render() 只轮询窗口事件，
        # 不再执行 step_graphics + draw_viewer（这是整机卡死的主因）。
        # 只在确实存在 viewer 时才改，避免 headless 下无谓写属性。
        if getattr(self.env, "viewer", None) is not None:
            self.env.enable_viewer_sync = bool(LIVE_VIEWER_SYNC)
        self._last_drawn_id = -1
        self.traj_xy_all = []     # 所有 trial 的轨迹
        self.goal_xy_all = []     # 所有 goal
        self.raw_path_all = []      # list of np.ndarray
        self.smooth_path_all = []  # list of np.ndarray

        self.stats = {
            "time_ms": [],
            "path_length": [],
            "smoothness": [],
            "progress": [],
            "success_count": 0,
            "total_trials": 0
        }
        # =====================================
        # 所有参数来自 viz_config.py 第十四节，勿在此写死数值
        self.bev = BEVMapper(
            cam_height=BEV_CAM_HEIGHT,
            height_range_thresh=BEV_HEIGHT_RANGE_THRESH,
            bev_x=BEV_X,
            bev_y=BEV_Y,
            bev_res=BEV_RES,
            min_points=BEV_MIN_POINTS,
            hfov_deg=BEV_HFOV_DEG,
            max_depth=BEV_MAX_DEPTH,
            v_start_ratio=BEV_V_START_RATIO,
            min_depth=BEV_MIN_DEPTH,
            enable_memory_fusion=BEV_ENABLE_MEMORY_FUSION,
            memory_max_age=BEV_MEMORY_MAX_AGE,
            # 跨格台阶判据：修复“正对垂直墙面漏检”（见 viz_config.BEV_ENABLE_STEP_JUDGE）
            enable_step_judge=BEV_ENABLE_STEP_JUDGE,
            step_thresh=BEV_STEP_THRESH,
            step_span=BEV_STEP_SPAN,
            # 相机几何标定结论（2026-09-16 实测）：横向符号 / 轴向深度针孔 / 俯角
            pitch_deg=BEV_CAM_PITCH_DEG,
            lateral_sign=BEV_LATERAL_SIGN,
            use_axial_geom=BEV_USE_AXIAL_GEOM,
            clip_invalid_eps=BEV_CLIP_INVALID_EPS,
            min_range=BEV_MIN_RANGE_M,
        )
        self.last_occ = None
        # 前向净空速度调制的限幅状态（上一个实际下发的 cmd_vx）。
        # 每个 trial 开始时重置回全速，避免把上一轮末尾的低速带进新一轮。
        self._last_cmd_vx = float(NAV_FORWARD_VX)
        # 便于诊断：记录本次 trial 的净空距离/vx 轨迹
        self.trial_clearance_log = []
        self.trial_narrow_steps = 0
        self.trial_min_path_clearance = float("inf")
        self.trial_memory_max_points = 0
        self.mode = mode
        self.mppi = None
        if self.mode in ('fmm', 'mppi'):
            self.fmm = FMMGradientController(
                bev_res=BEV_RES, bev_x=BEV_X, bev_y=BEV_Y,
                max_wz=FMM_MAX_WZ,
                yaw_k=FMM_YAW_K,
                lookahead_m=FMM_LOOKAHEAD_M,
                inflate_radius_m=FMM_INFLATE_RADIUS_M,
                goal_min_forward_m=FMM_GOAL_MIN_FORWARD_M,
                use_soft_cost=FMM_USE_SOFT_COST,
                hard_radius_m=FMM_HARD_RADIUS_M,
                soft_clearance_m=FMM_SOFT_CLEARANCE_M,
                soft_cost_weight=FMM_SOFT_COST_WEIGHT,
                soft_cost_power=FMM_SOFT_COST_POWER,
                # 转向决策锁存：压住对称鞍点处 wz 的逐步振荡（见 viz_config.FMM_TURN_*）
                hysteresis_thresh=(FMM_TURN_HYSTERESIS_THRESH
                                   if FMM_TURN_HYSTERESIS_ENABLE else 0.0),
                hysteresis_hold_steps=(int(FMM_TURN_HOLD_STEPS)
                                       if FMM_TURN_HYSTERESIS_ENABLE else 0),
                max_wz_accel=FMM_MAX_WZ_ACCEL,
                control_dt=float(self.env.dt),
                unreachable_turn_wz=FMM_UNREACHABLE_TURN_WZ,
                use_path_lookahead=FMM_USE_PATH_LOOKAHEAD,
                open_space_yaw_deadband=FMM_OPEN_SPACE_YAW_DEADBAND_RAD,
            )
            if self.mode == 'mppi':
                self.mppi = FootprintMPPIController(
                    bev_res=BEV_RES, bev_x=BEV_X, bev_y=BEV_Y,
                    horizon=MPPI_HORIZON,
                    num_samples=MPPI_NUM_SAMPLES,
                    rollout_dt=MPPI_ROLLOUT_DT,
                    max_vx=MPPI_MAX_VX,
                    min_vx=MPPI_MIN_VX,
                    max_wz=MPPI_MAX_WZ,
                    footprint_length_m=MPPI_FOOTPRINT_LENGTH_M,
                    footprint_width_m=MPPI_FOOTPRINT_WIDTH_M,
                    safety_margin_m=MPPI_FOOTPRINT_MARGIN_M,
                    temperature=MPPI_TEMPERATURE,
                    noise_vx=MPPI_NOISE_VX,
                    noise_wz=MPPI_NOISE_WZ,
                )
        elif self.mode == 'rrt':
            self.fmm = RRTController(
                bev_res=BEV_RES, bev_x=BEV_X, bev_y=BEV_Y,
                max_wz=RRT_MAX_WZ,
                yaw_k=RRT_YAW_K,
                lookahead_m=RRT_LOOKAHEAD_M,
                inflate_radius_m=RRT_INFLATE_RADIUS_M,
                goal_min_forward_m=FMM_GOAL_MIN_FORWARD_M,
            )
        elif self.mode == 'astar_dwa': # 新增
            self.fmm = AStarDWAController(
                bev_res=BEV_RES, bev_x=BEV_X, bev_y=BEV_Y,
                max_vx=DWA_MAX_VX, max_wz=DWA_MAX_WZ,
                max_acc_vx=DWA_MAX_ACC_VX, max_acc_wz=DWA_MAX_ACC_WZ,
                dt=DWA_DT, predict_time=DWA_PREDICT_TIME,
                inflate_radius_m=DWA_INFLATE_RADIUS_M,
            )
        else:
            raise ValueError("unknown planner mode: %r" % self.mode)
        self.temporal_memory = WorldObstacleMemory(
            bev_res=BEV_RES, bev_x=BEV_X, bev_y=BEV_Y,
            max_age_steps=TEMPORAL_MEMORY_MAX_AGE_STEPS,
            voxel_size_m=TEMPORAL_MEMORY_VOXEL_M,
            max_points=TEMPORAL_MEMORY_MAX_POINTS,
        )
        self._use_temporal_memory = bool(
            TEMPORAL_MEMORY_ENABLE and self.mode == 'mppi')
        self.vio = SimulatedVIO(
            dt=float(self.env.dt),
            position_sigma_m=VIO_POSITION_SIGMA_M,
            yaw_sigma_rad=VIO_YAW_SIGMA_RAD,
            position_drift_m_sqrt_s=VIO_POSITION_DRIFT_M_SQRT_S,
            yaw_drift_rad_sqrt_s=VIO_YAW_DRIFT_RAD_SQRT_S,
            latency_steps=VIO_LATENCY_STEPS,
            dropout_probability=VIO_DROPOUT_PROBABILITY,
            seed=SCENE_RANDOM_SEED_BASE + VIO_RANDOM_SEED_OFFSET,
        )
        # ==================== 加这三行 ====================
        self.save_trajectory_dir = SAVE_TRAJECTORY_DIR
        os.makedirs(self.save_trajectory_dir, exist_ok=True)
        self.trial_index = 0
        self.final_hold_seconds = NAV_FINAL_HOLD_SECONDS
        self._bev_vis_fig = None
        self._bev_vis_saved_count = 0
        self._camera_vis_fig = None
        self._camera_vis_axes = None
        self._camera_vis_rgb = None
        self._camera_vis_depth = None
        if SAVE_BEV_FMM_IMAGES:
            os.makedirs(SAVE_BEV_FMM_DIR, exist_ok=True)
            for subdir in ("height", "obstacles", "potential"):
                os.makedirs(os.path.join(SAVE_BEV_FMM_DIR, subdir), exist_ok=True)
        if SAVE_TRIAL_ROUTE_IMAGE:
            os.makedirs(SAVE_TRIAL_ROUTE_DIR, exist_ok=True)
        if SAVE_CAMERA_IMAGES:
            os.makedirs(SAVE_CAMERA_IMAGE_DIR, exist_ok=True)
        # ==================================================
    def run_single_trial(self, trial_id=0):
        print(f"\n===== Trial {trial_id} =====")
        if NAV_RANDOM_STRAIGHT_ROUTE or SCENE_RANDOMIZE_OBSTACLES_EACH_TRIAL:
            # 场景随机数只由 trial_id 决定，不受上一轮运行步数/算法内部随机消耗影响。
            self.env.navigation_trial_seed = int(SCENE_RANDOM_SEED_BASE) + int(trial_id)
        self.env.reset()
        # 随机直线走廊：终点与本轮实际出生点保持同一 y，不再经过旧折线航点。
        if NAV_RANDOM_STRAIGHT_ROUTE:
            start_x = float(self.env.root_states[0, 0].item())
            start_y = float(self.env.root_states[0, 1].item())
            self.planner.set_waypoints([(float(NAV_STRAIGHT_GOAL_X), start_y)])
            print("随机直线路线: start=(%.2f, %.2f), goal=(%.2f, %.2f)"
                  % (start_x, start_y, NAV_STRAIGHT_GOAL_X, start_y))
            print("出生点 trial seed: %d | 静态障碍 seed: %d"
                  % (int(self.env.navigation_trial_seed), int(SCENE_RANDOM_SEED_BASE)))
            self._print_random_obstacle_layout()
        else:
            self.planner.reset()
        self._update_live_follow_camera()
        self._bev_vis_saved_count = 0
        self.state_log = []   # 清空日志
        obs = self.env.get_observations()
        dones = [False]
        prev_base_pos = None
        prev_base_yaw = None
        self.traj_xy_cur = []
        self.trial_start_xy = None
        self.trial_fmm_world_segments = []

        self._last_cmd_vx = float(NAV_FORWARD_VX)   # 速度调制限幅状态复位
        self.trial_clearance_log = []
        self.trial_narrow_steps = 0
        self.trial_min_path_clearance = float("inf")
        self.trial_memory_max_points = 0
        initial_true_pose = (
            float(self.env.root_states[0, 0].item()),
            float(self.env.root_states[0, 1].item()),
            float(self._get_base_yaw(0)),
        )
        self.vio.reset(
            initial_true_pose,
            seed=int(SCENE_RANDOM_SEED_BASE) + int(trial_id) + VIO_RANDOM_SEED_OFFSET)
        self.trial_vio_position_errors = []
        self.trial_vio_yaw_errors = []
        self.trial_false_arrival_error = None
        self.trial_base_contact_steps = 0
        self.trial_side_contact_steps = 0
        self.trial_max_base_contact_force = 0.0
        # 停滞逃逸状态复位（见 _reset_stall_escape / viz_config.STALL_*）
        self._reset_stall_escape()
        # 转向锁存状态复位：不把上一轮的转向惯性带进新一轮
        if hasattr(self.fmm, "reset_hysteresis"):
            self.fmm.reset_hysteresis()
        self.temporal_memory.reset()
        if self.mppi is not None:
            self.mppi.reset()
        self._mppi_active = False
        self._cached_mppi_cmd = (float(NAV_FORWARD_VX), 0.0)
        self._cached_raw_path = []
        self._planner_reachable = True
        self.trial_step_times = []  # 必须初始化，否则 A* 报错
        self.trial_total_length = 0
        self.trial_real_distance = 0.0  # 这里加

        self.trial_step_smoothness = []  # 记录每一步规划的平滑度
        # ----------------
        for step in range(self.max_steps):
            # --- 内置内存看门狗：viewer 逐帧重绘在远程桌面下会持续泄漏内存，
            #     吃满物理内存后触发 swap 会把整台机器（含远程桌面）拖死。
            #     这里超阈值就主动抛错退出，见 viz_config.SAFETY_MEM_*。
            if (SAFETY_MEM_GUARD_ENABLE
                    and step % max(1, int(SAFETY_MEM_CHECK_INTERVAL)) == 0):
                rss_gb = _read_self_rss_gb()
                if rss_gb is not None and rss_gb > SAFETY_MEM_LIMIT_GB:
                    raise MemoryGuardExceeded(
                        "内存看门狗触发：RSS=%.1fGB 超过上限 %.1fGB（trial %d, step %d）。"
                        "已主动中止以免整机卡死。原因通常是 viewer 逐帧重绘泄漏："
                        "请改用 ACTIVE_PRESET=\"nav_only\"，或把 LIVE_VIEWER_SYNC 设为 False、"
                        "LIVE_DRAW_INTERVAL 调大。"
                        % (rss_gb, SAFETY_MEM_LIMIT_GB, trial_id, step))
            # --- Show depth camera ---
            actions = self.policy(obs)
            t = step * self.env.dt
            base_pos = self.env.root_states[0, 0:3].cpu().numpy()
            base_yaw = self._get_base_yaw(0)
            nav_pose = self.vio.update((base_pos[0], base_pos[1], base_yaw))
            nav_x, nav_y, nav_yaw = map(float, nav_pose)
            self.trial_vio_position_errors.append(float(np.hypot(
                nav_x - base_pos[0], nav_y - base_pos[1])))
            self.trial_vio_yaw_errors.append(abs(wrap_angle(nav_yaw - base_yaw)))
   
            # ===== A. 时间融合（在 update BEV 之前）=====
            if BEV_ENABLE_MEMORY_FUSION and prev_base_pos is not None:
                dx = base_pos[0] - prev_base_pos[0]
                dy = base_pos[1] - prev_base_pos[1]
                self.bev.shift_with_motion(dx, dy, prev_base_yaw)
            base_xy = base_pos[:2].copy()
            if self.trial_start_xy is None:
                self.trial_start_xy = base_xy.copy()
            self.traj_xy_cur.append(base_xy)
            # ========== 新增：每一步真实行走距离 ==========
            # step_dist 必须先给默认值：step 0 时 prev_base_pos 还是 None，
            # 否则下面的停滞检测会读到未定义变量。
            step_dist = 0.0
            if prev_base_pos is not None:
                dx = base_pos[0] - prev_base_pos[0]
                dy = base_pos[1] - prev_base_pos[1]
                step_dist = float(np.hypot(dx, dy))
                self.trial_real_distance += step_dist
            # ==============================================
            prev_base_pos = base_pos.copy()
            prev_base_yaw = base_yaw

            # ===== 4. 更新 BEV =====
            fresh_bev = self._update_camera_and_bev(step)
            # ===== 5. 唯一的 wz：BEV + goal =====
            occ_bev_raw = self.bev.get_obstacle_map()
            # 障碍白名单：只保留【actor 障碍】（大石头/树/后续新增的动态障碍），
            # 地形边界墙与地形凸起一律不参与避障（见 viz_config.BEV_OBSTACLE_ACTORS_ONLY）。
            if BEV_OBSTACLE_ACTORS_ONLY:
                occ_bev_now = self._mask_occ_to_actors(
                    occ_bev_raw, base_pos, base_yaw)
            else:
                occ_bev_now = occ_bev_raw
            if self._use_temporal_memory:
                pose_xy_yaw = (nav_x, nav_y, nav_yaw)
                if fresh_bev:
                    observed_bev = self.bev.count >= int(BEV_MIN_POINTS)
                    self.temporal_memory.update(
                        occ_bev_now, pose_xy_yaw, step,
                        observed_local=observed_bev)
                occ_bev = self.temporal_memory.project(pose_xy_yaw)
                self.trial_memory_max_points = max(
                    self.trial_memory_max_points,
                    self.temporal_memory.point_count)
            else:
                occ_bev = occ_bev_now

            # 3) 计算 waypoint 在机器人坐标（goal_xy_bev）
            wp = self.planner.waypoints[self.planner.current_id]
            dx_w = wp[0] - nav_x
            dy_w = wp[1] - nav_y
            c = np.cos(-nav_yaw); s = np.sin(-nav_yaw)
            dx_r = c * dx_w - s * dy_w   # forward
            dy_r = s * dx_w + c * dy_w   # left

            # 4) FMM：重规划周期独立配置。当前按用户观感恢复到 50Hz；深度图
            # 仍以 12.5Hz 更新，但机器人位姿与局部目标每个控制步都会变化。
            replan_interval = max(1, int(FMM_REPLAN_INTERVAL_STEPS))
            plan_due = step == 0 or step % replan_interval == 0
            if plan_due:
                self.fmm.set_goal((dx_r, dy_r))
                start_plan_t = time.time()
                self.fmm.update(occ_bev)
                self.trial_step_times.append((time.time() - start_plan_t) * 1000)
                self._cached_raw_path = self.fmm.extract_path()
                if hasattr(self.fmm, "is_reachable"):
                    self._planner_reachable = bool(self.fmm.is_reachable())
            raw_path = self._cached_raw_path
            self._visualize_bev_fmm(step, trial_id, occ_bev, raw_path)

            # ===== 2. FMM路径（robot → world）=====
            raw_xy = []
            cos_yaw = np.cos(nav_yaw)
            sin_yaw = np.sin(nav_yaw)
            for r, col in raw_path:
                # BEV → robot frame
                x = r * self.fmm.bev_res
                y = (col - self.fmm.W // 2) * self.fmm.bev_res
                # robot → world
                wx = nav_x + cos_yaw * x - sin_yaw * y
                wy = nav_y + sin_yaw * x + cos_yaw * y
                raw_xy.append([wx, wy])
            raw_xy = np.array(raw_xy)
            self._record_trial_fmm_segment(step, raw_xy)

            # ===== 4. 保存 =====
            if step == 0: 
                if raw_xy is not None and len(raw_xy) > 0:
                    self.raw_path_all.append(raw_xy)

            wz = self.fmm.compute_wz()
            planner_vx = None
            if self.mppi is not None:
                obstacle_active = bool(np.any(occ_bev))
                if obstacle_active:
                    if not self._mppi_active:
                        self.mppi.reset()
                        self._mppi_active = True
                    mppi_due = (step == 0 or step % max(
                        1, int(MPPI_REPLAN_INTERVAL_STEPS)) == 0)
                    if mppi_due:
                        local_path_xy = np.asarray([
                            (r * self.fmm.bev_res,
                             (col - self.fmm.W // 2) * self.fmm.bev_res)
                            for r, col in raw_path
                        ], dtype=np.float64)
                        start_mppi_t = time.time()
                        self._cached_mppi_cmd = self.mppi.command(
                            occ_bev, local_path_xy, goal_xy=(dx_r, dy_r))
                        mppi_ms = (time.time() - start_mppi_t) * 1000.0
                        if self.trial_step_times:
                            self.trial_step_times[-1] += mppi_ms
                    planner_vx, wz = self._cached_mppi_cmd
                else:
                    # MPPI只负责有障碍的局部轨迹优化；空旷区沿用连续目标控制，
                    # 避免采样噪声在24m长直线中累积成无意义的额外路程。
                    self._mppi_active = False
                    self._cached_mppi_cmd = (float(NAV_FORWARD_VX), float(wz))
            narrow_passage = False
            if (NARROW_PASSAGE_ENABLE and self._planner_reachable
                    and hasattr(self.fmm, "path_min_clearance")):
                path_clearance = self.fmm.path_min_clearance(
                    NARROW_PASSAGE_LOOKAHEAD_M)
                self.trial_min_path_clearance = min(
                    self.trial_min_path_clearance, float(path_clearance))
                narrow_passage = path_clearance < float(NARROW_PASSAGE_CLEARANCE_M)
                if narrow_passage:
                    self.trial_narrow_steps += 1
                    wz = float(np.clip(
                        wz, -NARROW_PASSAGE_MAX_WZ, NARROW_PASSAGE_MAX_WZ))
            if NAV_DEBUG_START_STEPS > 0 and step < int(NAV_DEBUG_START_STEPS):
                lookahead_point = raw_path[min(len(raw_path) - 1,
                                               max(0, int(FMM_LOOKAHEAD_M / BEV_RES)))] \
                    if raw_path else None
                print("[start-debug] step=%d yaw=%+.3f goal_robot=(%+.2f,%+.2f) "
                      "occ=%d reachable=%s lookahead=%s wz=%+.3f"
                      % (step, nav_yaw, dx_r, dy_r, int(np.count_nonzero(occ_bev)),
                         self._planner_reachable, lookahead_point, wz))

            # ===== 轨迹优化（仅用于可视化 / 分析 / 未来 MPC）=====
            # smooth_path = self.fmm.optimize_fmm_path(
            #     waypoints=self.planner.waypoints
            # )
            # --- 增加以下记录逻辑 ---
            current_smooth = self._calculate_smoothness(raw_path)
            # 只有当路径有效（不是兜底值或 0）时才加入统计
            if (plan_due
                    and METRIC_SMOOTH_VALID_MIN < current_smooth < METRIC_SMOOTH_VALID_MAX):
                self.trial_step_smoothness.append(current_smooth)
            # ===== waypoint 切换 =====
            # 越过中间航点平面即前进（GPS 导航标准行为）：
            #   实测（blocker 场景）机器狗绕过石头后会【冲过】航点（如 wp1=(11,8)
            #   被冲到 (12.2,7.6)），此时 goal 在机器人坐标系的前向分量为负，
            #   set_goal 把它钳到 (0.1, 横向) => 纯横向目标 => wz 饱和原地打转，
            #   而 is_goal_reached 的 0.25m 阈值永远不满足 => 永久卡死。
            #   dx_r < -margin 表示航点已在身后 margin 米，直接判为通过。
            n_wp = len(self.planner.waypoints)
            is_final_goal = self.planner.current_id >= n_wp - 1
            crossed = (NAV_ADVANCE_ON_CROSS and not is_final_goal
                       and dx_r < -float(NAV_CROSS_MARGIN_M))
            # 中间航点用更宽的到达阈值：实测（blocker 场景 wz 日志）goal 落在
            # 最小转弯圆（v/wz_max ≈ 1.0m）内时，相对方位被锁在侧向 90°，
            # wz 饱和 ±1.0 连续上百步、机器狗全速跑圈（纯追踪极限环），
            # 0.3m 阈值永远不满足。中间航点只是途经点，1.0m 内即视为通过；
            # 最终目标点仍用 NAV_GOAL_REACH_THRESH_M 保精度。
            thresh = (NAV_MID_GOAL_REACH_THRESH_M
                      if self.planner.current_id < n_wp - 1
                      else NAV_GOAL_REACH_THRESH_M)
            # 规划器按 VIO 估计位姿判断到达；真值距离只用于独立评测，不反馈控制。
            # FMM 场在双层代价模式下是加权 travel time，不能代替几何距离。
            goal_distance_m = float(np.hypot(dx_w, dy_w))
            true_goal_distance_m = float(np.hypot(
                wp[0] - base_pos[0], wp[1] - base_pos[1]))
            if goal_distance_m <= float(thresh) or crossed:
                if is_final_goal:
                    print("最终目标估计/真值误差: %.3f / %.3f m (阈值 %.3f m)"
                          % (goal_distance_m, true_goal_distance_m,
                             NAV_GOAL_REACH_THRESH_M))
                    if true_goal_distance_m > float(NAV_GOAL_REACH_THRESH_M):
                        self.trial_false_arrival_error = true_goal_distance_m
                if raw_path is not None and len(raw_path) > 1:
                    # 原来写死 * 0.05，改 bev_res 后路径长度会算错，这里跟着 BEV_RES 走
                    pts = np.array(raw_path) * BEV_RES
                    self.trial_total_length += np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
                # ===== 记录当前 goal（世界坐标）=====
                self.goal_xy_all.append(
                    self.planner.waypoints[self.planner.current_id].copy()
                )
                self.planner.visited[self.planner.current_id] = True
                self.planner.current_id += 1

                if self.planner.current_id >= len(self.planner.waypoints):
                    self.planner.finished = True

            # ===== 4. 统一记录出口：无论是完成还是环境崩溃 =====
            if self.planner.finished or step == self.max_steps - 1:
                
                # 计算本轮全程的平均平滑度，而不是最后一帧的
                if self.trial_step_smoothness:
                    avg_smooth = np.mean(self.trial_step_smoothness)
                else:
                    avg_smooth = METRIC_SMOOTH_FALLBACK  # 失败且无有效路径时
                
                self.stats["smoothness"].append(avg_smooth)
                self.stats["time_ms"].append(np.mean(self.trial_step_times) if self.trial_step_times else 0)
                self.stats["path_length"].append(self.trial_total_length)
                self.stats["progress"].append(self.planner.current_id / len(self.planner.waypoints))

                elapsed_time = step * self.env.dt
                # self.traj_xy_all.append(np.array(self.traj_xy_cur))
                # 保存本次轨迹到文件
                if SAVE_TRAJECTORY_NPY:
                    self.trial_index += 1
                    traj = np.array(self.traj_xy_cur)
                    save_path = os.path.join(self.save_trajectory_dir, f"traj_{self.trial_index:02d}.npy")
                    np.save(save_path, traj)
                    print(f"✅ 轨迹已保存到: {save_path}")
                if self.planner.finished:
                    self._save_trial_route_plot(trial_id, "success")
                    self._show_completed_route()
                    if self.trial_false_arrival_error is not None:
                        return self._handle_fail(
                            "VIO false arrival (true error %.3f m)"
                            % self.trial_false_arrival_error,
                            elapsed_time)
                    self.stats["success_count"] += 1
                    return self._handle_success(elapsed_time)
                else:
                    reason = "Environment Termination" if dones[0] else "Timeout"
                    status = reason.lower().replace(" ", "_")
                    self._save_trial_route_plot(trial_id, status)
                    return self._handle_fail(reason, elapsed_time)

            # 下发动作：wz 由高层规划器给出；vx 原本是【恒定的】NAV_FORWARD_VX，
            # 现在按前向净空距离调制（见 viz_config.SPEED_MOD_*）。
            # 这是“开了避障成功率反而下降”的修复：上层只转向不减速时，
            # 机器狗会以 1.0m/s 全速撞上静态大石头（实测接触力 985N）而判摔。
            # ⚠ 转向与刹车【用不同的障碍图】（2026-09-17 实测定标）：
            #   · 转向 occ_bev：白名单图，只绕 actor 障碍（用户语义）。
            #   · 刹车 occ_bev_raw：原始几何图，含边界墙与地形起伏。
            #   为什么分开：转向问"要不要绕开它"（地形起伏四足就该踩过去，不绕），
            #   刹车问"脚下这段地面要不要减速"（边界墙/陡坎仍应触发降速）。
            #   白名单把 no_obstacles 障碍图归零后 clearance 恒为 3.0，
            #   刹车会完全失去边界墙这类真障碍的降速信号，故单独喂原始图。
            #   ⚠ 注：此处曾误判为"0.1m 地形台阶导致打滑卡死"。离线地形真值
            #     与停滞取证探针已证伪：全域单格坡度中位 0.0%、p99 仅 5.5%，
            #     离散障碍地形(id=4)实际用 discrete_obstacle_height_fixed=0.01m，
            #     地面基本是平的。真正的死锁机制见 _update_stall_escape()。
            fwd_clearance = self._forward_clearance(
                occ_bev_raw if BEV_CLEARANCE_USE_RAW_MAP else occ_bev)
            cmd_vx = self._modulate_vx(
                fwd_clearance, self.env.dt, wz,
                goal_distance=(goal_distance_m if is_final_goal else None),
                speed_cap=(NARROW_PASSAGE_MAX_VX if narrow_passage else None))
            if planner_vx is not None:
                cmd_vx = min(float(cmd_vx), float(planner_vx))

            if FMM_STOP_IF_UNREACHABLE and not self._planner_reachable:
                cmd_vx = 0.0

            # ===== 停滞逃逸：原地踏步死锁时接管指令（见 _update_stall_escape）=====
            if FMM_STOP_IF_UNREACHABLE and not self._planner_reachable:
                # 这是规划器主动停车，不是底层卡住；不要触发盲目倒退逃逸。
                self._stall_travel_buf = []
                in_escape = False
            else:
                in_escape = self._update_stall_escape(
                    step_dist, occ_bev=occ_bev, preferred_wz=wz)
            if in_escape and STALL_OVERRIDE_PLANNER:
                esc = self._stall_escape_cmd()
                if esc is not None:
                    cmd_vx, wz = esc

            self.trial_clearance_log.append((fwd_clearance, cmd_vx, float(wz)))
            self.env.commands[:] = 0.0
            self.env.commands[0, 0] = cmd_vx
            self.env.commands[0, 2] = wz
            self._update_live_follow_camera()
            obs, _, _, dones, infos, _, _ = self.env.step(actions)
            self._update_contact_metrics()
            # --- IsaacGym viewer 实时叠加可视化 ---
            # 节流：LIVE_DRAW_INTERVAL>1 时按间隔重画，省掉大量 add_lines/clear_lines
            if (LIVE_DRAW_VIEWER_OVERLAY and self.render
                    and self.env.viewer is not None
                    and step % max(1, int(LIVE_DRAW_INTERVAL)) == 0):
                self._draw_current_visualization()

                # 每当 current_id 变化时才记录，提高性能
                if self.planner.current_id != self._last_drawn_id:
                    self._last_drawn_id = self.planner.current_id

            # BEV/FMM 图片保存和实时窗口由文件开头的 SAVE_* / LIVE_* 开关统一控制

            # 数据记录
            real_vx = self.env.base_lin_vel[0, 0].detach().item()
            # 12 个关节的瞬时扭矩（力矩）值 平方和
            torque = self.env.torques[0].detach().cpu().numpy()
            torque_cost = np.sum(torque ** 2)
            roll, pitch, _ = self._quat_to_rpy(self.env.base_quat[0])
            body_wx = float(self.env.base_ang_vel[0, 0].item())
            body_wy = float(self.env.base_ang_vel[0, 1].item())
            self.state_log.append({
                # 记录【规划器真实下发】的 cmd_vx，不回读 env.commands：
                # 终止那一步 env.commands 已被 reset_idx 重采样成随机值（见
                # _diagnose_termination 注释），回读会让速度跟踪曲线混入噪声。
                "cmd_vx": float(cmd_vx),
                "cmd_wz": float(wz),
                "real_vx": real_vx,
                "torque_cost": torque_cost,
                "roll": float(roll),
                "pitch": float(pitch),
                "body_wx": body_wx,
                "body_wy": body_wy,
            })

            # 失败：环境终止
            if dones[0]:
                # 细分终止原因：check_termination 把 collision|timeout|cliff|stuck 合并成
                # 一个 reset_buf，光看 dones 无法区分是撞了还是卡住了还是掉了。
                # 诊断“开避障后成功率反而下降”这类问题必须知道到底哪一种。
                # ⚠ 必须显式传当步真实下发的 cmd_vx：env.step() 内部 reset_idx 已把
                #   env.commands 重采样成随机值，此时再读它拿到的是噪声（见函数注释）。
                #   cmd_vx 是本函数上文刚算出并下发的那个值，仍在局部作用域内。
                term_reason = _diagnose_termination(self.env, cmd_vx=cmd_vx, cmd_wz=wz)
                print(f"🔍 终止原因细分: {term_reason} (step {step})")
                elapsed_time = step * self.env.dt
                # self.traj_xy_all.append(np.array(self.traj_xy_cur))
                # 保存本次轨迹到文件
                if SAVE_TRAJECTORY_NPY:
                    self.trial_index += 20
                    traj = np.array(self.traj_xy_cur)
                    save_path = os.path.join(self.save_trajectory_dir, f"traj_{self.trial_index:02d}.npy")
                    np.save(save_path, traj)
                    print(f"✅ 轨迹已保存到: {save_path}")
                self._save_trial_route_plot(trial_id, "environment_termination")
                return self._handle_fail(term_reason, elapsed_time)

    def _print_random_obstacle_layout(self):
        """Print the current trial layout so failed cases can be reproduced and inspected."""
        if not SCENE_RANDOMIZE_OBSTACLES_EACH_TRIAL:
            return
        metadata = self._obstacle_layout_metadata()
        groups = [
            "大石头=%d" % metadata["big_stone_count"],
            "小碎石=%d" % metadata["small_stone_count"],
            "树=%d" % metadata["tree_count"],
            "hash=%s" % metadata["obstacle_layout_hash"],
        ]
        stones = getattr(self.env, "stone_root_states", None)
        if stones is not None and stones.shape[1] > 0:
            xy = stones[0, :min(5, stones.shape[1]), :2].detach().cpu().numpy()
            groups.append("stones前5=" + str(np.round(xy, 2).tolist()))
        trees = getattr(self.env, "static_obstacle_root_states", None)
        if trees is not None and trees.shape[1] > 0:
            xy = trees[0, :min(5, trees.shape[1]), :2].detach().cpu().numpy()
            groups.append("trees前5=" + str(np.round(xy, 2).tolist()))
        if groups:
            print("随机障碍布局: " + " | ".join(groups))

    def _obstacle_layout_metadata(self):
        """Return compact, stable metadata for paired-scene verification."""
        digest = hashlib.sha256()
        for name in ("stone_root_states", "static_obstacle_root_states"):
            states = getattr(self.env, name, None)
            if states is None or states.shape[1] == 0:
                values = np.empty((0, 3), dtype="<f4")
            else:
                values = states[0, :, :3].detach().cpu().numpy()
                values = np.round(values, 4).astype("<f4", copy=False)
            digest.update(name.encode("ascii"))
            digest.update(values.tobytes())
        return {
            "big_stone_count": int(SCENE_NUM_BIG_STONES),
            "small_stone_count": int(SCENE_NUM_SMALL_STONES_TOTAL),
            "tree_count": int(getattr(self.env, "num_static_obstacle_actors", 0)),
            "obstacle_layout_hash": digest.hexdigest()[:16],
        }

    def _reset_stall_escape(self):
        """复位停滞逃逸状态机（每个 trial 开始时调用）。"""
        self._stall_travel_buf = []      # 最近 STALL_WINDOW_STEPS 步的实际位移
        self._stall_phase = "idle"       # idle / reverse / turn
        self._stall_phase_left = 0       # 当前阶段还剩多少步
        self._stall_escape_count = 0     # 本 trial 已逃逸次数
        self._stall_turn_sign = 1.0      # 逃逸转向方向（交替用）

    def _update_stall_escape(self, step_dist, occ_bev=None, preferred_wz=0.0):
        """更新停滞检测状态机，返回本步是否处于逃逸中（True 则上层指令被覆盖）。

        为什么需要（2026-09-17 停滞取证探针实测，/tmp/pitch/logs/stall_probe.pkl）：
          机器狗会在粗糙斜坡的坑洼里【原地踏步死锁】：位置纹丝不动
          （(16.16,5.02) 持续 50+ 步），实际 vx 在 ±0.1 间抖动、yawrate 不跟踪 wz，
          而 BEV 障碍图全零 -> 上层认为"前方通畅" -> 永远重复同一条指令 -> 死锁到
          timeout。上层缺的不是感知，是【脱困动作】。
          （此前偶发的高成功率来自底层随机指令重采样 bug 的隐式逃逸，见
           viz_config.EVAL_COMMAND_RESAMPLING_TIME 注释。）
        做法：检测"窗口内实际路程 < 阈值" -> 依次执行 倒退 -> 原地转 的标准四足脱困。
        停滞窗口(30步=0.6s)必须短于环境自带的 TERM_STUCK_TIME_S(1.0s)，
        否则环境先判失败、轮不到逃逸。
        """
        if not STALL_ESCAPE_ENABLE:
            return False

        # --- 1. 逃逸动作进行中：直接推进阶段计时，不再检测新停滞 ---
        if self._stall_phase != "idle":
            self._stall_phase_left -= 1
            if self._stall_phase_left <= 0:
                if self._stall_phase == "reverse":
                    self._stall_phase = "turn"
                    self._stall_phase_left = int(STALL_TURN_STEPS)
                else:
                    self._stall_phase = "idle"
                    self._stall_travel_buf = []   # 逃逸后重新观察
            return True

        # --- 2. 检测停滞：维护滑动窗口内的实际路程 ---
        buf = self._stall_travel_buf
        buf.append(float(step_dist))
        win = int(STALL_WINDOW_STEPS)
        if len(buf) > win:
            del buf[0:len(buf) - win]
        if len(buf) < win:
            return False
        travel = float(sum(buf))
        if travel >= float(STALL_MIN_TRAVEL_M):
            return False

        # --- 3. 确认停滞：触发逃逸 ---
        if self._stall_escape_count >= int(STALL_MAX_ESCAPES):
            return False        # 放弃逃逸，让环境按原逻辑判 timeout/卡住
        self._stall_escape_count += 1
        if STALL_TURN_USE_BEV and occ_bev is not None:
            self._stall_turn_sign = float(choose_safer_turn_sign(
                occ_bev, preferred_wz=preferred_wz))
        elif STALL_TURN_ALTERNATE:
            self._stall_turn_sign = -self._stall_turn_sign
        self._stall_phase = "reverse"
        self._stall_phase_left = int(STALL_REVERSE_STEPS)
        if METRICS_PRINT_STATS:
            print("🔁 检测到停滞(%.0f步内仅走 %.3fm)，触发逃逸 #%d：倒退%d步+转向%d步(方向%+.0f)"
                  % (win, travel, self._stall_escape_count,
                     int(STALL_REVERSE_STEPS), int(STALL_TURN_STEPS), self._stall_turn_sign))
        return True

    def _stall_escape_cmd(self):
        """返回逃逸阶段的 (vx, wz)；倒退与原地转两段。"""
        if self._stall_phase == "reverse":
            return float(STALL_REVERSE_VX), 0.0
        if self._stall_phase == "turn":
            return 0.0, abs(float(STALL_TURN_WZ)) * float(self._stall_turn_sign)
        return None

    def _set_trial_follow_camera(self):
        base_pos = self.env.root_states[0, 0:3].detach().cpu().numpy()
        base_yaw = self._get_base_yaw(0)

        forward = np.array([np.cos(base_yaw), np.sin(base_yaw), 0.0], dtype=np.float32)
        right = np.array([np.sin(base_yaw), -np.cos(base_yaw), 0.0], dtype=np.float32)

        camera_position = (
            base_pos
            - TRIAL_FOLLOW_CAMERA_BACK_M * forward
            + TRIAL_FOLLOW_CAMERA_RIGHT_M * right
            + np.array([0.0, 0.0, TRIAL_FOLLOW_CAMERA_HEIGHT_M], dtype=np.float32)
        )
        camera_target = (
            base_pos
            + TRIAL_FOLLOW_CAMERA_LOOKAHEAD_M * forward
            + np.array([0.0, 0.0, TRIAL_FOLLOW_CAMERA_LOOKAT_HEIGHT_M], dtype=np.float32)
        )
        self.env.set_camera(camera_position, camera_target)

    def _update_live_follow_camera(self):
        if not LIVE_FOLLOW_VIEWER_CAMERA:
            return
        if not self.render or self.env.viewer is None:
            return
        self._set_trial_follow_camera()

    def _path_rc_to_plot_xy(self, path_rc):
        if path_rc is None or len(path_rc) == 0:
            return np.empty((0, 2), dtype=np.float32)
        path = np.asarray(path_rc, dtype=np.float32)
        x_m = path[:, 0] * self.fmm.bev_res
        y_m = (path[:, 1] - self.fmm.W // 2) * self.fmm.bev_res
        return np.column_stack([y_m, x_m])

    def _smooth_plot_path(self, xy, samples=160, window=9):
        if xy.shape[0] < 4 or not BEV_FMM_SMOOTH_PATH:
            return xy

        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        dist = np.concatenate([[0.0], np.cumsum(seg)])
        if dist[-1] < 1e-6:
            return xy

        sample_dist = np.linspace(0.0, dist[-1], samples)
        xs = np.interp(sample_dist, dist, xy[:, 0])
        ys = np.interp(sample_dist, dist, xy[:, 1])

        window = max(3, int(window))
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window, dtype=np.float32) / float(window)
        pad = window // 2
        xs = np.convolve(np.pad(xs, pad, mode="edge"), kernel, mode="valid")
        ys = np.convolve(np.pad(ys, pad, mode="edge"), kernel, mode="valid")
        return np.column_stack([xs, ys])

    def _smooth_world_path(self, xy, samples=260, window=11):
        if xy is None:
            return np.empty((0, 2), dtype=np.float32)

        xy = np.asarray(xy, dtype=np.float32)
        if xy.ndim != 2 or xy.shape[1] != 2:
            return np.empty((0, 2), dtype=np.float32)

        finite = np.all(np.isfinite(xy), axis=1)
        xy = xy[finite]
        if xy.shape[0] < 4 or not TRIAL_ROUTE_SMOOTH_TARGET_LINE:
            return xy

        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        dist = np.concatenate([[0.0], np.cumsum(seg)])
        if dist[-1] < 1e-6:
            return xy

        sample_dist = np.linspace(0.0, dist[-1], samples)
        xs = np.interp(sample_dist, dist, xy[:, 0])
        ys = np.interp(sample_dist, dist, xy[:, 1])

        window = max(3, int(window))
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window, dtype=np.float32) / float(window)
        pad = window // 2
        xs = np.convolve(np.pad(xs, pad, mode="edge"), kernel, mode="valid")
        ys = np.convolve(np.pad(ys, pad, mode="edge"), kernel, mode="valid")
        return np.column_stack([xs, ys])

    def _record_trial_fmm_segment(self, step, raw_xy):
        if not (SAVE_TRIAL_ROUTE_IMAGE and SAVE_TRIAL_ROUTE_LOCAL_FMM_SEGMENTS):
            return
        interval = max(1, int(SAVE_TRIAL_ROUTE_LOCAL_FMM_INTERVAL))
        if step % interval != 0:
            return

        raw_xy = np.asarray(raw_xy, dtype=np.float32)
        if raw_xy.ndim != 2 or raw_xy.shape[0] < 2 or raw_xy.shape[1] != 2:
            return
        finite = np.all(np.isfinite(raw_xy), axis=1)
        raw_xy = raw_xy[finite]
        if raw_xy.shape[0] >= 2:
            self.trial_fmm_world_segments.append(raw_xy.copy())

    def _get_trial_route_points(self):
        points = []
        if self.trial_start_xy is not None:
            points.append(np.asarray(self.trial_start_xy, dtype=np.float32))
        for wp in self.planner.waypoints:
            points.append(np.asarray(wp[:2], dtype=np.float32))

        if not points:
            return np.empty((0, 2), dtype=np.float32)
        return np.vstack(points)

    def _collect_trial_world_points(self, route_pts, traj):
        point_sets = []
        for pts in [route_pts, traj] + self.trial_fmm_world_segments:
            pts = np.asarray(pts, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[0] > 0 and pts.shape[1] >= 2:
                pts = pts[:, :2]
                pts = pts[np.all(np.isfinite(pts), axis=1)]
                if pts.shape[0] > 0:
                    point_sets.append(pts)
        if not point_sets:
            return np.zeros((0, 2), dtype=np.float32)
        return np.vstack(point_sets)

    def _trial_camera_setup(self, route_pts, traj):
        x_min, x_max = TRIAL_ROUTE_CAMERA_X_RANGE
        span_x = max(float(x_max) - float(x_min), 1.0)
        center_x = (float(x_min) + float(x_max)) * 0.5

        if TRIAL_ROUTE_CAMERA_Y_RANGE is not None:
            y_min, y_max = TRIAL_ROUTE_CAMERA_Y_RANGE
            center_y = (float(y_min) + float(y_max)) * 0.5
            span_y = max(float(y_max) - float(y_min), 1.0)
        elif hasattr(self.env, "terrain"):
            terrain_width = float(getattr(self.env.terrain, "env_width", self.env.cfg.terrain.terrain_width))
            border = float(getattr(self.env.terrain.cfg, "border_size", 0.0))
            y_min = -border
            y_max = terrain_width * int(getattr(self.env.terrain.cfg, "num_cols", 1)) - border
            center_y = (y_min + y_max) * 0.5
            span_y = max(y_max - y_min, 1.0)
        else:
            pts = self._collect_trial_world_points(route_pts, traj)
            if pts.shape[0] == 0:
                center_y = float(self.env.root_states[0, 1].detach().cpu().item())
                span_y = 12.0
            else:
                min_y = float(np.min(pts[:, 1]))
                max_y = float(np.max(pts[:, 1]))
                center_y = (min_y + max_y) * 0.5
                span_y = max(max_y - min_y, 12.0)

        center = np.array([center_x, center_y], dtype=np.float32)
        aspect = 16.0 / 9.0
        half_fov = np.deg2rad(TRIAL_ROUTE_CAMERA_FOV_DEG) * 0.5
        height_for_x = span_x / (2.0 * np.tan(half_fov))
        height_for_y = span_y * aspect / (2.0 * np.tan(half_fov))
        height = max(TRIAL_ROUTE_CAMERA_MIN_HEIGHT, height_for_x, height_for_y)
        visible_span_x = 2.0 * height * np.tan(half_fov)
        visible_span_y = visible_span_x / aspect
        return center, float(height), float(visible_span_x), float(visible_span_y)

    def _hide_small_stones_for_route_capture(self):
        if not TRIAL_ROUTE_HIDE_SMALL_STONES_IN_CAMERA:
            return None
        if not hasattr(self.env, "all_root_states") or not hasattr(self.env, "num_actors_per_env"):
            return None

        stone_cfg = getattr(self.env.cfg, "stone", None)
        if stone_cfg is None or not getattr(stone_cfg, "enable", False):
            return None

        num_stones = int(getattr(stone_cfg, "num_stones", 0))
        if num_stones <= 0:
            return None

        big_ids = set(int(i) for i in getattr(stone_cfg, "big_stone_indices", []))
        small_ids = [i for i in range(num_stones) if i not in big_ids]
        if len(small_ids) == 0:
            return None

        actor_indices = torch.tensor(
            [1 + i for i in small_ids],
            dtype=torch.int32,
            device=self.env.device
        )
        old_states = self.env.all_root_states[actor_indices.long()].clone()

        self.env.all_root_states[actor_indices.long(), 0] = -1000.0
        self.env.all_root_states[actor_indices.long(), 1] = -1000.0
        self.env.all_root_states[actor_indices.long(), 2] = -100.0
        self.env.gym.set_actor_root_state_tensor_indexed(
            self.env.sim,
            gymtorch.unwrap_tensor(self.env.all_root_states),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices)
        )
        return actor_indices, old_states

    def _restore_small_stones_after_route_capture(self, hidden_state):
        if hidden_state is None:
            return
        actor_indices, old_states = hidden_state
        self.env.all_root_states[actor_indices.long()] = old_states
        self.env.gym.set_actor_root_state_tensor_indexed(
            self.env.sim,
            gymtorch.unwrap_tensor(self.env.all_root_states),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices)
        )

    def _capture_fixed_overhead_image(self, route_pts, traj):
        if self.env.cam_handle is None or self.env.camera_color_tensor is None:
            return None, None, None

        center, height, span_x, span_y = self._trial_camera_setup(route_pts, traj)
        self._trial_route_camera_center = np.asarray(center, dtype=np.float32)
        self._trial_route_camera_height = float(height)
        self._trial_route_camera_span = np.array([span_x, span_y], dtype=np.float32)
        cam_pos = gymapi.Vec3(
            float(center[0]),
            float(center[1] - TRIAL_ROUTE_CAMERA_BACK_Y_M),
            float(height)
        )
        cam_target = gymapi.Vec3(
            float(center[0]),
            float(center[1]),
            float(TRIAL_ROUTE_CAMERA_LOOKAT_Z)
        )

        self.env.gym.set_camera_location(
            self.env.cam_handle,
            self.env.cam_env_handle,
            cam_pos,
            cam_target
        )

        hidden_state = self._hide_small_stones_for_route_capture()
        try:
            if hasattr(self.env.gym, "step_graphics"):
                self.env.gym.step_graphics(self.env.sim)
            self.env.gym.render_all_camera_sensors(self.env.sim)
            self.env.gym.start_access_image_tensors(self.env.sim)
            rgb = self.env.camera_color_tensor.detach().cpu().numpy()[..., :3].copy()
            self.env.gym.end_access_image_tensors(self.env.sim)
        finally:
            self._restore_small_stones_after_route_capture(hidden_state)

        view_matrix = None
        proj_matrix = None
        try:
            view_matrix = np.asarray(
                self.env.gym.get_camera_view_matrix(self.env.sim, self.env.cam_env_handle, self.env.cam_handle),
                dtype=np.float32
            )
            proj_matrix = np.asarray(
                self.env.gym.get_camera_proj_matrix(self.env.sim, self.env.cam_env_handle, self.env.cam_handle),
                dtype=np.float32
            )
        except Exception:
            view_matrix = None
            proj_matrix = None

        return rgb.astype(np.uint8), view_matrix, proj_matrix

    def _save_raw_route_camera_image(self, trial_id, status, rgb):
        os.makedirs(SAVE_TRIAL_ROUTE_DIR, exist_ok=True)
        raw_path = os.path.join(
            SAVE_TRIAL_ROUTE_DIR,
            f"trial_{trial_id:02d}_{status}_camera_raw.png"
        )
        plt.imsave(raw_path, rgb)
        print(f"Trial raw IsaacGym RGB image saved to: {raw_path}")
        return raw_path

    def _world_xy_to_overhead_pixels(self, xy, image_shape):
        xy = np.asarray(xy, dtype=np.float32)
        if xy.ndim != 2 or xy.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32)

        h, w = image_shape[:2]
        center = getattr(self, "_trial_route_camera_center", None)
        span = getattr(self, "_trial_route_camera_span", None)
        if center is None or span is None:
            return None

        span_x = float(span[0])
        span_y = float(span[1])
        x0 = float(center[0]) - span_x * 0.5
        y1 = float(center[1]) + span_y * 0.5
        pix = np.column_stack([
            (xy[:, 0] - x0) / span_x * w,
            (y1 - xy[:, 1]) / span_y * h,
        ])
        return pix.astype(np.float32)

    def _world_xy_to_xyz(self, xy):
        xy = np.asarray(xy, dtype=np.float32)
        if xy.ndim != 2 or xy.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        xyz = np.zeros((xy.shape[0], 3), dtype=np.float32)
        xyz[:, :2] = xy[:, :2]
        for i, p in enumerate(xy):
            if hasattr(self.env, "_get_terrain_height_at"):
                z = self.env._get_terrain_height_at(float(p[0]), float(p[1]))
            else:
                z = 0.0
            xyz[i, 2] = float(z) + TRIAL_ROUTE_OVERLAY_Z_OFFSET
        return xyz

    def _project_world_points_to_image(self, xy, view_matrix, proj_matrix, image_shape):
        if view_matrix is None or proj_matrix is None:
            return self._world_xy_to_overhead_pixels(xy, image_shape)

        xyz = self._world_xy_to_xyz(xy)
        if xyz.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32)

        h, w = image_shape[:2]
        pts_h = np.column_stack([xyz, np.ones(xyz.shape[0], dtype=np.float32)])
        candidates = []
        for clip in [
            (proj_matrix @ view_matrix @ pts_h.T).T,
            pts_h @ view_matrix @ proj_matrix,
        ]:
            denom = clip[:, 3]
            valid = np.isfinite(denom) & (np.abs(denom) > 1e-6)
            ndc = np.full((clip.shape[0], 2), np.nan, dtype=np.float32)
            ndc[valid] = clip[valid, :2] / denom[valid, None]
            pix = np.column_stack([
                (ndc[:, 0] + 1.0) * 0.5 * w,
                (1.0 - ndc[:, 1]) * 0.5 * h,
            ])
            in_frame = (
                np.isfinite(pix[:, 0]) & np.isfinite(pix[:, 1]) &
                (pix[:, 0] >= -0.1 * w) & (pix[:, 0] <= 1.1 * w) &
                (pix[:, 1] >= -0.1 * h) & (pix[:, 1] <= 1.1 * h)
            )
            candidates.append((np.count_nonzero(in_frame), pix.astype(np.float32)))

        candidates.sort(key=lambda item: item[0], reverse=True)
        if candidates[0][0] > 0:
            return candidates[0][1]
        return self._world_xy_to_overhead_pixels(xy, image_shape)

    def _plot_projected_line(self, ax, xy, view_matrix, proj_matrix, image_shape, **kwargs):
        pix = self._project_world_points_to_image(xy, view_matrix, proj_matrix, image_shape)
        if pix is None or pix.shape[0] < 2:
            return False
        finite = np.all(np.isfinite(pix), axis=1)
        pix = pix[finite]
        if pix.shape[0] < 2:
            return False
        ax.plot(pix[:, 0], pix[:, 1], **kwargs)
        return True

    def _plot_projected_point(self, ax, xy, view_matrix, proj_matrix, image_shape, **kwargs):
        pix = self._project_world_points_to_image(np.asarray([xy], dtype=np.float32), view_matrix, proj_matrix, image_shape)
        if pix is None or pix.shape[0] == 0 or not np.all(np.isfinite(pix[0])):
            return False
        ax.scatter([pix[0, 0]], [pix[0, 1]], **kwargs)
        return True

    def _save_trial_route_plot(self, trial_id, status):
        if not SAVE_TRIAL_ROUTE_IMAGE:
            return

        traj = np.asarray(self.traj_xy_cur, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] == 0:
            return

        route_pts = self._get_trial_route_points()
        rgb, view_matrix, proj_matrix = self._capture_fixed_overhead_image(route_pts, traj)
        if rgb is None:
            print("Trial route image skipped: no IsaacGym camera tensor is available.")
            return

        raw_path = self._save_raw_route_camera_image(trial_id, status, rgb)
        if np.max(rgb) <= 3:
            print("Warning: raw IsaacGym RGB image is almost black; check camera pose, lighting, or sensor rendering.")
        base_img = plt.imread(raw_path)

        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        ax.imshow(base_img)
        ax.axis("off")

        fmm_drawn = False
        if SAVE_TRIAL_ROUTE_LOCAL_FMM_SEGMENTS and len(self.trial_fmm_world_segments) > 0:
            for seg in self.trial_fmm_world_segments:
                drawn = self._plot_projected_line(
                    ax, seg, view_matrix, proj_matrix, rgb.shape,
                    color="#006dff", linewidth=1.6, alpha=0.45
                )
                fmm_drawn = fmm_drawn or drawn
            if fmm_drawn:
                ax.plot([], [], color="#006dff", linewidth=2.0, alpha=0.65, label="FMM planned path")

        if traj.shape[0] >= 2:
            if self._plot_projected_line(
                ax, traj, view_matrix, proj_matrix, rgb.shape,
                color="#00d45a", linewidth=3.0, alpha=0.95
            ):
                ax.plot([], [], color="#00d45a", linewidth=3.0, label="Executed trajectory")

        if TRIAL_ROUTE_SHOW_TARGET_ORDER_LINE and route_pts.shape[0] >= 2:
            route_line = self._smooth_world_path(route_pts)
            if self._plot_projected_line(
                ax, route_line, view_matrix, proj_matrix, rgb.shape,
                color="#ffcc00", linewidth=2.0, alpha=0.85, linestyle="--"
            ):
                ax.plot([], [], color="#ffcc00", linewidth=2.0, linestyle="--", label="Target-order reference")

        if route_pts.shape[0] >= 2:
            if route_pts.shape[0] > 2:
                for idx, p in enumerate(route_pts[1:-1], start=1):
                    self._plot_projected_point(
                        ax, p, view_matrix, proj_matrix, rgb.shape,
                        s=68, c="#ff9f1a", edgecolors="black", linewidths=0.8, zorder=6
                    )
                    pix = self._project_world_points_to_image(np.asarray([p], dtype=np.float32), view_matrix, proj_matrix, rgb.shape)
                    if pix is not None and np.all(np.isfinite(pix[0])):
                        ax.text(pix[0, 0], pix[0, 1], str(idx), fontsize=8, ha="center", va="center", zorder=7)
                ax.scatter([], [], s=68, c="#ff9f1a", edgecolors="black", label="Waypoints")

            if self._plot_projected_point(
                ax, route_pts[0], view_matrix, proj_matrix, rgb.shape,
                s=140, c="red", marker="*", edgecolors="black", linewidths=0.8, zorder=7
            ):
                ax.scatter([], [], s=140, c="red", marker="*", edgecolors="black", label="Start")
            if self._plot_projected_point(
                ax, route_pts[-1], view_matrix, proj_matrix, rgb.shape,
                s=120, c="lime", marker="X", edgecolors="black", linewidths=0.8, zorder=7
            ):
                ax.scatter([], [], s=120, c="lime", marker="X", edgecolors="black", label="Final goal")

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=9, framealpha=0.85)

        os.makedirs(SAVE_TRIAL_ROUTE_DIR, exist_ok=True)
        save_path = os.path.join(
            SAVE_TRIAL_ROUTE_DIR,
            f"trial_{trial_id:02d}_{status}_route_camera.png"
        )
        fig.tight_layout(pad=0)
        fig.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"Trial IsaacGym camera route image saved to: {save_path}")

    def _format_fmm_potential(self):
        dist = getattr(self.fmm, "fmm_dist", None)
        if dist is None:
            return None

        potential = np.asarray(dist, dtype=np.float32).copy()
        large = float(getattr(self.fmm, "large", 1e6))
        potential[potential >= large * 0.5] = np.nan

        finite = np.isfinite(potential)
        if not np.any(finite):
            return None

        vmax = np.nanpercentile(potential[finite], 95)
        if np.isfinite(vmax) and vmax > 0:
            potential = np.clip(potential, 0.0, vmax)
        return potential

    def _draw_bev_path_overlay(self, ax, path_xy, smooth_xy, title_suffix=""):
        if smooth_xy.shape[0] >= 2:
            ax.plot(smooth_xy[:, 0], smooth_xy[:, 1], color="dodgerblue", linewidth=3.0, label="Path")
        if path_xy.shape[0] >= 2:
            ax.plot(path_xy[:, 0], path_xy[:, 1], color="white", linewidth=1.0, alpha=0.65, linestyle="--", label="Raw")

        start = (0.0, float(self.fmm.robot_r) * self.fmm.bev_res)
        ax.scatter([start[0]], [start[1]], c="red", s=55, marker="*", label="Start")
        if getattr(self.fmm, "goal_rc", None) is not None:
            gr, gc = self.fmm.goal_rc
            goal = ((gc - self.fmm.W // 2) * self.fmm.bev_res, gr * self.fmm.bev_res)
            ax.scatter([goal[0]], [goal[1]], c="lime", s=55, marker="o", edgecolors="black", linewidths=0.6, label="Goal")

        y_min = -self.fmm.W * self.fmm.bev_res * 0.5
        y_max = self.fmm.W * self.fmm.bev_res * 0.5
        x_max = self.fmm.H * self.fmm.bev_res
        ax.set_xlim(y_min, y_max)
        ax.set_ylim(0.0, x_max)
        ax.set_xlabel("y (m)")
        ax.set_ylabel("x (m)")
        if title_suffix:
            ax.set_title(title_suffix)

    def _save_bev_fmm_single_panel(self, subdir, trial_id, step, title, draw_image, path_xy, smooth_xy):
        save_dir = os.path.join(SAVE_BEV_FMM_DIR, subdir)
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(6.3, 5.6))
        image_artist = draw_image(ax)
        self._draw_bev_path_overlay(ax, path_xy, smooth_xy, title)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=8, framealpha=0.85)
        if image_artist is not None:
            fig.colorbar(image_artist, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Trial {trial_id} Step {step}: {title}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        save_path = os.path.join(save_dir, f"trial_{trial_id:02d}_step_{step:05d}.png")
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def _save_bev_fmm_separate_images(self, trial_id, step, hr, obstacle_rgb, potential, occ_bev, path_xy, smooth_xy, extent_m):
        self._save_bev_fmm_single_panel(
            "height",
            trial_id,
            step,
            "Height",
            lambda ax: ax.imshow(hr, origin="lower", extent=extent_m, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto"),
            path_xy,
            smooth_xy,
        )
        self._save_bev_fmm_single_panel(
            "obstacles",
            trial_id,
            step,
            "Obstacles",
            lambda ax: (ax.imshow(obstacle_rgb, origin="lower", extent=extent_m, interpolation="nearest", aspect="auto"), None)[1],
            path_xy,
            smooth_xy,
        )
        if BEV_FMM_SHOW_POTENTIAL:
            def draw_potential(ax):
                if potential is not None:
                    im = ax.imshow(potential, origin="lower", extent=extent_m, cmap="turbo", interpolation="nearest", aspect="auto")
                    blocked = np.ma.masked_where(occ_bev <= 0, occ_bev)
                    ax.imshow(blocked, origin="lower", extent=extent_m, cmap="Reds", alpha=0.45, interpolation="nearest", aspect="auto")
                    return im
                return ax.imshow(occ_bev, origin="lower", extent=extent_m, cmap="gray_r", interpolation="nearest", aspect="auto")

            self._save_bev_fmm_single_panel(
                "potential",
                trial_id,
                step,
                "Potential",
                draw_potential,
                path_xy,
                smooth_xy,
            )

    def _visualize_bev_fmm(self, step, trial_id, occ_bev, path_rc):
        should_save = (
            SAVE_BEV_FMM_IMAGES
            and step % SAVE_BEV_FMM_INTERVAL == 0
            and self._bev_vis_saved_count < SAVE_BEV_FMM_MAX_PER_TRIAL
        )
        should_show = LIVE_SHOW_BEV_FMM_WINDOW and step % LIVE_BEV_FMM_WINDOW_INTERVAL == 0

        if not (should_save or should_show):
            return

        hr = np.clip(self.bev.max_z - self.bev.min_z, 0.0, 1.0)
        path_xy = self._path_rc_to_plot_xy(path_rc)
        smooth_xy = self._smooth_plot_path(path_xy)
        potential = self._format_fmm_potential()

        obstacle_rgb = np.ones((*occ_bev.shape, 3), dtype=np.float32)
        obstacle_rgb[occ_bev > 0] = np.array([0.18, 0.18, 0.18], dtype=np.float32)
        extent_m = [
            -self.fmm.W * self.fmm.bev_res * 0.5,
            self.fmm.W * self.fmm.bev_res * 0.5,
            0.0,
            self.fmm.H * self.fmm.bev_res,
        ]

        ncols = 3 if BEV_FMM_SHOW_POTENTIAL else 2
        if should_show:
            if self._bev_vis_fig is None:
                plt.ion()
                self._bev_vis_fig = plt.figure(figsize=(5.2 * ncols, 4.8))
            fig = self._bev_vis_fig
            fig.clf()
        else:
            fig = plt.figure(figsize=(5.2 * ncols, 4.8))

        axes = fig.subplots(1, ncols)
        if ncols == 1:
            axes = [axes]

        ax = axes[0]
        im0 = ax.imshow(hr, origin="lower", extent=extent_m, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
        self._draw_bev_path_overlay(ax, path_xy, smooth_xy, "Height")
        fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1]
        ax.imshow(obstacle_rgb, origin="lower", extent=extent_m, interpolation="nearest", aspect="auto")
        self._draw_bev_path_overlay(ax, path_xy, smooth_xy, "Obstacles")

        if BEV_FMM_SHOW_POTENTIAL:
            ax = axes[2]
            if potential is not None:
                im2 = ax.imshow(potential, origin="lower", extent=extent_m, cmap="turbo", interpolation="nearest", aspect="auto")
                blocked = np.ma.masked_where(occ_bev <= 0, occ_bev)
                ax.imshow(blocked, origin="lower", extent=extent_m, cmap="Reds", alpha=0.45, interpolation="nearest", aspect="auto")
                fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.imshow(occ_bev, origin="lower", extent=extent_m, cmap="gray_r", interpolation="nearest", aspect="auto")
            self._draw_bev_path_overlay(ax, path_xy, smooth_xy, "Potential")

        handles, labels = axes[-1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=min(5, len(by_label)), fontsize=8)

        fig.suptitle(f"Trial {trial_id} Step {step}", fontsize=12)
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))

        if should_save:
            self._save_bev_fmm_separate_images(
                trial_id,
                step,
                hr,
                obstacle_rgb,
                potential,
                occ_bev,
                path_xy,
                smooth_xy,
                extent_m,
            )
            self._bev_vis_saved_count += 1

        if should_show:
            plt.pause(0.001)
        else:
            plt.close(fig)
            
    #可视化目标点，预测路径，已走过路线
    def _draw_current_visualization(self):
        self.env.gym.clear_lines(self.env.viewer)
        if LIVE_DRAW_WAYPOINTS:
            self._draw_waypoints()
        if LIVE_DRAW_TRAJECTORY:
            self._draw_traj()
        if LIVE_DRAW_FMM_PATH:
            self._draw_fmm_path_spheres()

    def _show_completed_route(self):
        if not LIVE_HOLD_FINAL_ROUTE:
            return
        # Make the final "all waypoints visited" state visible before the next trial resets.
        if self.render and self.env.viewer is not None:
            self._draw_current_visualization()
            self.env.render(sync_frame_time=True)
        self._hold_final_visualization()

    def _hold_final_visualization(self):
        if not self.render or self.env.viewer is None:
            return

        old_viewer_sync = self.env.enable_viewer_sync
        self.env.enable_viewer_sync = True
        print(f"Final route visualization hold: {self.final_hold_seconds:.1f}s")

        end_time = time.monotonic() + self.final_hold_seconds
        while time.monotonic() < end_time:
            self._draw_current_visualization()
            self.env.render(sync_frame_time=True)
            time.sleep(0.02)

        self.env.enable_viewer_sync = old_viewer_sync

    def _compute_metrics(self):
        """
        Low-level locomotion evaluation metrics:
        - Velocity tracking error (MAE)
        - Orientation stability error
        - Angular velocity stability error
        - Energy proxy
        """
        if len(self.state_log) == 0:
            return {}

        # 速度跟踪
        cmd_vx = np.array([s["cmd_vx"] for s in self.state_log])
        real_vx = np.array([s["real_vx"] for s in self.state_log])
        vel_tracking_mae = np.mean(np.abs(cmd_vx - real_vx))

        # 能耗
        torque_cost = np.array([s["torque_cost"] for s in self.state_log])
        energy = np.mean(torque_cost)

        cmd_wz = np.array([s["cmd_wz"] for s in self.state_log], dtype=np.float64)
        cmd_wz_variation = (float(np.mean(np.abs(np.diff(cmd_wz))))
                            if cmd_wz.size > 1 else 0.0)
        active_sign = np.sign(cmd_wz[np.abs(cmd_wz) > 0.05])
        cmd_wz_sign_flips = (int(np.count_nonzero(np.diff(active_sign) != 0))
                             if active_sign.size > 1 else 0)

        # 稳定性
        stab = self._compute_stability_errors()

        return {
            "vel_tracking_mae": float(vel_tracking_mae),
            "orientation_error": stab["orientation_error"],
            "angular_vel_error": stab["angular_vel_error"],
            "energy": float(energy),
            "cmd_wz_variation": cmd_wz_variation,
            "cmd_wz_sign_flips": cmd_wz_sign_flips,
        }

    def _compute_stability_errors(self):
        """
        Compute stability-related errors for low-level locomotion evaluation.
        Returns:
            orientation_error: mean sqrt(roll^2 + pitch^2)
            angular_vel_error: mean sqrt(wx^2 + wy^2)
        """
        if len(self.state_log) == 0:
            return {
                "orientation_error": 0.0,
                "angular_vel_error": 0.0,
            }

        roll_arr = np.array([s["roll"] for s in self.state_log], dtype=np.float64)
        pitch_arr = np.array([s["pitch"] for s in self.state_log], dtype=np.float64)
        wx_arr = np.array([s["body_wx"] for s in self.state_log], dtype=np.float64)
        wy_arr = np.array([s["body_wy"] for s in self.state_log], dtype=np.float64)

        orientation_error = np.mean(np.sqrt(roll_arr**2 + pitch_arr**2))
        angular_vel_error = np.mean(np.sqrt(wx_arr**2 + wy_arr**2))

        return {
            "orientation_error": float(orientation_error),
            "angular_vel_error": float(angular_vel_error),
        }
    def _quat_to_rpy(self, quat):
        """
        Convert quaternion to roll, pitch, yaw (rad)
        quat: torch.Tensor [4]
        """
        q = quat.cpu().numpy()
        w, x, y, z = q[3], q[0], q[1], q[2]

        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, pitch, yaw

    def _save_camera_images(self, step, rgb, depth_pos):
        if not SAVE_CAMERA_IMAGES or step % SAVE_CAMERA_IMAGE_INTERVAL != 0:
            return

        os.makedirs(SAVE_CAMERA_IMAGE_DIR, exist_ok=True)
        # rgb is None => CAM_READ_RGB=False（省内存，只读 depth）；
        # 配置正确时 SAVE_RGB_IMAGE=True 会自动把 CAM_READ_RGB 推导成 True。
        if SAVE_RGB_IMAGE and rgb is not None:
            rgb_path = os.path.join(SAVE_CAMERA_IMAGE_DIR, f"rgb_{step:06d}.png")
            plt.imsave(rgb_path, rgb)

        if SAVE_DEPTH_IMAGE:
            depth_path = os.path.join(SAVE_CAMERA_IMAGE_DIR, f"depth_{step:06d}.png")
            plt.imsave(depth_path, depth_pos, cmap="gray", vmin=0.0, vmax=6.0)

    def _show_camera_images(self, step, rgb, depth_pos):
        if not LIVE_SHOW_CAMERA_IMAGES or step % LIVE_CAMERA_IMAGE_INTERVAL != 0:
            return
        # 这个窗口要同屏显示 RGB+Depth，缺 RGB 就没意义（CAM_READ_RGB=False）
        if rgb is None:
            return

        if self._camera_vis_fig is None:
            plt.ion()
            self._camera_vis_fig, self._camera_vis_axes = plt.subplots(1, 2, figsize=(10, 5))
            ax_rgb, ax_depth = self._camera_vis_axes
            self._camera_vis_rgb = ax_rgb.imshow(rgb)
            ax_rgb.set_title("RGB")
            self._camera_vis_depth = ax_depth.imshow(depth_pos, cmap="gray", vmin=0.0, vmax=6.0)
            ax_depth.set_title("Depth")
        else:
            self._camera_vis_rgb.set_data(rgb)
            self._camera_vis_depth.set_data(depth_pos)

        plt.pause(0.001)

    def _handle_camera_images(self, step, rgb, depth_pos):
        self._save_camera_images(step, rgb, depth_pos)
        self._show_camera_images(step, rgb, depth_pos)
                    
    def _mask_occ_to_actors(self, occ_bev, base_pos, base_yaw):
        """把障碍图限制在【actor 障碍】附近（白名单）。

        为什么需要（2026-09-17 实测定位）：
          BEV 的“同格高度差”判据不区分障碍来源。no_obstacles 场景（无任何石头）
          里它仍稳定检出 ~55 个障碍格，世界坐标聚类全部落在 (8~13, 11) ——
          即 terrain 四周那圈 4m 高【边界墙】（起点距左墙仅 2.4m，落在 BEV
          横向 ±3m 内）。FMM 于是常年看见一堵左墙、持续往右推，观感上像
          “在绕地形上的离散障碍”。地形凸起（hr≈0.01m）低于阈值、本就不会被检出。
          用户语义：只绕【大石头 / 后续新增的动态障碍】这类 actor 障碍。
        做法：actor 位置（env.all_root_states[1:num_actors_per_env]，含石头与树，
          后续新增动态障碍只要建为 actor 即自动纳入）转到 BEV 格坐标，
          按 BEV_ACTOR_MASK_RADIUS_M 画圆掩膜，障碍图与掩膜取交。
        掩膜失败（拿不到 actor 状态）时原样返回，绝不中断导航。

        ⚠ 2026-09-17 修复的真 bug：场景里【一个障碍 actor 都没有】时
          （num_actors_per_env <= 1，如 no_obstacles 论文场景），本函数原先
          `return occ_bev` —— 把含边界墙的原始障碍图原样放行，白名单形同虚设，
          用户观察到的“绕地形上的离散障碍”因此始终存在。
          正确语义是【没有任何 actor 障碍 => 障碍图全零】。
        """
        try:
            n = int(getattr(self.env, "num_actors_per_env", 0) or 0)
            states = getattr(self.env, "all_root_states", None)
            if states is None:
                return occ_bev          # 拿不到状态：降级为原行为，绝不中断导航
            if n <= 1:
                # 场景里没有任何障碍 actor：白名单的交集为空 => 全零障碍图
                return np.zeros_like(occ_bev)
            pos = states[1:n, :2].detach().cpu().numpy().astype(np.float64)
            if pos.size == 0:
                return np.zeros_like(occ_bev)
            dx = pos[:, 0] - base_pos[0]
            dy = pos[:, 1] - base_pos[1]
            c = np.cos(-base_yaw); s_ = np.sin(-base_yaw)
            xf = c * dx - s_ * dy        # 前向
            yl = s_ * dx + c * dy        # 左侧
            res = float(self.bev.bev_res)
            H, W = occ_bev.shape
            rr = (xf / res).astype(np.int64)
            cc = (yl / res).astype(np.int64) + W // 2
            rad = int(max(1, round(float(BEV_ACTOR_MASK_RADIUS_M) / res)))
            mask = np.zeros((H, W), dtype=np.uint8)
            for r0_, c0_ in zip(rr, cc):
                lo_r, hi_r = max(0, r0_ - rad), min(H, r0_ + rad + 1)
                lo_c, hi_c = max(0, c0_ - rad), min(W, c0_ + rad + 1)
                if lo_r < hi_r and lo_c < hi_c:
                    mask[lo_r:hi_r, lo_c:hi_c] = 1
            return (occ_bev & mask).astype(np.uint8)
        except Exception:
            return occ_bev

    def _forward_clearance(self, occ_bev):
        """返回机器人正前方的净空距离 [m]，供速度调制使用。

        上层原本只算 wz（转向），vx 恒为 NAV_FORWARD_VX，于是“看得见障碍却仍全速
        撞上去”。这里补上缺失的距离感知。

        两个数据源，取【更保守（更近）】的那个：
          1) 像素级判据 BEVMapper.forward_obstacle_distance（主）
             BEV 的“同格内高度差”判据对正对的垂直墙面天然失效：墙面上每个像素
             深度几乎相同，反投影后落在同一格、高度也几乎相同，格内高度差≈0。
             实测（走向 0.76m 高的大石头）：距石 1.76m 时锥内净空 0.55m，
             但 1.10m 起近距障碍格归 0，净空【跳回最大值 3.00m】，
             速度调制于是误判“前方无障”反而加速冲向石头 ——
             感知失效被读成通路，这是最危险的失效方向。像素级判据没有这个盲区
             （离线单测：墙在 0.3~3.0m 全部测准，误差 <0.05m）。
          2) BEV 栅格锥扫（后备）
             障碍图已由规划链路算好，顺手扫一遍锥内栅格。BEV 栅格已是机器人坐标系
             （r = 前方 x，c = 左侧 y，机器人在 r=0, c=W//2），无需再做坐标变换。

        返回 SPEED_MOD_MAX_RANGE 表示前方无碍（全速）。
        """
        best = float(SPEED_MOD_MAX_RANGE)

        # --- 数据源 1：像素级（用最近一次读到的深度图）---
        depth = getattr(self, "_last_depth_pos", None)
        # ⚠ 默认关闭：该判据的地面模型用了 sin(θ)（斜距），而 IMAGE_DEPTH 是轴向深度、
        #   应为 tan(θ)，实测在无障碍地形上误报 99%。见 viz_config.SPEED_MOD_USE_PIXEL_JUDGE。
        if (SPEED_MOD_USE_PIXEL_JUDGE and depth is not None
                and hasattr(self.bev, "forward_obstacle_distance")):
            try:
                dist_px, _hits = self.bev.forward_obstacle_distance(
                    depth,
                    half_angle_deg=SPEED_MOD_CONE_HALF_ANGLE,
                    ground_ratio=SPEED_MOD_GROUND_RATIO,
                    max_range=SPEED_MOD_MAX_RANGE,
                    # ROI 与 BEV 建图解耦：BEV 用 0.55（只看地面，避免把地形起伏
                    # 误判成障碍，实测论文场景 0.55 -> 0.90 而 0.10 -> 0.70），
                    # 但刹车判据必须能看见高出相机的直立物，所以用更宽的 0.10。
                    v_start_ratio=SPEED_MOD_ROI_START,
                    min_fwd=SPEED_MOD_PIXEL_MIN_FWD)
                best = min(best, float(dist_px))
            except Exception:
                pass    # 感知降级不该中断导航，下面的 BEV 锥扫仍兜底

        # --- 数据源 2：BEV 栅格锥扫 ---
        if occ_bev is None or not np.any(occ_bev):
            return best

        H, W = occ_bev.shape
        res = float(self.bev.bev_res)
        max_cells = int(min(H - 1, SPEED_MOD_MAX_RANGE / res))
        if max_cells < 1:
            return best

        c0 = W // 2
        # 逐行(前向距离)检查该距离上的锥内是否有障碍，取最近的障碍距离。
        # 锥宽 = r * tan(半角)，离得越远横向容忍越大。
        # ⚠ 半角必须先转成弧度再取 tan：原实现是 int(ceil(deg*pi/180))，
        #   既取整又当弧度用（25° -> ceil(0.436) = 0 -> tan(0) = 0），
        #   锥宽恒为 0，等于只扫机器人正前方那一列，侧向障碍全部漏掉。
        tan_half = float(np.tan(np.deg2rad(SPEED_MOD_CONE_HALF_ANGLE)))
        for r in range(1, max_cells + 1):
            half_w = int(np.floor(r * tan_half))
            c_lo = max(0, c0 - half_w)
            c_hi = min(W - 1, c0 + half_w)
            if c_lo <= c_hi and occ_bev[r, c_lo:c_hi + 1].any():
                return min(best, float(r * res))
        return best

    def _couple_vx_wz(self, target_vx, wz, clearance=None):
        """速度-转向耦合：急转时压低前进速度目标值。

        为什么需要（2026-09-16 wz 日志实测定位）：
          FMM 把机器狗当【全向点机器人】：wz=-1.0 + vx=1.0 时它认为“在原地左转”。
          但盲走底层是差速近似：转弯半径 = vx/wz = 1.0m，实际轨迹是半径 1m 的圆弧，
          圆弧弯向障碍一侧 -> 顶着石头楔死（指令 0.61 实际 0.011，clearance=3.0
          无障却动不了）。急转时把 vx 压到 0.3，转弯半径缩到 0.3m，
          机器狗才能真正“转得动”而不是画大圆弧撞上去。
        作用于【限幅前的目标值】，加速度限幅仍由 _modulate_vx 统一施加。
        """
        if not SPEED_TURN_COUPLING_ENABLE:
            return target_vx
        # 启用条件（2026-09-17 四组实测定标，各 10 trial）：
        #   · clearance < SPEED_TURN_CLEARANCE_M：耦合【默认唯一生效档】。绕障碍的
        #     中速转降速，防止 1m 转弯圆弧顶住石头。实测 no_obstacles 0.90、
        #     stones_only 0.80。
        #   · SPEED_TURN_HARD_ENABLE=True 时额外启用「急转档」：|wz| >=
        #     SPEED_TURN_WZ_HARD 不看 clearance 直接耦合。**默认关**，实测净负收益：
        #     no_obstacles 0.90->0.00~0.10、stones_only 0.80->0.00。失效机理是航点
        #     转角处 |wz| 达 0.7~1.0，vx 被压到 0.3 后盲走底层在粗糙坡失去前进动量
        #     （指令 0.8~1.0、实际 0.000）判"卡住"。
        #   · 其余（开阔地中缓转）：不耦合。无条件全耦合同样会让底层在粗糙地面
        #     失去动量顶死（stones_only 0.60、I 组 0.00）。
        a0 = abs(float(wz))
        hard = bool(SPEED_TURN_HARD_ENABLE) and a0 >= float(SPEED_TURN_WZ_HARD)
        near = (clearance is not None and SPEED_TURN_CLEARANCE_M > 0
                and float(clearance) < float(SPEED_TURN_CLEARANCE_M))
        if not (hard or near):
            return target_vx
        a = min(abs(float(wz)), float(SPEED_TURN_WZ_HI))
        lo, hi = float(SPEED_TURN_WZ_LO), float(SPEED_TURN_WZ_HI)
        if a <= lo:
            return target_vx
        if a >= hi:
            return min(target_vx, float(SPEED_TURN_MIN_VX))
        r = (a - lo) / max(hi - lo, 1e-9)
        return target_vx + r * (float(SPEED_TURN_MIN_VX) - target_vx)

    def _modulate_vx(self, clearance, dt, wz=0.0, goal_distance=None,
                     speed_cap=None):
        """按前向净空距离把 vx 从 NAV_FORWARD_VX 梯形降到 SPEED_MOD_MIN_VX。

        带加速度限幅（SPEED_MOD_MAX_ACC），避免 vx 阶跃把盲走底层踢失稳 ——
        底层是按平滑速度指令训练的，突变的 cmd_vx 本身就可能导致摔倒。
        """
        lo, hi = float(SPEED_MOD_STOP_DIST), float(SPEED_MOD_SLOW_DIST)
        if not SPEED_MOD_ENABLE or clearance >= hi:
            target = float(NAV_FORWARD_VX)
        elif clearance <= lo:
            target = float(SPEED_MOD_MIN_VX)
        else:
            ratio = (clearance - lo) / (hi - lo)
            target = float(SPEED_MOD_MIN_VX
                           + ratio * (NAV_FORWARD_VX - SPEED_MOD_MIN_VX))
        # 最终目标接近速度：与障碍调速取更保守值，但到达判定仍是独立的欧氏距离。
        if goal_distance is not None and NAV_GOAL_SLOW_DIST_M > NAV_GOAL_REACH_THRESH_M:
            distance = float(goal_distance)
            if distance < float(NAV_GOAL_SLOW_DIST_M):
                ratio = np.clip(
                    (distance - NAV_GOAL_REACH_THRESH_M)
                    / (NAV_GOAL_SLOW_DIST_M - NAV_GOAL_REACH_THRESH_M),
                    0.0, 1.0)
                goal_target = (NAV_GOAL_APPROACH_MIN_VX
                               + ratio * (NAV_FORWARD_VX - NAV_GOAL_APPROACH_MIN_VX))
                target = min(target, float(goal_target))
        if speed_cap is not None:
            target = min(target, float(speed_cap))
        # 加速度限幅
        max_delta = float(SPEED_MOD_MAX_ACC) * max(dt, 1e-6)
        cur = self._last_cmd_vx
        limited = float(np.clip(target, cur - max_delta, cur + max_delta))
        # 速度-转向耦合：急转目标降速（见 _couple_vx_wz），再统一限幅
        limited = self._couple_vx_wz(limited, wz, clearance)
        # 下限保护：绝不低于 MIN_VX（也不允许负速，避免倒退）
        speed_floor = min(SPEED_MOD_MIN_VX, NAV_FORWARD_VX)
        if goal_distance is not None:
            speed_floor = min(speed_floor, NAV_GOAL_APPROACH_MIN_VX)
        limited = float(max(limited, speed_floor))
        self._last_cmd_vx = limited
        return limited

    def _update_camera_and_bev(self, step):

        # 无相机（headless 跑 train、或 cam 创建失败）时直接跳过，
        # 否则下面 set_camera_location(None, ...) 会报错。
        # 与 _capture_fixed_overhead_image 的守卫保持一致。
        if self.env.cam_handle is None or self.env.camera_depth_tensor is None:
            return False

        # === 相机跟随 ===
        base_pos = self.env.root_states[0, 0:3].cpu().numpy()
        base_yaw = self._get_base_yaw(0)

        fx = np.cos(base_yaw)
        fy = np.sin(base_yaw)
        # 机位参数见 viz_config.py 第十四节；CAM_FOLLOW_HEIGHT 需与 BEV_CAM_HEIGHT 一致
        cam_pos = gymapi.Vec3(
            base_pos[0] + CAM_FOLLOW_FORWARD * fx,
            base_pos[1] + CAM_FOLLOW_FORWARD * fy,
            base_pos[2] + CAM_FOLLOW_HEIGHT
        )

        # 光轴俯角（向下为正）：把注视点沿光轴方向下压 look*tan(pitch)。
        # 目的：平视时近距直立障碍（石头顶只比相机高 0.138m）落在 ROI 上沿之外，
        # 1.1m 内必然漏检（几何下界，换判据无解）；下压光轴可把盲区下界推近。
        # 见 viz_config.BEV_CAM_PITCH_DEG 的实测记录。
        pitch = float(np.deg2rad(BEV_CAM_PITCH_DEG))
        cam_target = gymapi.Vec3(
            base_pos[0] + CAM_FOLLOW_LOOKAT_FORWARD * fx,
            base_pos[1] + CAM_FOLLOW_LOOKAT_FORWARD * fy,
            base_pos[2] + CAM_FOLLOW_HEIGHT - CAM_FOLLOW_LOOKAT_FORWARD * np.tan(pitch)
        )

        self.env.gym.set_camera_location(
            self.env.cam_handle,
            self.env.cam_env_handle,
            cam_pos,
            cam_target
        )

        # === 读深度 ===
        # CAM_READ_ENABLE=False 时跳过整段 GPU→CPU 拷贝（每步两张图）。
        # 只有 BEV_ENABLE_UPDATE / 存图 / 实时窗口 才真正需要深度数据，
        # 见 viz_config.CAM_READ_ENABLE 的说明；check_consistency 会提示误配。
        if not CAM_READ_ENABLE:
            return False
        # 感知降频：主循环 50Hz，而 BEV 栅格 5cm、机器狗 1.0m/s，每步只挪 0.4 格，
        # 每步重建障碍图是严重过采样。间隔内直接复用上一次的 BEV（min_z/max_z 不变），
        # 渲染拷贝与反投影开销同步降到 1/N。开关见 viz_config.PERCEPT_UPDATE_INTERVAL。
        interval = PERCEPT_UPDATE_INTERVAL if PERCEPT_UPDATE_INTERVAL else 1
        if interval > 1 and step % int(interval) != 0:
            return False
        # ⚠ 关键：IsaacGym 要求先 step_graphics 同步相机位姿，再 render_all_camera_sensors，
        #   否则拿到的是上一帧的陈旧深度图。原本这一步是靠 env.step() 里的
        #   env.render() 顺带做的，而 render() 只在 enable_viewer_sync=True 时才调
        #   step_graphics —— 所以一旦用 LIVE_VIEWER_SYNC=False 关掉逐帧重绘，
        #   深度相机就会静默失效、导航跟着退化。这里显式补上，把两者解耦。
        #   viewer 已在逐帧重绘时不重复调用，避免白白多花一次 step_graphics。
        if (getattr(self.env, "viewer", None) is None
                or not getattr(self.env, "enable_viewer_sync", True)):
            if hasattr(self.env.gym, "step_graphics"):
                self.env.gym.step_graphics(self.env.sim)
        self.env.gym.render_all_camera_sensors(self.env.sim)
        self.env.gym.start_access_image_tensors(self.env.sim)
        depth = self.env.camera_depth_tensor.detach().cpu().numpy()
        # RGB 只在真要存图/显示时才拷回来。避障链路只需要 depth，
        # 原代码每步无条件多拷一张 float32（320x180 = 0.23MB），泄漏量直接翻倍。
        # 消费方判定见 viz_config.CAM_READ_RGB / cam_rgb_consumers()。
        rgb = None
        if CAM_READ_RGB:
            rgb = self.env.camera_color_tensor.detach().cpu().numpy()
        self.env.gym.end_access_image_tensors(self.env.sim)

        depth = np.nan_to_num(depth, 0.0)
        # 裁剪上限与 BEV_MAX_DEPTH 一致（BEV 建图也按同一上限过滤无效深度）
        depth_pos = np.clip(-depth, 0.0, BEV_MAX_DEPTH)
        # 缓存给像素级前向净空判据用（BEVMapper.forward_obstacle_distance）。
        # 感知降频时中间步不重新渲染，沿用这张即可 —— 与 BEV 障碍图的复用节奏一致。
        self._last_depth_pos = depth_pos
        if rgb is not None:
            rgb = rgb[..., :3].astype(np.uint8)

        # === BEV Height Map ===
        # 开关见 viz_config.BEV_ENABLE_UPDATE。
        # ⚠ 原代码这一行被注释掉，导致 BEV 障碍图恒为空（上层看不到障碍）。
        #   默认仍保持原行为；改成 True 才会真正把深度图喂进 BEV 建图。
        if BEV_ENABLE_UPDATE:
            self.bev.update(depth_pos)
        # rgb=None 表示本轮没拷 RGB（CAM_READ_RGB=False）。depth 存图仍要照常执行，
        # 只有依赖 RGB 的那两个分支各自跳过（见 _save_camera_images / _show_camera_images）。
        self._handle_camera_images(step, rgb, depth_pos)
        return True

    def run(self, num_trials):
        success = 0
        success_times = []
        if SCENE_RANDOMIZE_OBSTACLES_EACH_TRIAL and int(num_trials) > 1:
            print("[场景提示] GPU PhysX 静态障碍在进程创建时固化；本进程的 %d 个 trial "
                  "共用 seed=%d 的障碍布局。需要逐轮随机时，请每轮用不同的 "
                  "HIMLOCO_SCENE_SEED 重新启动进程。"
                  % (int(num_trials), int(SCENE_RANDOM_SEED_BASE)))
        if LIVE_SHOW_CAMERA_IMAGES or LIVE_SHOW_BEV_FMM_WINDOW:
            plt.ion()
        for i in range(num_trials):
            trial_id = i + 1
            try:
                ok, t = self.run_single_trial(trial_id)
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt: evaluation interrupted.")
                break
            except MemoryGuardExceeded as mem_err:
                # 主动中止而非拖死整机；已完成的 trial 指标照常汇总输出
                print("\n⛔ " + str(mem_err))
                print("   已完成的 trial 结果仍会照常汇总（见下方 Evaluation Result）。")
                break
        # 打印本轮结果
            if len(self.stats["smoothness"]) > 0:
                s = self.stats["smoothness"][-1]
                m = self.stats["time_ms"][-1]
                p = self.stats["progress"][-1] * 100
                print(f"📊 Trial {i+1} 实时指标: 进度={p:.0f}%, 平滑度={s:.4f} rad, 平均规划={m:.2f} ms")
            if ok:
                success += 1
                success_times.append(t)
        if SAVE_SUMMARY_TRAJECTORY_PLOT:
            os.makedirs(SAVE_SUMMARY_TRAJECTORY_PLOT_DIR, exist_ok=True)
            plot_all_saved_trajectories(
                folder_path=os.path.abspath(self.save_trajectory_dir),
                goal_xy_all=self.goal_xy_all,
                save_path=os.path.join(SAVE_SUMMARY_TRAJECTORY_PLOT_DIR, "all_saved_trajectory.png")
            )
        print("\n========== Evaluation Result ==========")
        print(f"Total trials : {num_trials}")
        print(f"Successes   : {success}")
        print(f"Success rate: {success / num_trials:.2f}")
        return success / num_trials

    def _handle_fail(self, reason, elapsed_time):
        print(f"❌ Fail: {reason}")
        print(f"Time used: {elapsed_time:.2f} seconds")
        print(f"Total Distance Walked : {self.trial_real_distance:.3f} m")
        metrics = self._compute_metrics()
        real_avg_speed = self.trial_real_distance / elapsed_time
        print(f"Real Average Speed : {real_avg_speed:.3f} m/s")
        print(f"Avg Torque Cost      : {metrics['energy']:.3f}")
        print(f"Orientation Error     : {metrics['orientation_error']:.3f} rad")
        print(f"Angular Vel Error     : {metrics['angular_vel_error']:.3f} rad/s")
        print(f"Command Wz Variation  : {metrics['cmd_wz_variation']:.4f} rad/s/step")
        print(f"Command Wz Sign Flips : {metrics['cmd_wz_sign_flips']}")
        self._print_narrow_passage_stats()
        self._write_trial_result(False, reason, elapsed_time, metrics)
        return False, elapsed_time
    
    def _handle_success(self, elapsed_time):
        print("✅ Success: reached all waypoints")
        print(f"Time used: {elapsed_time:.2f} seconds")
        print(f"Total Distance Walked : {self.trial_real_distance:.3f} m")
        metrics = self._compute_metrics()
        real_avg_speed = self.trial_real_distance / elapsed_time
        print(f"Real Average Speed : {real_avg_speed:.3f} m/s")
        print(f"Avg Torque Cost      : {metrics['energy']:.3f}")
        print(f"Orientation Error     : {metrics['orientation_error']:.3f} rad")
        print(f"Angular Vel Error     : {metrics['angular_vel_error']:.3f} rad/s")
        print(f"Command Wz Variation  : {metrics['cmd_wz_variation']:.4f} rad/s/step")
        print(f"Command Wz Sign Flips : {metrics['cmd_wz_sign_flips']}")
        self._print_narrow_passage_stats()
        self._write_trial_result(True, "success", elapsed_time, metrics)
        return True, elapsed_time

    def _write_trial_result(self, success, reason, elapsed_time, metrics):
        if not RESULT_JSON_PATH:
            return
        pos = np.asarray(self.trial_vio_position_errors, dtype=np.float64)
        yaw = np.asarray(self.trial_vio_yaw_errors, dtype=np.float64)
        plan = np.asarray(self.trial_step_times, dtype=np.float64)
        smooth = np.asarray(self.trial_step_smoothness, dtype=np.float64)
        record = {
            "success": bool(success),
            "failure_reason": str(reason),
            "scene": ACTIVE_SCENE_PRESET,
            "scene_seed": int(SCENE_RANDOM_SEED_BASE),
            "planner_mode": self.mode,
            "temporal_memory": bool(self._use_temporal_memory),
            "soft_cost": bool(FMM_USE_SOFT_COST),
            "actor_truth_whitelist": bool(BEV_OBSTACLE_ACTORS_ONLY),
            "vio_noise_level": VIO_NOISE_LEVEL,
            "elapsed_s": float(elapsed_time),
            "distance_m": float(self.trial_real_distance),
            "planning_mean_ms": float(plan.mean()) if plan.size else 0.0,
            "planning_p95_ms": float(np.percentile(plan, 95)) if plan.size else 0.0,
            "path_smoothness_mean": float(smooth.mean()) if smooth.size else None,
            "energy_torque_sq_mean": float(metrics["energy"]),
            "cmd_wz_variation": float(metrics["cmd_wz_variation"]),
            "cmd_wz_sign_flips": int(metrics["cmd_wz_sign_flips"]),
            "orientation_error_rad": float(metrics["orientation_error"]),
            "angular_vel_error_rad_s": float(metrics["angular_vel_error"]),
            "narrow_steps": int(self.trial_narrow_steps),
            "min_path_clearance_m": (
                float(self.trial_min_path_clearance)
                if np.isfinite(self.trial_min_path_clearance) else None),
            "memory_max_voxels": int(self.trial_memory_max_points),
            "body_contact_steps": int(self.trial_base_contact_steps),
            "side_contact_steps": int(self.trial_side_contact_steps),
            "peak_body_contact_n": float(self.trial_max_base_contact_force),
            "vio_position_rms_m": (
                float(np.sqrt(np.mean(pos ** 2))) if pos.size else 0.0),
            "vio_yaw_rms_deg": (
                float(np.rad2deg(np.sqrt(np.mean(yaw ** 2)))) if yaw.size else 0.0),
            "false_arrival_true_error_m": (
                float(self.trial_false_arrival_error)
                if self.trial_false_arrival_error is not None else None),
            "final_x_m": float(self.env.root_states[0, 0].item()),
            "final_y_m": float(self.env.root_states[0, 1].item()),
            "waypoint_progress": float(
                self.planner.current_id / max(1, len(self.planner.waypoints))),
            "stall_escape_count": int(self._stall_escape_count),
        }
        record.update(self._obstacle_layout_metadata())
        output_path = os.path.abspath(RESULT_JSON_PATH)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2, sort_keys=True)

    def _print_narrow_passage_stats(self):
        clearance = self.trial_min_path_clearance
        clearance_text = "n/a" if not np.isfinite(clearance) else "%.3f m" % clearance
        print("Narrow Passage       : %d steps, min path clearance %s"
              % (self.trial_narrow_steps, clearance_text))
        if self._use_temporal_memory:
            print("Temporal Memory      : max %d world voxels"
                  % self.trial_memory_max_points)
        if self.trial_vio_position_errors:
            pos = np.asarray(self.trial_vio_position_errors, dtype=np.float64)
            yaw = np.asarray(self.trial_vio_yaw_errors, dtype=np.float64)
            print("VIO Pose Error       : rms %.3f m / %.2f deg, max %.3f m / %.2f deg"
                  % (float(np.sqrt(np.mean(pos ** 2))),
                     float(np.rad2deg(np.sqrt(np.mean(yaw ** 2)))),
                     float(np.max(pos)), float(np.rad2deg(np.max(yaw)))))
        print("Body Contact         : %d steps, side %d steps, peak %.1f N"
              % (self.trial_base_contact_steps, self.trial_side_contact_steps,
                 self.trial_max_base_contact_force))

    def _update_contact_metrics(self):
        """Record simulator contact truth for evaluation only, never for planning."""
        try:
            indices = self.env.termination_contact_indices
            if indices.numel() == 0:
                return
            forces = self.env.contact_forces[0, indices, :]
            magnitude = torch.norm(forces, dim=-1)
            peak = float(magnitude.max().item())
            self.trial_max_base_contact_force = max(
                self.trial_max_base_contact_force, peak)
            threshold = float(getattr(
                getattr(self.env.cfg, "termination", None),
                "collision_force_threshold", 1.0))
            if bool(torch.any(magnitude > threshold)):
                self.trial_base_contact_steps += 1
            horizontal = torch.norm(forces[:, :2], dim=-1)
            if bool(torch.any(horizontal > threshold)):
                self.trial_side_contact_steps += 1
        except Exception:
            # Metrics must never alter navigation behavior.
            return


    def _get_base_yaw(self, env_id=0):
        base_quat = self.env.base_quat[env_id:env_id+1]
        forward = quat_apply(
            base_quat,
            self.env.forward_vec[env_id:env_id+1]
        )[0]
        return torch.atan2(forward[1], forward[0]).item()

    def _draw_waypoints(self):
        sphere_radius = DRAW_WAYPOINT_RADIUS_M
        for idx, wp in enumerate(self.planner.waypoints):
            if self.planner.visited[idx]:
                color = (0, 1, 0)   
            elif idx == self.planner.current_id:
                color = (1, 1, 0)   
            else:
                color = (1, 0, 0)  
            sphere_geom = gymutil.WireframeSphereGeometry(
                radius=sphere_radius,
                num_lats=8,
                num_lons=8,
                pose=None,
                color=color 
            )
            pose = gymapi.Transform()
            pose.p = self._get_draw_position(wp[0], wp[1], sphere_radius,
                                             clearance=DRAW_WAYPOINT_CLEARANCE_M)
            gymutil.draw_lines(
                sphere_geom,
                self.env.gym,
                self.env.viewer,
                self.env.envs[0],
                pose
            )

    def _get_draw_position(self, x, y, radius, clearance=DRAW_DEFAULT_CLEARANCE_M):
        if hasattr(self.env, "_get_terrain_height_at"):
            terrain_z = self.env._get_terrain_height_at(float(x), float(y))
        else:
            terrain_z = 0.0
        draw_z = terrain_z + float(radius) + float(clearance)
        return gymapi.Vec3(float(x), float(y), float(draw_z))

    def _draw_traj(self):
        if len(self.traj_xy_cur) < 1:
            return

        sphere_radius = DRAW_TRAJ_RADIUS_M
        step_skip = DRAW_TRAJ_STEP_SKIP
        # 原实现每步重画【全部】历史轨迹点：轨迹越长每步越慢（O(N²)），
        # 1500 步时每步要画 ~187 个球 × 72 条线 ≈ 1.3 万条线段，远程桌面下直接卡死。
        # 这里按上限自动放大抽样间隔，把球数压到 LIVE_DRAW_TRAJ_MAX_POINTS 以内。
        max_pts = int(LIVE_DRAW_TRAJ_MAX_POINTS)
        if max_pts > 0:
            n_pts = len(range(0, len(self.traj_xy_cur), step_skip))
            if n_pts > max_pts:
                step_skip = max(1, int(len(self.traj_xy_cur) / max_pts))

        sphere_geom = gymutil.WireframeSphereGeometry(
            radius=sphere_radius,
            num_lats=6,
            num_lons=6,
            pose=None,
            color=(0, 1, 0)   # 绿色：已走过轨迹
        )

        for i in range(0, len(self.traj_xy_cur), step_skip):
            p = self.traj_xy_cur[i]

            pose = gymapi.Transform()
            pose.p = self._get_draw_position(p[0], p[1], sphere_radius,
                                             clearance=DRAW_TRAJ_CLEARANCE_M)

            gymutil.draw_lines(
                sphere_geom,
                self.env.gym,
                self.env.viewer,
                self.env.envs[0],
                pose
            )
    def _draw_fmm_path_spheres(self):
        rc_path = self.fmm.extract_path()
        if len(rc_path) < 2:
            return

        # ===== 参数（在 viz_config.py 的 DRAW_FMM_PATH_* 里调）=====
        sphere_radius = DRAW_FMM_PATH_RADIUS_M   # 控制“粗细”
        step_skip = DRAW_FMM_PATH_STEP_SKIP      # 每隔几个点画一个（降低开销）
        # 同 _draw_traj：按上限压住球数，避免路径长时绘制开销线性膨胀
        max_pts = int(LIVE_DRAW_FMM_PATH_MAX_POINTS)
        if max_pts > 0:
            n_pts = len(range(0, len(rc_path), step_skip))
            if n_pts > max_pts:
                step_skip = max(1, int(len(rc_path) / max_pts))

        # ===== 当前机器人位姿 =====
        base_pos = self.env.root_states[0, :3].cpu().numpy()
        yaw = self._get_base_yaw(0)

        c_y = np.cos(yaw)
        s_y = np.sin(yaw)

        # ===== 创建一个球的几何体（复用，性能更好）=====
        sphere_geom = gymutil.WireframeSphereGeometry(
            radius=sphere_radius,
            num_lats=6,
            num_lons=6,
            color=(0, 0, 1)   # 🔴 红色路径
        )

        # ===== 遍历路径点 =====
        for i in range(0, len(rc_path), step_skip):
            r, c = rc_path[i]

            # --- BEV → robot frame ---
            x = r * self.fmm.bev_res
            y = (c - self.fmm.W // 2) * self.fmm.bev_res

            # --- robot → world ---
            wx = base_pos[0] + c_y * x - s_y * y
            wy = base_pos[1] + s_y * x + c_y * y

            # --- 设置球位置 ---
            pose = gymapi.Transform()
            pose.p = self._get_draw_position(wx, wy, sphere_radius,
                                             clearance=DRAW_FMM_PATH_CLEARANCE_M)

            # --- 画球 ---
            gymutil.draw_lines(
                sphere_geom,
                self.env.gym,
                self.env.viewer,
                self.env.envs[0],
                pose
            )
    def _calculate_smoothness(self, path_rc):
            # 如果路径点太少，返回一个标识值，后续会被过滤掉
            if path_rc is None or len(path_rc) < METRIC_SMOOTH_MIN_POINTS:
                return 0.0 # 或者返回 -1.0 
            
            # 与 _record 路径长度处保持一致：栅格坐标 -> 米，跟着 BEV_RES 走。
            # 角度本身是尺度无关的，但下面 norms 的阈值判断依赖实际米数。
            pts = np.array(path_rc) * BEV_RES
            vecs = np.diff(pts, axis=0)
            
            # 防止分母为 0 的微小位移检查
            norms = np.linalg.norm(vecs, axis=1)
            if np.sum(norms) < METRIC_SMOOTH_MIN_PATH_LEN_M: return 0.0

            angles = np.arctan2(vecs[:, 1], vecs[:, 0])
            angle_diffs = np.abs(np.diff(angles))
            angle_diffs = np.where(angle_diffs > np.pi, 2*np.pi - angle_diffs, angle_diffs)
            
            return np.mean(angle_diffs)

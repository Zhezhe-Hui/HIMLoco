from legged_gym import LEGGED_GYM_ROOT_DIR
import os
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
        )
        self.last_occ = None
        self.mode = mode
        if self.mode == 'fmm':
            self.fmm = FMMGradientController(
                bev_res=BEV_RES, bev_x=BEV_X, bev_y=BEV_Y,
                max_wz=FMM_MAX_WZ,
                yaw_k=FMM_YAW_K,
                lookahead_m=FMM_LOOKAHEAD_M,
                inflate_radius_m=FMM_INFLATE_RADIUS_M,
                goal_min_forward_m=FMM_GOAL_MIN_FORWARD_M,
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
        self.planner.reset()
        self.env.reset()
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
   
            # ===== A. 时间融合（在 update BEV 之前）=====
            if prev_base_pos is not None:
                dx = base_pos[0] - prev_base_pos[0]
                dy = base_pos[1] - prev_base_pos[1]
                self.bev.shift_with_motion(dx, dy, prev_base_yaw)
            base_xy = base_pos[:2].copy()
            if self.trial_start_xy is None:
                self.trial_start_xy = base_xy.copy()
            self.traj_xy_cur.append(base_xy)
            # ========== 新增：每一步真实行走距离 ==========
            if prev_base_pos is not None:
                dx = base_pos[0] - prev_base_pos[0]
                dy = base_pos[1] - prev_base_pos[1]
                step_dist = np.hypot(dx, dy)
                self.trial_real_distance += step_dist
            # ==============================================
            prev_base_pos = base_pos.copy()
            prev_base_yaw = base_yaw

            # ===== 4. 更新 BEV =====
            self._update_camera_and_bev(step)
            # ===== 5. 唯一的 wz：BEV + goal =====
            occ_bev = self.bev.get_obstacle_map()

            # 3) 计算 waypoint 在机器人坐标（goal_xy_bev）
            wp = self.planner.waypoints[self.planner.current_id]
            dx_w = wp[0] - base_pos[0]
            dy_w = wp[1] - base_pos[1]
            c = np.cos(-base_yaw); s = np.sin(-base_yaw)
            dx_r = c * dx_w - s * dy_w   # forward
            dy_r = s * dx_w + c * dy_w   # left

            # 4) FMM
            self.fmm.set_goal((dx_r, dy_r))
            start_plan_t = time.time()
            #计算距离场保存
            self.fmm.update(occ_bev)
            # ===== 1. 计时规划 (所有算法通用) =====
            self.trial_step_times.append((time.time() - start_plan_t) * 1000) # 存入 ms
            # ===== 1. 提取 FMM 原始路径 =====
            raw_path = self.fmm.extract_path()
            self._visualize_bev_fmm(step, trial_id, occ_bev, raw_path)

            # ===== 2. FMM路径（robot → world）=====
            raw_xy = []
            cos_yaw = np.cos(base_yaw)
            sin_yaw = np.sin(base_yaw)
            for r, col in raw_path:
                # BEV → robot frame
                x = r * self.fmm.bev_res
                y = (col - self.fmm.W // 2) * self.fmm.bev_res
                # robot → world
                wx = base_pos[0] + cos_yaw * x - sin_yaw * y
                wy = base_pos[1] + sin_yaw * x + cos_yaw * y
                raw_xy.append([wx, wy])
            raw_xy = np.array(raw_xy)
            self._record_trial_fmm_segment(step, raw_xy)

            # ===== 4. 保存 =====
            if step == 0: 
                if raw_xy is not None and len(raw_xy) > 0:
                    self.raw_path_all.append(raw_xy)

            wz = self.fmm.compute_wz()

            # ===== 轨迹优化（仅用于可视化 / 分析 / 未来 MPC）=====
            # smooth_path = self.fmm.optimize_fmm_path(
            #     waypoints=self.planner.waypoints
            # )
            # --- 增加以下记录逻辑 ---
            current_smooth = self._calculate_smoothness(raw_path)
            # 只有当路径有效（不是兜底值或 0）时才加入统计
            if METRIC_SMOOTH_VALID_MIN < current_smooth < METRIC_SMOOTH_VALID_MAX:
                self.trial_step_smoothness.append(current_smooth)
            # ===== waypoint 切换 =====
            if self.fmm.is_goal_reached(thresh_m=NAV_GOAL_REACH_THRESH_M):
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
                    self.stats["success_count"] += 1
                    return self._handle_success(elapsed_time)
                else:
                    reason = "Environment Termination" if dones[0] else "Timeout"
                    status = reason.lower().replace(" ", "_")
                    self._save_trial_route_plot(trial_id, status)
                    return self._handle_fail(reason, elapsed_time)

            # 下发动作：vx 固定前进，wz 由高层规划器给出（速度见 viz_config.NAV_FORWARD_VX）
            self.env.commands[:] = 0.0
            self.env.commands[0, 0] = NAV_FORWARD_VX
            self.env.commands[0, 2] = wz
            self._update_live_follow_camera()
            obs, _, _, dones, infos, _, _ = self.env.step(actions)
            # 数据记录
            real_vx = self.env.base_lin_vel[0, 0].detach().item()
            torque = self.env.torques[0].detach().cpu().numpy()
            torque_cost = np.sum(torque ** 2)
            self.state_log.append({
                "cmd_vx": float(self.env.commands[0, 0].item()),
                "real_vx": real_vx,
                "torque_cost": torque_cost
            })
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
            self.state_log.append({
                "cmd_vx": float(self.env.commands[0, 0].item()),
                "real_vx": real_vx,
                "torque_cost": torque_cost
            })

            # 失败：环境终止
            if dones[0]:
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
                return self._handle_fail("environment termination", elapsed_time)

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

        # 稳定性
        stab = self._compute_stability_errors()

        return {
            "vel_tracking_mae": float(vel_tracking_mae),
            "orientation_error": stab["orientation_error"],
            "angular_vel_error": stab["angular_vel_error"],
            "energy": float(energy),
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

        roll_list = []
        pitch_list = []
        wx_list = []
        wy_list = []

        for i in range(len(self.state_log)):
            # 当前姿态
            quat = self.env.base_quat[0]
            roll, pitch, _ = self._quat_to_rpy(quat)

            roll_list.append(roll)
            pitch_list.append(pitch)

            # 当前角速度
            wx = self.env.base_ang_vel[0, 0].item()
            wy = self.env.base_ang_vel[0, 1].item()
            wx_list.append(wx)
            wy_list.append(wy)

        roll_arr = np.array(roll_list)
        pitch_arr = np.array(pitch_list)
        wx_arr = np.array(wx_list)
        wy_arr = np.array(wy_list)

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
        if SAVE_RGB_IMAGE:
            rgb_path = os.path.join(SAVE_CAMERA_IMAGE_DIR, f"rgb_{step:06d}.png")
            plt.imsave(rgb_path, rgb)

        if SAVE_DEPTH_IMAGE:
            depth_path = os.path.join(SAVE_CAMERA_IMAGE_DIR, f"depth_{step:06d}.png")
            plt.imsave(depth_path, depth_pos, cmap="gray", vmin=0.0, vmax=6.0)

    def _show_camera_images(self, step, rgb, depth_pos):
        if not LIVE_SHOW_CAMERA_IMAGES or step % LIVE_CAMERA_IMAGE_INTERVAL != 0:
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
                    
    def _update_camera_and_bev(self, step):

        # 无相机（headless 跑 train、或 cam 创建失败）时直接跳过，
        # 否则下面 set_camera_location(None, ...) 会报错。
        # 与 _capture_fixed_overhead_image 的守卫保持一致。
        if self.env.cam_handle is None or self.env.camera_depth_tensor is None:
            return

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

        cam_target = gymapi.Vec3(
            base_pos[0] + CAM_FOLLOW_LOOKAT_FORWARD * fx,
            base_pos[1] + CAM_FOLLOW_LOOKAT_FORWARD * fy,
            base_pos[2] + CAM_FOLLOW_HEIGHT
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
            return
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
        rgb   = self.env.camera_color_tensor.detach().cpu().numpy()
        self.env.gym.end_access_image_tensors(self.env.sim)

        depth = np.nan_to_num(depth, 0.0)
        # 裁剪上限与 BEV_MAX_DEPTH 一致（BEV 建图也按同一上限过滤无效深度）
        depth_pos = np.clip(-depth, 0.0, BEV_MAX_DEPTH)
        rgb = rgb[..., :3]
        rgb = rgb.astype(np.uint8)

        # === BEV Height Map ===
        # 开关见 viz_config.BEV_ENABLE_UPDATE。
        # ⚠ 原代码这一行被注释掉，导致 BEV 障碍图恒为空（上层看不到障碍）。
        #   默认仍保持原行为；改成 True 才会真正把深度图喂进 BEV 建图。
        if BEV_ENABLE_UPDATE:
            self.bev.update(depth_pos)
        self._handle_camera_images(step, rgb, depth_pos)

    def run(self, num_trials):
        success = 0
        success_times = []
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
        return True, elapsed_time


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

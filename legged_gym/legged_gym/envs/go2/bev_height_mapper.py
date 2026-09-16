# legged_gym/envs/go2/bev_mapper.py
import numpy as np
import matplotlib.pyplot as plt
import skfmm
from numpy import ma
import skimage.morphology as morph
from legged_gym.envs.go2.go2_config import Go2RoughCfg
import matplotlib.image as mpimg
import glob
import os

class BEVMapper:
    """
    Scheme A: Height range BEV mapper (FMM-style)

    Depth -> BEV min/max height -> obstacle if height range large
    """

    def __init__(
        self,
        bev_x=6.0,          # 前方范围 (m)
        bev_y=6.0,          # 左右范围 (m)
        bev_res=0.05,       # 分辨率 (m)
        cam_height=0.25,    # 相机高度 (m)
        hfov_deg=90.0,
        max_depth=6.0,
        v_start_ratio=0.55,
        height_range_thresh=0.25,  #  关键阈值
        min_points=8,
        # 以下参数由 viz_config.py 第十四节统一配置
        min_depth=0.05,             # 最小有效深度 (m)
        enable_memory_fusion=False, # 是否用时序融合（即时 ∪ 历史记忆）
        memory_max_age=30,          # 记忆寿命（帧）
    ):
        self.bev_x = bev_x
        self.bev_y = bev_y
        self.bev_res = bev_res
        self.cam_height = cam_height
        self.hfov_deg = hfov_deg
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.v_start_ratio = v_start_ratio
        self.height_range_thresh = height_range_thresh
        self.min_points = min_points
        self.enable_memory_fusion = bool(enable_memory_fusion)

        self.H = int(bev_x / bev_res)
        self.W = int(bev_y / bev_res)
        # === 新增：局部记忆障碍图（叠加式）===
        self.occ_memory = np.zeros((self.H, self.W), dtype=np.uint8)
        self.occ_age = np.zeros((self.H, self.W), dtype=np.int32)

        # 记忆寿命（单位：帧）
        self.max_age = int(memory_max_age)

        self.reset()

        # vis
        self.fig = None
        self.im0 = None
        self.im1 = None

    def reset(self):
        self.min_z = np.full((self.H, self.W), np.inf, dtype=np.float32)
        self.max_z = np.full((self.H, self.W), -np.inf, dtype=np.float32)
        self.count = np.zeros((self.H, self.W), dtype=np.int32)

    @staticmethod
    def _compute_fx_fy(W, H, hfov_rad):
        fx = W / (2.0 * np.tan(hfov_rad / 2.0))
        vfov = 2.0 * np.arctan(np.tan(hfov_rad / 2.0) * (H / W))
        fy = H / (2.0 * np.tan(vfov / 2.0))
        return fx, fy

    def update(self, depth_m: np.ndarray):
        self.reset()

        H_img, W_img = depth_m.shape
        cx = (W_img - 1) * 0.5
        cy = (H_img - 1) * 0.5

        hfov = np.deg2rad(self.hfov_deg)
        fx, fy = self._compute_fx_fy(W_img, H_img, hfov)

        v0 = int(self.v_start_ratio * H_img)

        d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        d[(d < self.min_depth) | (d > self.max_depth)] = 0.0

        # 只取下半部分图像
        d_roi = d[v0:H_img, :]

        # 找有效深度点
        valid = d_roi > 0.0
        if not np.any(valid):
            return

        v_idx, u_idx = np.nonzero(valid)
        depth = d_roi[v_idx, u_idx]

        # 注意 v_idx 是 roi 内部索引，要加回 v0
        v = v_idx + v0
        u = u_idx

        du = (u - cx) / fx
        dv = (v - cy) / fy

        x = depth
        y = du * depth
        z = self.cam_height - dv * depth

        ix = (x / self.bev_res).astype(np.int32)
        iy = ((y + self.bev_y * 0.5) / self.bev_res).astype(np.int32)

        inside = (
            (ix >= 0) & (ix < self.H) &
            (iy >= 0) & (iy < self.W)
        )

        if not np.any(inside):
            return

        ix = ix[inside]
        iy = iy[inside]
        z = z[inside]

        # 对同一个 BEV cell 聚合 min_z / max_z / count
        np.minimum.at(self.min_z, (ix, iy), z)
        np.maximum.at(self.max_z, (ix, iy), z)
        np.add.at(self.count, (ix, iy), 1)

    def update_memory(self, occ_now):
        """
        occ_now: 当前帧的二值障碍图 (0/1)
        """

        # 1) 所有已有记忆老化
        self.occ_age[self.occ_memory > 0] += 1

        # 2) 当前观测到的障碍，直接写入记忆
        new_obs = occ_now > 0
        self.occ_memory[new_obs] = 1
        self.occ_age[new_obs] = 0

        # 3) 超过寿命的障碍清除
        expired = self.occ_age > self.max_age
        self.occ_memory[expired] = 0
        self.occ_age[expired] = 0

    def get_obstacle_map(self):
        """
        返回给规划器用的障碍图：
        occ_for_planner = 即时障碍 ∪ 记忆障碍
        """

        # 1) 当前帧障碍
        occ_now = np.zeros((self.H, self.W), dtype=np.uint8)

        valid = self.count >= self.min_points
        height_range = self.max_z - self.min_z
        occ_now[valid & (height_range > self.height_range_thresh)] = 1

        # 2) 更新记忆
        self.update_memory(occ_now)

        # 3) 给规划器的障碍 = 即时 ∪ 记忆（时序融合）；关闭时只用当前帧
        #    开关来自 viz_config.BEV_ENABLE_MEMORY_FUSION，不要再靠注释切换
        if self.enable_memory_fusion:
            return np.maximum(occ_now, self.occ_memory)
        return occ_now

    def shift_with_motion(self, dx_w, dy_w, yaw):
        """
        dx_w, dy_w : world frame 位移
        yaw        : 当前机器人 yaw
        """

        # world -> robot frame
        c = np.cos(-yaw)
        s = np.sin(-yaw)
        dx_r = c * dx_w - s * dy_w   # forward
        dy_r = s * dx_w + c * dy_w   # left

        shift_x = int(np.round(dx_r / self.bev_res))
        shift_y = int(np.round(dy_r / self.bev_res))

        if shift_x == 0 and shift_y == 0:
            return

        self.occ_memory = np.roll(self.occ_memory, -shift_x, axis=0)
        self.occ_memory = np.roll(self.occ_memory, -shift_y, axis=1)

        self.occ_age = np.roll(self.occ_age, -shift_x, axis=0)
        self.occ_age = np.roll(self.occ_age, -shift_y, axis=1)

        # 新区域清空
        if shift_x > 0:
            self.occ_memory[-shift_x:, :] = 0
            self.occ_age[-shift_x:, :] = 0
        elif shift_x < 0:
            self.occ_memory[:-shift_x, :] = 0
            self.occ_age[:-shift_x, :] = 0

        if shift_y > 0:
            self.occ_memory[:, -shift_y:] = 0
            self.occ_age[:, -shift_y:] = 0
        elif shift_y < 0:
            self.occ_memory[:, :-shift_y] = 0
            self.occ_age[:, :-shift_y] = 0

    def visualize(self):
        hr = np.clip(self.max_z - self.min_z, 0.0, 1.0)
        occ = self.get_obstacle_map()

        if self.fig is None:
            plt.ion()
            self.fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 4))
            self.im0 = ax0.imshow(hr, origin="lower", cmap="viridis")
            ax0.set_title("BEV Height Range")
            self.im1 = ax1.imshow(
                                    occ,
                                    origin="lower",
                                    cmap="gray",
                                    vmin=0,
                                    vmax=1,
                                    interpolation="nearest"   # ⭐ 必须加
                                )

            ax1.set_title("BEV Obstacle Map")
        else:
            self.im0.set_data(hr)
            self.im1.set_data(occ)

        plt.pause(0.001)
        
class FMMGradientController:
    """
    Local FMM gradient follower (goal + obstacles) -> wz
    - occ_bev: 1 obstacle / 0 free
    - goal_xy_bev: (x forward, y left) in meters (robot frame)
    """

    def __init__(
        self,
        bev_res=0.05,
        bev_x=6.0,
        bev_y=6.0,
        max_wz=1.0,
        yaw_k=2.0,              # 把“期望方向角”转成 wz 的比例
        lookahead_m=0.6,        # 在前方 lookahead 处取梯度，避免 r=0 视野太近导致 dc≈0
        inflate_radius_m=0.15,  # 障碍膨胀（接近原项目的 config space 思想）
        goal_min_forward_m=0.10,  # goal 在身后时钳到前方的最小值
    ):
        self.bev_res = float(bev_res)
        self.bev_x = float(bev_x)
        self.bev_y = float(bev_y)
        self.max_wz = float(max_wz)
        self.yaw_k = float(yaw_k)
        self.lookahead_m = float(lookahead_m)
        self.inflate_radius_m = float(inflate_radius_m)
        self.goal_min_forward_m = float(goal_min_forward_m)

        self.H = int(self.bev_x / self.bev_res)
        self.W = int(self.bev_y / self.bev_res)

        self.large = 1e6
        self.goal_rc = None
        self.fmm_dist = None

        # 机器人在 BEV 的“起点”索引：r=0, c=中间
        self.robot_r = 0
        self.robot_c = self.W // 2

    def set_goal(self, goal_xy_bev):
        gx_m, gy_m = float(goal_xy_bev[0]), float(goal_xy_bev[1])

        # 局部规划：goal 在身后会导致梯度怪象，钳到前方一点
        gx_m = max(gx_m, self.goal_min_forward_m)

        gr = int(gx_m / self.bev_res)
        gc = int(gy_m / self.bev_res) + (self.W // 2)

        gr = int(np.clip(gr, 0, self.H - 1))
        gc = int(np.clip(gc, 0, self.W - 1))

        self.goal_rc = (gr, gc)

    def _inflate(self, occ_bev):
        if self.inflate_radius_m <= 1e-6:
            return occ_bev
        rad = int(np.round(self.inflate_radius_m / self.bev_res))
        if rad <= 0:
            return occ_bev
        selem = morph.disk(rad)
        inflated = morph.binary_dilation(occ_bev.astype(bool), selem)
        return inflated.astype(np.uint8)

    def update(self, occ_bev):
        if self.goal_rc is None:
            self.fmm_dist = None
            return

        #  1) 膨胀障碍：不然会“擦边撞”
        occ = self._inflate(occ_bev)

        #  2) traversable: 1 free / 0 obstacle
        traversable = (1 - occ).astype(np.float32)

        # masked array：障碍为 mask（不可走）
        trav_ma = ma.masked_values(traversable, 0.0)

        gr, gc = self.goal_rc

        # goal 如果落在障碍里：直接把它钳成 free（否则 skfmm 会返回 mask 常量）
        trav_ma[gr, gc] = 0.0

        dist_ma = skfmm.distance(trav_ma, dx=1.0)
        self.fmm_dist = ma.filled(dist_ma, self.large)

    def compute_wz(self):
        if self.fmm_dist is None:
            return 0.0

        #  在 lookahead 行取梯度（关键）
        r = int(np.clip(self.lookahead_m / self.bev_res, 1, self.H - 2))
        c = int(np.clip(self.robot_c, 1, self.W - 2))

        # central diff
        d_up = self.fmm_dist[r - 1, c]
        d_dn = self.fmm_dist[r + 1, c]
        d_lt = self.fmm_dist[r, c - 1]
        d_rt = self.fmm_dist[r, c + 1]

        grad_r = 0.5 * (d_dn - d_up)
        grad_c = 0.5 * (d_rt - d_lt)

        # negative gradient direction = go “downhill”
        dir_r = -grad_r
        dir_c = -grad_c
        n = float(np.hypot(dir_r, dir_c) + 1e-6)
        dir_r /= n
        dir_c /= n

        # BEV: r ~ x forward, c ~ y left
        desired_yaw = float(np.arctan2(dir_c, dir_r))  # [-pi, pi]
        wz = float(np.clip(self.yaw_k * desired_yaw, -self.max_wz, self.max_wz))

        return wz

    def is_goal_reached(self, thresh_m=0.25):
        """用 dist 场判断是否接近 goal（可选）"""
        if self.fmm_dist is None:
            return False
        thresh_cells = float(thresh_m / self.bev_res)
        return self.fmm_dist[self.robot_r, self.robot_c] < thresh_cells
        
    def extract_path(self, max_steps=300):
        """
        沿 FMM 距离场，从 robot 走到 goal，返回 [(r,c), ...]
        允许 robot_r=0，使用 forward diff / central diff 混合
        """
        if self.fmm_dist is None:
            return []

        r, c = int(self.robot_r), int(self.robot_c)
        H, W = self.fmm_dist.shape
        path = [(r, c)]

        for _ in range(max_steps):
            # 到 goal 附近就停
            if self.fmm_dist[r, c] < 1.0:
                break

            # 越界就停（注意：允许 r==0）
            if r < 0 or r >= H or c < 0 or c >= W:
                break

            # 为了算梯度，至少要保证邻域存在
            if c <= 0 or c >= W - 1:
                break
            if r >= H - 1:  # r+1 要存在
                break

            # ---------- 梯度 ----------
            # r 方向：r==0 时用 forward diff（不能 central）
            if r == 0:
                grad_r = self.fmm_dist[r + 1, c] - self.fmm_dist[r, c]
            else:
                # central diff
                if r <= 0 or r >= H - 1:
                    break
                grad_r = 0.5 * (self.fmm_dist[r + 1, c] - self.fmm_dist[r - 1, c])

            # c 方向：用 central diff（c 保证在 1..W-2）
            grad_c = 0.5 * (self.fmm_dist[r, c + 1] - self.fmm_dist[r, c - 1])

            # 沿负梯度走
            step_r = int(np.sign(-grad_r))
            step_c = int(np.sign(-grad_c))

            if step_r == 0 and step_c == 0:
                break

            r2 = r + step_r
            c2 = c + step_c

            # clamp
            r2 = int(np.clip(r2, 0, H - 1))
            c2 = int(np.clip(c2, 0, W - 1))

            # 如果没有移动，也停
            if r2 == r and c2 == c:
                break

            r, c = r2, c2
            path.append((r, c))

        return path

    def optimize_fmm_path(self, waypoints):
        """
        完整流程：
        FMM path → metric → RDP → Bezier
        返回：robot frame 下的平滑轨迹 (x,y)
        """

        # ===== 提取 FMM 离散路径 =====
        rc_path = self.extract_path()
        if len(rc_path) < 3:
            return None

        # ===== rc → meter (robot frame) =====
        path_xy = []
        for r, c in rc_path:
            x = r * self.bev_res
            y = (c - self.W // 2) * self.bev_res
            path_xy.append([x, y])
        path_xy = np.array(path_xy)

        # =====  RDP 简化 =====
        path_xy = rdp(path_xy, epsilon=0.15)

        # =====  Bezier 平滑（段数 = waypoint 数 - 1）=====
        num_segments = max(len(waypoints) - 1, 1)
        smooth_path = bezier_smooth(
            path_xy,
            num_segments=num_segments,
            samples_per_seg=25
        )

        return smooth_path       

def rdp(points, epsilon):
    """
    Ramer–Douglas–Peucker
    points: (N,2)
    """
    if len(points) < 3:
        return points

    start, end = points[0], points[-1]
    line = end - start
    line_norm = np.linalg.norm(line) + 1e-8

    dmax = 0.0
    idx = 0
    for i in range(1, len(points) - 1):
        v = points[i] - start
        d = np.linalg.norm(np.cross(line, v)) / line_norm
        if d > dmax:
            idx = i
            dmax = d

    if dmax > epsilon:
        left = rdp(points[:idx + 1], epsilon)
        right = rdp(points[idx:], epsilon)
        return np.vstack((left[:-1], right))
    else:
        return np.vstack((start, end))
def bezier_smooth(path, num_segments, samples_per_seg=20):
    """
    用路径关键点做 Bezier 平滑
    num_segments = waypoint 数 - 1
    """
    if len(path) < 4:
        return path

    idxs = np.linspace(0, len(path) - 1, num_segments + 1).astype(int)
    smooth = []

    for i in range(len(idxs) - 1):
        p0 = path[max(idxs[i] - 1, 0)]
        p1 = path[idxs[i]]
        p2 = path[idxs[i + 1]]
        p3 = path[min(idxs[i + 1] + 1, len(path) - 1)]

        for t in np.linspace(0, 1, samples_per_seg):
            pt = (
                (1 - t)**3 * p1 +
                3 * (1 - t)**2 * t * (p1 + 0.3 * (p2 - p0)) +
                3 * (1 - t) * t**2 * (p2 - 0.3 * (p3 - p1)) +
                t**3 * p2
            )
            smooth.append(pt)

    return np.array(smooth)

def plot_all_trials(trajs, goals, raw_paths=None, save_path=None):

    plt.figure(figsize=(10,8))
    # ===== 轨迹 =====
    if trajs is not None:
        for traj in trajs:
            plt.plot(traj[:, 0], traj[:, 1], 'forestgreen', alpha=0.6, label="Robot Trajectory")
    # ===== FMM路径 =====
    if raw_paths is not None:
        for p in raw_paths[:5]:
            plt.plot(p[:, 0], p[:, 1], 'r--', alpha=0.5, label="FMM Raw")
    # ===== 目标点 =====
    if len(goals) > 0:
        G = np.array(goals)
        plt.scatter(G[:, 0], G[:, 1], c='green', s=60, marker='o', label="Waypoints")
    # ===== 障碍 =====
    stone_cfg = Go2RoughCfg.stone
    stones = np.array(stone_cfg.big_stone_positions, dtype=float)
    plt.scatter(stones[:, 0], stones[:, 1], c='darkred', s=200, marker='s', label="Obstacles")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("rrt_Height")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # ===== ✅ 保存 =====
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📸 Saved: {save_path}")
    plt.close()   

def plot_all_saved_trajectories(folder_path, goal_xy_all, save_path=None):
    # 读取所有保存的轨迹
    traj_files = sorted(glob.glob(os.path.join(folder_path, "traj_*.npy")))
    all_trajs = [np.load(f) for f in traj_files]

    plt.figure(figsize=(10, 8))

    # ===== 轨迹（和原函数完全一样）=====
    for traj in all_trajs:
        plt.plot(traj[:, 0], traj[:, 1], 'forestgreen', alpha=0.6, label="Robot Trajectory")

    # ===== 目标点（正常画出来！）=====
    if len(goal_xy_all) > 0:
        G = np.array(goal_xy_all)
        plt.scatter(G[:, 0], G[:, 1], c='green', s=60, marker='o', label="Waypoints")

    # ===== 障碍（完全一样）=====
    stone_cfg = Go2RoughCfg.stone
    stones = np.array(stone_cfg.big_stone_positions, dtype=float)
    plt.scatter(stones[:, 0], stones[:, 1], c='darkred', s=200, marker='s', label="Obstacles")

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("FMM_Obstacles")

    # 图例去重（完全一样）
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # 保存
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📸 Saved: {save_path}")
    plt.close()
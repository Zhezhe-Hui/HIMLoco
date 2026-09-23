# legged_gym/envs/go2/bev_mapper.py
import math
import numpy as np
import matplotlib.pyplot as plt
import skfmm
from numpy import ma
import skimage.morphology as morph
from scipy.ndimage import distance_transform_edt
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
        # 跨格台阶判据：修复“正对垂直墙面/大石头漏检”，见 get_obstacle_map_with_step
        enable_step_judge=True,     # 是否叠加跨格台阶判据
        step_thresh=None,           # 台阶阈值 [m]，None = 复用 height_range_thresh
        step_span=1,                # 与前方第几格比较
        # ---- 相机几何标定结论（2026-09-16 实测，见 viz_config 第十四节）----
        pitch_deg=0.0,              # 光轴俯角 [deg]，向下为正
        lateral_sign=-1.0,          # 横向符号：-1 = y_left = -du*depth（实测成像不镜像）
        use_axial_geom=True,        # True = 按轴向深度做严格针孔反投影
        clip_invalid_eps=1e-3,      # 深度被上游 clip 到 max_depth 的像素视为无效
        min_range=0.0,              # 近距掩蔽 [m]：前向距离小于它的像素直接丢弃
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
        self.pitch_rad = float(np.deg2rad(pitch_deg))
        self.lateral_sign = float(lateral_sign)
        self.use_axial_geom = bool(use_axial_geom)
        self.clip_invalid_eps = float(clip_invalid_eps)
        self.min_range = float(min_range)
        self.enable_memory_fusion = bool(enable_memory_fusion)
        self.enable_step_judge = bool(enable_step_judge)
        self.step_thresh = step_thresh
        self.step_span = int(step_span)

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
        # ⚠ 上游 pathplanner 会把深度 clip 到 [0, max_depth]：天空/超远地面因此
        #   恰好等于 max_depth，而 `d > max_depth` 判不掉它们，会被当成
        #   “6m 处的有效障碍”写进 BEV（实测这是放宽 ROI 后 occ 暴涨的来源之一）。
        #   故把【等于/接近 max_depth】的像素一并视为无效。
        d[(d < self.min_depth) | (d > self.max_depth - self.clip_invalid_eps)] = 0.0

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

        if self.use_axial_geom:
            # IsaacGym IMAGE_DEPTH 实测为【轴向深度】（沿光轴），不是欧氏距离：
            #   地面模型 h/tan(θ) 误差 +0.6%，h/sin(θ) 误差 -1~-9.5%（2026-09-16 标定）。
            # 光轴俯角 p（向下为正）、像素相对光轴偏角 a=atan(dv) 时，严格针孔反投影：
            #   前向距离 x = d*cos(p+a) = d*(cos p - dv*sin p)
            #   向下分量 dz = d*sin(p+a) = d*(sin p + dv*cos p)
            cp, sp = np.cos(self.pitch_rad), np.sin(self.pitch_rad)
            x = depth * (cp - dv * sp)
            z = self.cam_height - depth * (sp + dv * cp)
        else:
            # 原实现（把轴向深度当欧氏距离的近似），保留作对照开关
            x = depth
            z = self.cam_height - dv * depth
        # 横向符号实测标定：单石头定向实验（石头在机器人左侧 1.0m）命中图像左半，
        # 即成像【不镜像】，du>0 = 图像右 = 机器人【右】侧。而 FMM/路径回投影整条链
        # 约定“列号增大 = 左”，所以这里必须取负号，否则障碍图左右镜像、
        # FMM 会朝障碍一侧转向（2026-09-16 实测定位的真 bug）。
        y = self.lateral_sign * du * x

        # 近距掩蔽：俯角机位下最近可见地面可近至 0.35m，该区域混有掠射角深度噪声
        # 与机身自遮挡，会把 clearance 打成 0.3~0.5m 的间歇尖峰（实测 pitch=25 时
        # vx 被反复砍到 0.25、机器狗在碎石坡上失去动量顶死）。机器狗对 <0.5m 的
        # 障碍本来也来不及反应（盲区），故直接丢弃。0 = 不掩蔽（平视原行为）。
        if self.min_range > 0.0:
            keep = x >= self.min_range
            if not np.any(keep):
                return
            x = x[keep]; y = y[keep]; z = z[keep]

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

    def forward_obstacle_distance(self, depth_m, half_angle_deg=25.0,
                                  ground_ratio=0.85, max_range=3.0,
                                  v_start_ratio=None, min_fwd=0.0):
        """直接由深度图算【正前方最近障碍距离】，返回 (距离[m], 命中像素数)。

        为什么需要它（2026-09-16 实测定位的关键缺陷）：
          get_obstacle_map() 的判据是“同一 BEV 格内 max_z-min_z > 阈值”。
          这对【正对的垂直墙面】天然失效 —— 墙面上每个像素的深度几乎相同，
          反投影后落在同一格、高度也几乎相同，格内高度差≈0，于是墙【检不出来】。
          实测（走向 0.76m 高的大石头）：距石 1.76m 时能检出 181 个障碍格，
          但 1.10m 起近距障碍格归 0，前向净空从 0.55m 直接跳回最大值 3.00m，
          于是速度调制误判“前方无障”反而【加速】冲向石头 —— 感知失效被读成通路，
          这是最危险的失效方向。

        判据改为【像素级】：每一行 v 对应一个总俯角（像素偏角 + 光轴俯角），
          若该行是地面，其【轴向深度】应为 cam_height / tan(总俯角)。
          ⚠ 2026-09-16 标定修正：原实现用 sin(θ)（欧氏距离模型），而 IsaacGym
          IMAGE_DEPTH 实测是轴向深度，sin 模型把地面深度系统性算小 1~9.5%，
          导致无障碍地形误报 99%；改 tan 后三张真实平地平地图 0 误报、
          有石图正确报出 2.19m 净空。
          实测深度显著小于该值 => 这个像素打在了【比地面更近的直立物】上。
        这个判据与障碍高度差无关，所以近距、正对墙面都成立，且天然覆盖 0.3m 起。

        v_start_ratio: 本判据专用的纵向 ROI 起始比例，None = 沿用 BEV 建图的
            self.v_start_ratio。建议显式传一个更宽的值（如 0.10）：
            BEV 建图为了不误判地形起伏只看下半张图，那样会连带裁掉高出相机的
            直立障碍 —— 而那恰恰是本判据要抓的东西。

        返回 max_range 表示前方无碍。
        """
        if depth_m is None or depth_m.size == 0:
            return float(max_range), 0

        H_img, W_img = depth_m.shape
        cy = (H_img - 1) * 0.5
        cx = (W_img - 1) * 0.5
        hfov = np.deg2rad(self.hfov_deg)
        fx, fy = self._compute_fx_fy(W_img, H_img, hfov)

        # 只取水平锥内、且 v_start_ratio 以下的像素（上半张是天空/远景，深度不可信）。
        # ⚠ ROI 与 BEV 建图的 self.v_start_ratio【解耦】：
        #   BEV 栅格图喂给 FMM 做转向，它的 ROI 是调过参的（0.55 只看地面，
        #   放宽会把地形起伏/远景误判成障碍，实测论文场景成功率 0.90 -> 0.70）；
        #   而这里的前向净空判据需要看到【高出相机的直立物】，必须用更宽的 ROI。
        #   两个判据目标不同（转向 vs 刹车），各自用最优 ROI，互不干扰。
        roi_start = self.v_start_ratio if v_start_ratio is None else float(v_start_ratio)
        v0 = int(roi_start * H_img)
        half_w = int(W_img * np.tan(np.deg2rad(half_angle_deg))
                     / (2.0 * np.tan(hfov / 2.0)))
        u_lo = max(0, int(cx) - half_w)
        u_hi = min(W_img - 1, int(cx) + half_w)
        if v0 >= H_img or u_lo > u_hi:
            return float(max_range), 0

        rows = np.arange(v0, H_img, dtype=np.float64)
        dv = (rows - cy) / fy
        # 俯角向下为正：只有朝下的行才可能看到地面
        down = dv > 1e-6
        if not np.any(down):
            return float(max_range), 0
        rows = rows[down]
        dv = dv[down]
        theta = np.arctan(dv)                      # 相对光轴的俯角 [rad]
        # 总俯角 = 像素偏角 + 光轴俯角。地面模型必须与深度语义匹配：
        # 实测 IMAGE_DEPTH 是轴向深度，该行若是地面其轴向深度应为 h/tan(总俯角)；
        # 原实现用 h/sin(θ)（欧氏距离），系统性把地面深度算小 1~9.5%，
        # 导致无障碍地形上误报 99%（2026-09-16 标定）。
        total = theta + self.pitch_rad
        keep = total > 1e-6
        if not np.any(keep):
            return float(max_range), 0
        rows = rows[keep]
        dv = dv[keep]
        theta = theta[keep]
        total = total[keep]
        ground_depth = self.cam_height / np.tan(total)   # 该行若是地面应有的轴向深度

        d = np.asarray(depth_m)[v0:H_img, u_lo:u_hi + 1]
        d = d[down, :]
        d = d[keep, :]    # ⚠ 必须与 rows/dv/total 同步过滤，否则行错位
        if d.size == 0:
            return float(max_range), 0
        d = np.nan_to_num(d.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        # 有效深度 且 明显比“该行地面深度”更近 => 命中直立障碍
        valid = (d > self.min_depth) & (d < self.max_depth)
        hit = valid & (d < ground_ratio * ground_depth[:, None])
        if not np.any(hit):
            return float(max_range), 0

        # 命中像素的前向距离 = 轴向深度 * cos(总俯角)，取最近的那个
        hit_depth = d[hit]
        cos_t = np.cos(total)
        # 逐行广播：hit 的行索引 -> cos_t
        row_idx = np.where(hit)[0]
        fwd = hit_depth * cos_t[row_idx]
        # min_fwd：自遮挡带掩蔽。俯角机位下机器狗自身前腿/胸口（离相机 0.35~0.5m）
        # 会进宽 ROI 并被本判据当成直立物，实测把 vx 永久压到最低速、端到端 0/10。
        # 该带内障碍机器狗本来也来不及反应（<0.6m @1m/s 仅 0.6s），故忽略。
        lo = max(0.05, float(min_fwd))
        fwd = fwd[(fwd > lo) & (fwd <= max_range)]
        if fwd.size == 0:
            return float(max_range), 0
        return float(fwd.min()), int(hit.sum())

    def get_obstacle_map_with_step(self, step_thresh=None, step_span=1):
        """障碍图 = 同格高度差判据 ∪ 跨格台阶判据（返回二值 uint8 图）。

        为什么需要第二个判据（2026-09-16 实测定位）：
          原判据是“同一格内 max_z-min_z > height_range_thresh”。
          对【正对的垂直墙面/大石头】它天然失效：墙面上所有像素深度几乎相同，
          反投影后挤在 ix 固定的少数几格里，格内高度差被摊薄到 ≈0。
          实测（8 块挡路大石头、0.76m 高、宽 1.0m）：
            距石 2.04m 时 occ=140 格，1.10m 起 occ→0，前向净空从 0.55m 跳回 3.00m，
            规划器据此认为“前方无障”，wz≈0，机器狗刹住后直直顶着石头磨到 timeout，
            成功率 0/10（这是 blocker 场景全 timeout 的唯一根因）。
          跨格台阶判据：把本格 max_z 与【更近 step_span 格】的 min_z 相比。
            墙面：max_z≈0.76 vs 墙前地面 min_z≈-0.08 -> 台阶 0.84 > 阈值，检出 ✓
            坡地：每格只有 bev_res×坡度 的抬升（10% 坡 = 0.005m/格，30% = 0.015m/格），
                  远小于阈值 0.2m，不会误判 ✓（这正是它比“放宽 ROI”稳的原因：
                  放宽 ROI 会把地形起伏整片判成障碍，实测论文场景 0.90 -> 0.70）

        step_thresh: 台阶阈值 [m]，None = 复用 height_range_thresh
        step_span:   与前方第几格比较，默认 1（相邻格，对坡度最不敏感）
        """
        # 1) 原判据：同格内高度差
        valid = self.count >= self.min_points
        height_range = self.max_z - self.min_z
        occ = np.zeros((self.H, self.W), dtype=np.uint8)
        occ[valid & (height_range > self.height_range_thresh)] = 1

        # 2) 新判据：与更近处的地面比，检测“突然升高的台阶”
        thresh = self.height_range_thresh if step_thresh is None else float(step_thresh)
        span = max(1, int(step_span))
        if span < self.H:
            min_z_safe = np.where(valid, self.min_z, np.nan)
            max_z_safe = np.where(valid, self.max_z, np.nan)
            step_up = max_z_safe[span:, :] - min_z_safe[:-span, :]
            hit = np.zeros_like(step_up, dtype=bool)
            ok = np.isfinite(step_up)
            hit[ok] = step_up[ok] > thresh
            occ[span:, :] |= hit.astype(np.uint8)

        return occ

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

        即时障碍的判据由 enable_step_judge 决定：
          False -> 只有“同格内高度差”（原实现，对正对墙面漏检）
          True  -> 再叠加“跨格台阶”（见 get_obstacle_map_with_step）
        """

        # 1) 当前帧障碍
        if self.enable_step_judge:
            occ_now = self.get_obstacle_map_with_step(
                step_thresh=self.step_thresh, step_span=self.step_span)
        else:
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
        # 硬碰撞层 + 软安全代价层。关闭时完全复现原始固定膨胀 FMM。
        use_soft_cost=False,
        hard_radius_m=None,
        soft_clearance_m=0.65,
        soft_cost_weight=4.0,
        soft_cost_power=2.0,
        # --- 转向决策锁存（修复对称鞍点处 wz 逐步振荡导致的原地打转）---
        hysteresis_thresh=0.0,    # 梯度横向分量小于该值时沿用上次转向符号；0=关闭
        hysteresis_hold_steps=0,  # 转向符号最少保持多少步；0=关闭
        max_wz_accel=0.0,         # 角速度变化率上限 [rad/s^2]；0=不限幅
        control_dt=0.02,
        unreachable_turn_wz=0.6,  # 局部目标不可达时原地搜索的角速度
        use_path_lookahead=True,   # 沿 FMM 路径取前视点，而不是在固定栅格取梯度
        open_space_yaw_deadband=0.02,  # 空旷区目标角死区，抑制步态/定位噪声
    ):
        self.bev_res = float(bev_res)
        self.bev_x = float(bev_x)
        self.bev_y = float(bev_y)
        self.max_wz = float(max_wz)
        self.yaw_k = float(yaw_k)
        self.lookahead_m = float(lookahead_m)
        self.inflate_radius_m = float(inflate_radius_m)
        self.goal_min_forward_m = float(goal_min_forward_m)
        self.use_soft_cost = bool(use_soft_cost)
        self.hard_radius_m = (self.inflate_radius_m if hard_radius_m is None
                              else float(hard_radius_m))
        self.soft_clearance_m = float(soft_clearance_m)
        self.soft_cost_weight = float(soft_cost_weight)
        self.soft_cost_power = float(soft_cost_power)

        if self.hard_radius_m < 0.0:
            raise ValueError("hard_radius_m must be >= 0")
        if self.soft_clearance_m < self.hard_radius_m:
            raise ValueError("soft_clearance_m must be >= hard_radius_m")
        if self.soft_cost_weight < 0.0:
            raise ValueError("soft_cost_weight must be >= 0")
        if self.soft_cost_power <= 0.0:
            raise ValueError("soft_cost_power must be > 0")

        self.H = int(self.bev_x / self.bev_res)
        self.W = int(self.bev_y / self.bev_res)

        self.large = 1e6
        self.goal_rc = None
        self.goal_xy_m = None
        self.fmm_dist = None
        self.hard_obstacle_map = None
        self.soft_cost_map = None
        self.travel_speed_map = None
        self.obstacle_clearance_map = None
        self._path_cache = None

        # 机器人在 BEV 的“起点”索引：r=0, c=中间
        self.robot_r = 0
        self.robot_c = self.W // 2

        self.hysteresis_thresh = float(hysteresis_thresh)
        self.hysteresis_hold_steps = int(hysteresis_hold_steps)
        self.max_wz_accel = float(max_wz_accel)
        self.control_dt = float(control_dt)
        self.unreachable_turn_wz = float(unreachable_turn_wz)
        self.use_path_lookahead = bool(use_path_lookahead)
        self.open_space_yaw_deadband = float(open_space_yaw_deadband)
        if self.open_space_yaw_deadband < 0.0:
            raise ValueError("open_space_yaw_deadband must be >= 0")
        self._last_turn_sign = 0      # 上一次的转向符号（+1 左 / -1 右）
        self._turn_sign_age = 0       # 当前符号已保持的步数
        self._last_wz = 0.0

    def set_goal(self, goal_xy_bev):
        gx_m, gy_m = float(goal_xy_bev[0]), float(goal_xy_bev[1])

        # 局部规划：goal 在身后会导致梯度怪象，钳到前方一点
        gx_m = max(gx_m, self.goal_min_forward_m)
        self.goal_xy_m = (gx_m, gy_m)

        gr = int(gx_m / self.bev_res)
        gc = int(gy_m / self.bev_res) + (self.W // 2)

        gr = int(np.clip(gr, 0, self.H - 1))
        gc = int(np.clip(gc, 0, self.W - 1))

        self.goal_rc = (gr, gc)

    def _inflate(self, occ_bev, radius_m=None):
        radius_m = self.inflate_radius_m if radius_m is None else float(radius_m)
        if radius_m <= 1e-6:
            return occ_bev
        rad = int(np.round(radius_m / self.bev_res))
        if rad <= 0:
            return occ_bev
        selem = morph.disk(rad)
        inflated = morph.binary_dilation(occ_bev.astype(bool), selem)
        return inflated.astype(np.uint8)

    def build_cost_layers(self, occ_bev):
        """Build the hard collision mask and continuous obstacle proximity cost.

        ``hard_obstacle_map`` is the only non-traversable region. The soft layer
        does not close a narrow passage; it only slows FMM propagation near an
        obstacle so that an available high-clearance route is preferred.
        """
        occ = np.asarray(occ_bev, dtype=np.uint8)
        if occ.shape != (self.H, self.W):
            raise ValueError(
                "occ_bev shape %s does not match planner shape (%d, %d)"
                % (occ.shape, self.H, self.W))

        hard_occ = self._inflate(occ, self.hard_radius_m)
        soft_cost = np.zeros_like(occ, dtype=np.float32)

        if np.any(occ):
            clearance_m = distance_transform_edt(occ == 0) * self.bev_res
            self.obstacle_clearance_map = clearance_m.astype(np.float32)
        else:
            clearance_m = np.full_like(occ, np.inf, dtype=np.float32)
            self.obstacle_clearance_map = clearance_m

        if np.any(occ) and self.soft_clearance_m > self.hard_radius_m:
            soft_width = self.soft_clearance_m - self.hard_radius_m
            normalized = np.clip(
                (self.soft_clearance_m - clearance_m) / soft_width,
                0.0,
                1.0,
            )
            soft_cost = np.power(normalized, self.soft_cost_power).astype(np.float32)
            soft_cost[hard_occ > 0] = 1.0

        return hard_occ, soft_cost

    def update(self, occ_bev):
        self._path_cache = None
        if self.goal_rc is None:
            self.fmm_dist = None
            return

        # 1) 原始模式只有固定膨胀；改进模式把硬碰撞与软安全距离分开。
        if self.use_soft_cost:
            occ, soft_cost = self.build_cost_layers(occ_bev)
        else:
            occ = self._inflate(occ_bev)
            soft_cost = np.zeros_like(occ, dtype=np.float32)
            raw_occ = np.asarray(occ_bev, dtype=np.uint8)
            if np.any(raw_occ):
                self.obstacle_clearance_map = (
                    distance_transform_edt(raw_occ == 0) * self.bev_res
                ).astype(np.float32)
            else:
                self.obstacle_clearance_map = np.full_like(
                    raw_occ, np.inf, dtype=np.float32)

        self.hard_obstacle_map = occ.copy()
        self.soft_cost_map = soft_cost

        #  2) traversable: 1 free / 0 obstacle
        traversable = (1 - occ).astype(np.float32)

        # masked array：硬障碍为 mask（不可走）
        trav_ma = ma.masked_values(traversable, 0.0)

        gr, gc = self.goal_rc

        # goal 如果落在障碍里：直接把它钳成 free（否则 skfmm 会返回 mask 常量）
        trav_ma[gr, gc] = 0.0

        if self.use_soft_cost:
            # 代价越高，传播速度越慢；travel_time 因而偏好更大的障碍净空。
            speed = 1.0 / (1.0 + self.soft_cost_weight * soft_cost)
            speed = np.maximum(speed, 1e-3).astype(np.float32)
            speed_ma = ma.array(speed, mask=ma.getmaskarray(trav_ma))
            speed_ma[gr, gc] = 1.0
            dist_ma = skfmm.travel_time(trav_ma, speed_ma, dx=1.0)
            self.travel_speed_map = speed
        else:
            dist_ma = skfmm.distance(trav_ma, dx=1.0)
            self.travel_speed_map = np.ones_like(traversable, dtype=np.float32)
        self.fmm_dist = ma.filled(dist_ma, self.large)

    def compute_wz(self):
        if self.fmm_dist is None:
            return 0.0

        # 密集障碍可能把局部 goal 与机器人完全隔开。此时距离场在机器人处
        # 没有下降方向；把零梯度解释成“直行”会正面撞障碍。先朝障碍较少的一侧
        # 原地搜索，等下一张深度图出现可行通道后再恢复前进。
        if not self.is_reachable():
            if self._last_turn_sign == 0:
                occ = self.hard_obstacle_map
                left_count = int(np.count_nonzero(occ[:, self.robot_c + 1:]))
                right_count = int(np.count_nonzero(occ[:, :self.robot_c]))
                self._last_turn_sign = 1 if left_count <= right_count else -1
                self._turn_sign_age = 0
            target = self._last_turn_sign * self.unreachable_turn_wz
            return self._rate_limit_wz(target)

        # 空旷区直接跟踪连续目标方向。把远处目标先裁到 BEV 边界，再沿离散
        # FMM 路径取前视点，会把厘米级横向误差量化成一个栅格斜线；转向锁存
        # 随后会把这个偶然符号保持 0.8 s，形成起步无意义转向。
        has_obstacles = (self.hard_obstacle_map is not None
                         and bool(np.any(self.hard_obstacle_map)))
        if not has_obstacles and self.goal_xy_m is not None:
            self._last_turn_sign = 0
            self._turn_sign_age = 0
            desired_yaw = float(np.arctan2(self.goal_xy_m[1], self.goal_xy_m[0]))
            if abs(desired_yaw) < self.open_space_yaw_deadband:
                desired_yaw = 0.0
            wz = float(np.clip(
                self.yaw_k * desired_yaw, -self.max_wz, self.max_wz))
            return self._rate_limit_wz(wz)

        # 窄缝中必须先对准通道中心。原实现固定在 (lookahead, 中心列) 取梯度，
        # 该采样点可能恰好落到石头边缘，方向会偏向一侧。现在优先沿已经提取出的
        # FMM 路径取前视点，直接跟踪可行通道的中心线。
        path = self.extract_path() if self.use_path_lookahead else []
        if len(path) > 1:
            lookahead_cells = max(1.0, self.lookahead_m / self.bev_res)
            target = path[-1]
            for point in path[1:]:
                if np.hypot(point[0] - self.robot_r,
                            point[1] - self.robot_c) >= lookahead_cells:
                    target = point
                    break
            dir_r = float(target[0] - self.robot_r)
            dir_c = float(target[1] - self.robot_c)
        else:
            r = int(np.clip(self.lookahead_m / self.bev_res, 1, self.H - 2))
            c = int(np.clip(self.robot_c, 1, self.W - 2))
            grad_r = 0.5 * (self.fmm_dist[r + 1, c] - self.fmm_dist[r - 1, c])
            grad_c = 0.5 * (self.fmm_dist[r, c + 1] - self.fmm_dist[r, c - 1])
            dir_r = -grad_r
            dir_c = -grad_c
        n = float(np.hypot(dir_r, dir_c) + 1e-6)
        dir_r /= n
        dir_c /= n

        # BEV: r ~ x forward, c ~ y left
        desired_yaw = float(np.arctan2(dir_c, dir_r))  # [-pi, pi]

        # ---- 转向决策锁存 ----
        # 对称鞍点问题：当 goal 与机器人连线【正对】一个障碍时，左右绕行代价相同，
        # FMM 距离场的横向梯度 dir_c 在 0 附近抖动，其符号每步翻转，
        # 于是 wz 在 +max_wz / -max_wz 之间来回跳，机器狗原地打转直至顶死在障碍上。
        # 实测（15 块石头场景 -> 8 块挡路石头场景）：wz 连续 170 步在 ±1.0 振荡，
        # 位移停在 x=7.31 不再前进，最终 stuck/timeout，成功率 0/10。
        # 解法：横向分量太小时不做新决策，沿用上一次已确立的转向符号（滞回），
        #      并保证该符号至少维持若干步，让机器狗把绕行动作【做完整】。
        if self.hysteresis_thresh > 0.0 or self.hysteresis_hold_steps > 0:
            sign = 0
            if desired_yaw > 0:
                sign = 1
            elif desired_yaw < 0:
                sign = -1
            lateral_weak = abs(dir_c) < self.hysteresis_thresh
            hold_active = (self.hysteresis_hold_steps > 0
                           and self._last_turn_sign != 0
                           and self._turn_sign_age < self.hysteresis_hold_steps)
            if (lateral_weak or hold_active) and self._last_turn_sign != 0 and sign != 0:
                # 证据不足 / 仍在保持期内：不翻转，沿用既有方向
                desired_yaw = math.copysign(abs(desired_yaw), self._last_turn_sign)
            else:
                if sign != 0:
                    self._last_turn_sign = sign
                    self._turn_sign_age = 0
            self._turn_sign_age += 1

        wz = float(np.clip(self.yaw_k * desired_yaw, -self.max_wz, self.max_wz))
        return self._rate_limit_wz(wz)

    def _rate_limit_wz(self, target_wz):
        """Limit command slew so one noisy replan cannot flip full-left to full-right."""
        target_wz = float(np.clip(target_wz, -self.max_wz, self.max_wz))
        if self.max_wz_accel > 0.0 and self.control_dt > 0.0:
            delta = self.max_wz_accel * self.control_dt
            target_wz = float(np.clip(
                target_wz, self._last_wz - delta, self._last_wz + delta))
        self._last_wz = target_wz
        return target_wz

    def is_reachable(self):
        """Return whether the current local goal connects to the robot cell."""
        if self.fmm_dist is None:
            return False
        value = float(self.fmm_dist[self.robot_r, self.robot_c])
        return np.isfinite(value) and value < self.large * 0.5

    def reset_hysteresis(self):
        """每个 trial 开始时调用，避免把上一轮的转向惯性带进新一轮。"""
        self._last_turn_sign = 0
        self._turn_sign_age = 0
        self._last_wz = 0.0

    def is_goal_reached(self, thresh_m=0.25):
        """用 dist 场判断是否接近 goal（可选）"""
        if self.fmm_dist is None:
            return False
        thresh_cells = float(thresh_m / self.bev_res)
        return self.fmm_dist[self.robot_r, self.robot_c] < thresh_cells
        
    def extract_path(self, max_steps=300):
        """
        沿 FMM 距离场，从 robot 走到 goal，返回 [(r,c), ...]。

        使用严格下降的 8 邻域离散搜索，而不是分别对两个梯度分量取 sign。
        后者会在离散势场的相邻格之间往返振荡，导致可视化路径总是跑满
        ``max_steps``，并污染路径长度指标。
        """
        if self.fmm_dist is None:
            return []
        if max_steps == 300 and self._path_cache is not None:
            return list(self._path_cache)

        r, c = int(self.robot_r), int(self.robot_c)
        H, W = self.fmm_dist.shape
        path = [(r, c)]

        for _ in range(max_steps):
            if (r, c) == self.goal_rc or self.fmm_dist[r, c] < 1.0:
                break

            current = float(self.fmm_dist[r, c])
            best = None
            best_value = current
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= H or nc < 0 or nc >= W:
                        continue
                    value = float(self.fmm_dist[nr, nc])
                    if value < best_value - 1e-6:
                        best = (nr, nc)
                        best_value = value

            # 局部不可达、平坦区或被 large 常量包围时停止，避免循环。
            if best is None or best_value >= self.large:
                break

            r, c = best
            path.append((r, c))

        if max_steps == 300:
            self._path_cache = list(path)
        return path

    def path_min_clearance(self, distance_m=1.5):
        """Return minimum raw obstacle clearance along the near-term path."""
        if self.obstacle_clearance_map is None or not self.is_reachable():
            return 0.0
        path = self.extract_path()
        if not path:
            return 0.0
        max_cells = max(1.0, float(distance_m) / self.bev_res)
        values = []
        for r, c in path:
            if np.hypot(r - self.robot_r, c - self.robot_c) > max_cells:
                break
            values.append(float(self.obstacle_clearance_map[r, c]))
        return min(values) if values else 0.0

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

import numpy as np
import random
import skimage.morphology as morph

class RRTController:
    """
    RRT 路径规划控制器 - 接口完全兼容 FMMGradientController
    """
    def __init__(
        self,
        bev_res=0.05,
        bev_x=6.0,
        bev_y=6.0,
        max_wz=1.0,
        yaw_k=2.0,
        lookahead_m=0.6,
        inflate_radius_m=0.25,
    ):
        self.bev_res = float(bev_res)
        self.bev_x = float(bev_x)
        self.bev_y = float(bev_y)
        self.max_wz = float(max_wz)
        self.yaw_k = float(yaw_k)
        self.lookahead_m = float(lookahead_m)
        self.inflate_radius_m = float(inflate_radius_m)

        self.H = int(self.bev_x / self.bev_res)
        self.W = int(self.bev_y / self.bev_res)

        # 机器人在 BEV 中的位置 (前进方向起始行=0, 左右居中)
        self.robot_r = 0
        self.robot_c = self.W // 2
        
        self.goal_rc = None
        self.goal_xy_m = (0, 0) # 存储米制坐标，用于判断是否到达
        self.path_rc = []       # 存储索引路径 [(r, c), ...]

    def set_goal(self, goal_xy_bev):
        """
        接收机器人坐标系下的目标 (x_forward, y_left)
        """
        self.goal_xy_m = goal_xy_bev
        gx_m, gy_m = float(goal_xy_bev[0]), float(goal_xy_bev[1])

        # 钳制目标点，防止越界
        gx_m = max(gx_m, 0.10)
        gr = int(np.clip(gx_m / self.bev_res, 0, self.H - 1))
        gc = int(np.clip(gy_m / self.bev_res + (self.W // 2), 0, self.W - 1))

        self.goal_rc = (gr, gc)

    def is_goal_reached(self, thresh_m=0.3):
        """
        判断是否到达目标点 (必须实现此函数以解决 AttributeError)
        由于是机器人坐标系，机器人始终在 (0,0)
        """
        dist = np.linalg.norm(np.array(self.goal_xy_m))
        return dist < thresh_m

    def _check_collision(self, node, occ):
        r, c = int(node[0]), int(node[1])
        if 0 <= r < self.H and 0 <= c < self.W:
            return occ[r, c] > 0
        return True
    def _check_edge_collision(self, p1, p2, occ):
        """
        检查 p1 -> p2 线段是否碰撞
        """
        num = int(np.linalg.norm(p2 - p1))
        if num < 2:
            return self._check_collision(p2, occ)

        for t in np.linspace(0, 1, num):
            p = p1 + t * (p2 - p1)
            r, c = int(p[0]), int(p[1])
            if not (0 <= r < self.H and 0 <= c < self.W):
                return True
            if occ[r, c] > 0:
                return True
        return False
    def update(self, occ_bev):
        """
        执行 RRT 采样并生成路径
        """
        if self.goal_rc is None:
            self.path_rc = []
            return

        # 1. 膨胀障碍物
        rad = int(np.round(self.inflate_radius_m / self.bev_res))
        occ = morph.binary_dilation(occ_bev.astype(bool), morph.disk(rad)).astype(np.uint8) if rad > 0 else occ_bev

        # 2. RRT 算法核心
        start = np.array([self.robot_r, self.robot_c])
        goal = np.array(self.goal_rc)
        nodes = [start]
        parent = {0: None}
        
        step_size = 2.0 # 像素步长
        found = False
        
        # 限制迭代次数保证实时性
        for _ in range(300):
            # 采样 (15% 概率直连目标)
            if random.random() < 0.6:
                rand_node = goal
            else:
                rand_node = np.array([random.randint(0, self.H-1), random.randint(0, self.W-1)])
            
            # 寻最近点
            dists = [np.linalg.norm(n - rand_node) for n in nodes]
            nearest_idx = np.argmin(dists)
            nearest_node = nodes[nearest_idx]
            
            # 步进
            diff = rand_node - nearest_node
            dist = np.linalg.norm(diff)
            if dist < 1e-6: continue
            
            new_node = nearest_node + (diff / dist) * step_size if dist > step_size else rand_node
            
            # 碰撞检测
            if not self._check_edge_collision(nearest_node, new_node, occ):
                nodes.append(new_node)
                parent[len(nodes)-1] = nearest_idx
                if np.linalg.norm(new_node - goal) < step_size:
                    parent[len(nodes)] = len(nodes)-1
                    nodes.append(goal)
                    found = True
                    break
        
        # 3. 提取路径索引
        if found:
            path = []
            curr = len(nodes) - 1
            while curr is not None:
                path.append(tuple(nodes[curr]))
                curr = parent[curr]
            self.path_rc = path[::-1]
        else:
            self.path_rc = []

    def compute_wz(self):
        """
        根据规划路径计算转向角 wz
        """
        if not self.path_rc or len(self.path_rc) < 2:
            return 0.0
        
        # 像素级预瞄距离
        lookahead_px = self.lookahead_m / self.bev_res
        
        # 寻找预瞄点
        target_node = self.path_rc[-1]
        for node in self.path_rc:
            dist = np.linalg.norm(np.array(node) - np.array([self.robot_r, self.robot_c]))
            if dist >= lookahead_px:
                target_node = node
                break
        
        # 计算误差角 (r->x_forward, c->y_left)
        dr = target_node[0] - self.robot_r
        dc = target_node[1] - self.robot_c
        error_yaw = np.arctan2(dc, dr)
        
        return float(np.clip(self.yaw_k * error_yaw, -self.max_wz, self.max_wz))

    def extract_path(self):
        """
        返回索引路径，用于 pathplanner.py 中的渲染画线
        """
        return self.path_rc
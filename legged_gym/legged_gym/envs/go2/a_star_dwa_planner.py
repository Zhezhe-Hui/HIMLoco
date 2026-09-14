import numpy as np
import heapq
import skimage.morphology as morph

class AStarDWAController:
    def __init__(
        self,
        bev_res=0.05,
        bev_x=6.0,
        bev_y=6.0,
        max_vx=1.0,      
        max_wz=1.2,      
        max_acc_vx=2.0,  
        max_acc_wz=3.0,  
        dt=0.1,          
        predict_time=1.2, 
        inflate_radius_m=0.25, # 如果还是撞墙，可调至0.3
    ):
        self.bev_res = bev_res
        self.H = int(bev_x / bev_res)
        self.W = int(bev_y / bev_res)
        self.dt = dt
        self.predict_time = predict_time
        self.max_vx = max_vx
        self.max_wz = max_wz
        self.max_acc_vx = max_acc_vx
        self.max_acc_wz = max_acc_wz
        self.inflate_radius_m = inflate_radius_m
        
        self.current_vel = np.array([0.0, 0.0])
        self.robot_r = 0
        self.robot_c = self.W // 2
        
        self.global_path = [] 
        self.goal_rc = None
        self.goal_robot_frame = None

    def set_goal(self, goal_robot_frame):
        self.goal_robot_frame = goal_robot_frame
        dx, dy = goal_robot_frame
        self.goal_rc = (int(dx / self.bev_res), int(dy / self.bev_res) + self.W // 2)

    def update(self, occ_bev):
        # 1. 地图预处理
        k_size = max(1, int(self.inflate_radius_m / self.bev_res))
        self.inflated_map = morph.binary_dilation(occ_bev, morph.disk(k_size))
        
        # 2. A* 搜索逻辑
        new_path = self._a_star_search(self.inflated_map, (self.robot_r, self.robot_c), self.goal_rc)
        
        # 如果搜到新路径则更新，否则尝试保留旧路径（避免蓝色球闪烁消失）
        if len(new_path) > 0:
            self.global_path = new_path
        
        # 3. DWA 计算
        best_vx, best_wz = self._dwa_compute(self.inflated_map)
        self.current_vel = np.array([best_vx, best_wz])

    def _a_star_search(self, grid, start, goal):
        if goal is None: return []
        
        # 检查终点是否越界
        gr, gc = goal
        if not (0 <= gr < self.H and 0 <= gc < self.W): return []
        
        # 如果起点或终点被障碍物覆盖，A* 会失败。这里做一点宽容处理
        if grid[start[0], start[1]]:
            # 强行返回一个直线点，防止导航直接挂掉
            return [start, goal] if not grid[gr, gc] else []

        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {start: None}
        g_score = {start: 0}
        
        while open_list:
            _, current = heapq.heappop(open_list)
            
            # 只要接近目标就算成功
            if abs(current[0]-gr) <= 1 and abs(current[1]-gc) <= 1:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                nr, nc = current[0] + dr, current[1] + dc
                if 0 <= nr < self.H and 0 <= nc < self.W:
                    if grid[nr, nc]: continue
                    
                    cost = np.sqrt(dr**2 + dc**2)
                    tentative_g = g_score[current] + cost
                    if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                        g_score[(nr, nc)] = tentative_g
                        f_score = tentative_g + np.sqrt((nr-gr)**2 + (nc-gc)**2)
                        came_from[(nr, nc)] = current
                        heapq.heappush(open_list, (f_score, (nr, nc)))
        return []

    def _dwa_compute(self, grid):
        dw = [
            max(0.0, self.current_vel[0] - self.max_acc_vx * self.dt),
            min(self.max_vx, self.current_vel[0] + self.max_acc_vx * self.dt),
            max(-self.max_wz, self.current_vel[1] - self.max_acc_wz * self.dt),
            min(self.max_wz, self.current_vel[1] + self.max_acc_wz * self.dt)
        ]
        
        # 确定参考点：如果有 A* 路径，取路径上的预瞄点；如果没有，取最终目标点
        if len(self.global_path) > 5:
            # 预瞄点取路径上约 0.8 米处
            idx = min(len(self.global_path)-1, int(0.8 / self.bev_res))
            ref_node = self.global_path[idx]
            ref_xy = np.array([ref_node[0]*self.bev_res, (ref_node[1]-self.W//2)*self.bev_res])
        elif self.goal_robot_frame is not None:
            ref_xy = np.array(self.goal_robot_frame)
        else:
            return 0.0, 0.0

        best_score = -float('inf')
        best_v = [0.0, 0.0]

        # 速度采样
        for v in np.linspace(dw[0], dw[1], 10):
            for w in np.linspace(dw[2], dw[3], 15):
                traj = self._predict_trajectory(v, w)
                score = self._evaluate_traj(traj, grid, ref_xy)
                if score > best_score:
                    best_score = score
                    best_v = [v, w]
        
        # 兜底：如果所有路径都碰撞，得分都是 -1e6，强制原地慢转
        if best_score < -1e5:
            return 0.0, 0.5 if self.goal_robot_frame[1] > 0 else -0.5
            
        return best_v

    def _predict_trajectory(self, v, w):
        traj = []
        x, y, theta = 0.0, 0.0, 0.0
        for _ in range(int(self.predict_time / self.dt)):
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt
            theta += w * self.dt
            traj.append([x, y, theta])
        return np.array(traj)

    def _evaluate_traj(self, traj, grid, ref_xy):
        # 1. 严格碰撞检查
        for pt in traj[::2]:
            r = int(self.robot_r + pt[0] / self.bev_res)
            c = int(self.robot_c + pt[1] / self.bev_res)
            if not (0 <= r < self.H and 0 <= c < self.W) or grid[r, c]:
                return -1e6

        # 2. 距离参考点的距离
        dist_score = 1.0 / (np.linalg.norm(traj[-1][:2] - ref_xy) + 1.0)
        
        # 3. 朝向参考点的角度
        target_angle = np.arctan2(ref_xy[1]-traj[-1][1], ref_xy[0]-traj[-1][0])
        angle_err = abs(self._normalize_angle(target_angle - traj[-1][2]))
        heading_score = (np.pi - angle_err) / np.pi
        
        # 4. 速度分
        vel_score = traj[-1][0] / self.max_vx

        return 4.0 * dist_score + 2.0 * heading_score + 0.5 * vel_score

    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def compute_wz(self):
        return self.current_vel[1]

    def extract_path(self):
        return self.global_path

    def is_goal_reached(self, thresh_m=0.3):
        if self.goal_robot_frame is None: return False
        return np.linalg.norm(self.goal_robot_frame) < thresh_m
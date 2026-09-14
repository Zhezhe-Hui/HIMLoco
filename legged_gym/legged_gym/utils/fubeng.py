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
import numpy as np
from numpy.random import choice
from scipy import interpolate
from isaacgym import gymtorch, gymapi, gymutil
import os
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import trimesh
class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.num_envs = num_robots   # ⭐ 关键修复
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        elif hasattr(cfg, 'terrain_sequence') and cfg.terrain_sequence is not None:
            # ✅ 新增：支持有序地形序列
            self.sequence_terrain()
        else:    
            self.randomized_terrain()  

        # self.add_wall(0,432,2,4, height=3)
        # self.add_wall(0,432,44,46, height=3)
        self.stone_positions = self.generate_stones()

        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
            # ============================================================
            # 加载石头 Mesh 并合并进地形（方案 A 核心）
            # ============================================================
            self._merge_stone_meshes()

    def _merge_stone_meshes(self):
        """
        把 stone.obj 加载进来，并将其合并到 terrain 的 vertices / triangles。
        不创建 actor，避免 rigid_body_states 混乱。
        """

        stone_obj_path = os.path.join(LEGGED_GYM_ROOT_DIR, "resources/stone/stone.obj")
        if not os.path.exists(stone_obj_path):
            print("[STONE] stone.obj not found, skip adding stones.")
            return

        stone_mesh = trimesh.load(stone_obj_path, force='mesh')

        # 原 terrain 网格
        base_vertices = self.vertices
        base_triangles = self.triangles

        new_vertices = base_vertices.copy()
        new_triangles = base_triangles.copy()

        # 遍历 Terrain.generate_stones 生成的石头位置
        if not hasattr(self, "stone_positions"):
            print("[STONE] No stone_positions found, skip merging stones.")
            return

        for (row, col), stones in self.stone_positions.items():
            for (sx, sy) in stones:
                # 将 stone mesh 平移到 (sx, sy)
                scale = 0.1   # ← 调小这个值，例如 0.05、0.1、0.2 可调
                stone_v = (stone_mesh.vertices * scale).copy()

                stone_v[:, 0] += sx
                stone_v[:, 1] += sy
                stone_v[:, 2] += 0.0  # 微调高度避免穿地

                # 三角片索引偏移
                idx_offset = new_vertices.shape[0]
                stone_f = stone_mesh.faces + idx_offset

                # 合并
                new_vertices = np.vstack((new_vertices, stone_v))
                new_triangles = np.vstack((new_triangles, stone_f))

        self.vertices = new_vertices.astype(np.float32)
        self.triangles = new_triangles.astype(np.uint32)

        print(f"[STONE] Merge done, total vertices = {len(self.vertices)}, triangles = {len(self.triangles)}")

    def generate_stones(self):
        """
        为 terrain 网格生成石头，而不是为每个环境生成。
        """
        stones = {}

        num_stones = 200
        env_size = 5.0

        rows = self.env_origins.shape[0]
        cols = self.env_origins.shape[1]

        for r in range(rows):
            for c in range(cols):
                origin = self.env_origins[r][c]  #安全！永不越界
                ox, oy = float(origin[0]), float(origin[1])

                stones[(r, c)] = []
                for _ in range(num_stones):
                    x = ox + np.random.uniform(-env_size, env_size)
                    y = oy + np.random.uniform(-env_size, env_size)
                    stones[(r, c)].append((x, y))
        return stones

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
    def sequence_terrain(self):
        """✅ 新增：有序地形生成（按 terrain_sequence 指定）"""
        terrain_sequence = self.cfg.terrain_sequence
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            
            # ✅ 从序列中获取地形类型
            terrain_type_idx = terrain_sequence[i % len(terrain_sequence)]
            
            # ✅ 转换为 choice 值（0.0 - 1.0 范围）
            # 6 种地形类型映射到 [0, 1) 区间
            choice = (terrain_type_idx + 0.5) / 6.0
            
            # 可选：使用固定难度或随机难度
            difficulty = 0.5  # 固定难度
            # difficulty = np.random.choice([0.5, 0.75, 0.9])  # 或随机难度
            
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)    
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        amplitude = 0.01 + 0.07 * difficulty
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.1
        stepping_stones_size = max(0.5, 1.5 * (1.05 - difficulty))  # 确保至少0.5m
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *=  -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-amplitude, max_height=amplitude, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.30, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            slope = 0.0  # 强制坡度为0
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)  # 用平滑地形函数生成平地
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def add_wall(self, x1, x2, y1, y2, height):
        """
        在 heightfield 中任意位置添加一个矩形墙体区域
        参数:
            x1, x2: 行方向位置（像素）
            y1, y2: 列方向位置（像素）
            height: 墙的实际高度（米）
        """
        h = int(height / self.cfg.vertical_scale)
        self.height_field_raw[x1:x2, y1:y2] = h

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

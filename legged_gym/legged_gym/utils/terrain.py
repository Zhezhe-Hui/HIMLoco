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

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

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
        border_ground_m = float(getattr(cfg, "border_ground_height_m", 0.0))
        border_ground_px = int(round(border_ground_m / cfg.vertical_scale))
        # border_size 是可行走缓冲带，不应沿用旧的 -5m 深坑初始化。外侧仍由
        # add_all_walls() 封闭，地形主体随后会覆盖中央区域。
        self.height_field_raw = np.full(
            (self.tot_rows, self.tot_cols), border_ground_px, dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        elif hasattr(cfg, 'terrain_sequence') and cfg.terrain_sequence is not None:
            self.sequence_terrain()
        else:    
            self.randomized_terrain()  
        self.add_all_walls()
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    def add_all_walls(self, thickness_m=None, height_m=None):
        # 参数可在 config 的 terrain 段调整；None 表示沿用 config 值
        thickness_m = float(getattr(self.cfg, "wall_thickness_m", 1.0)) \
            if thickness_m is None else float(thickness_m)
        height_m = float(getattr(self.cfg, "wall_height_m", 4.0)) \
            if height_m is None else float(height_m)
        if not bool(getattr(self.cfg, "wall_enable", True)):
            return

        thick_px = int(thickness_m / self.cfg.horizontal_scale)
        height_px = int(height_m / self.cfg.vertical_scale)

        # 地图大小
        rows, cols = self.height_field_raw.shape

        # 安全边界——不要和机器人出生点 (center) 重叠
        margin = int(float(getattr(self.cfg, "wall_margin_m", 1.0)) / self.cfg.horizontal_scale)

        # 左墙
        self.height_field_raw[margin:rows-margin, :thick_px] = height_px
        # 右墙
        self.height_field_raw[margin:rows-margin, cols-thick_px:] = height_px
        # 上墙 (top)
        self.height_field_raw[:thick_px, margin:cols-margin] = height_px
        # 下墙 (bottom)
        self.height_field_raw[rows-thick_px:, margin:cols-margin] = height_px

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
            
    def make_terrain_by_id(self, terrain_id, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale
        )
        # difficulty 建议范围 [0, 1]
        # 以下系数全部可在 config 的 terrain 段调整（基类给了与原实现一致的默认值）
        cfg = self.cfg
        slope = difficulty * float(getattr(cfg, "slope_scale", 0.4))
        amplitude = float(getattr(cfg, "amplitude_min", 0.01)) \
            + float(getattr(cfg, "amplitude_scale", 0.07)) * difficulty
        step_height = float(getattr(cfg, "step_height_scale", 0.18)) * difficulty
        discrete_obstacles_height = float(getattr(cfg, "discrete_obstacle_base", 0.05)) \
            + float(getattr(cfg, "discrete_obstacle_scale", 0.1)) * difficulty
        platform = float(getattr(cfg, "platform_size_default", 3.0))
        platform_rough = float(getattr(cfg, "platform_size_rough_slope", 0.5))
        step_w = float(getattr(cfg, "step_width", 0.50))

        # ---------------- 地形定义 ----------------
        if terrain_id == 0:
            # 0 = 光滑斜坡
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=platform)
        elif terrain_id == 1:
            # 1 = 粗糙斜坡
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=platform_rough)
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=float(getattr(cfg, "rough_slope_min_height", -0.05)),
                max_height=float(getattr(cfg, "rough_slope_max_height", 0.05)),
                step=float(getattr(cfg, "rough_slope_step", 0.005)),
                downsampled_scale=float(getattr(cfg, "rough_slope_downsample", 0.2)))
        elif terrain_id == 2:
            # 2 = 下楼梯
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_w, step_height=-abs(step_height), platform_size=platform)
        elif terrain_id == 3:
            # 3 = 上楼梯
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_w, step_height=abs(step_height), platform_size=platform)
        elif terrain_id == 4:
            # 4 = 离散障碍
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                float(getattr(cfg, "discrete_obstacle_height_fixed", 0.01)),
                int(getattr(cfg, "discrete_rect_min_size", 2)),
                int(getattr(cfg, "discrete_rect_max_size", 3)),
                int(getattr(cfg, "discrete_num_rectangles", 50)),
                platform_size=platform)
        elif terrain_id == 5:
            # 5 = 平地
            terrain_utils.pyramid_sloped_terrain(terrain, slope=0, platform_size=platform)
        elif terrain_id == 6:
            # 6 = 离散高度
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=float(getattr(cfg, "height_field_min", 0.0)),
                max_height=float(getattr(cfg, "height_field_max", 0.08)),
                step=float(getattr(cfg, "height_field_step", 0.005)),
                downsampled_scale=float(getattr(cfg, "height_field_downsample", 0.2)))
        elif terrain_id == 7:
            # 7 = 砖块长条障碍：类似论文图里的小矩形砖/短墙
            brick_obstacles_terrain(
                terrain,
                obstacle_height=getattr(self.cfg, "brick_obstacle_height", 0.22),
                brick_length=getattr(self.cfg, "brick_length", 1.0),
                brick_width=getattr(self.cfg, "brick_width", 0.35),
                num_bricks=getattr(self.cfg, "num_bricks", 70),
                platform_size=getattr(self.cfg, "brick_platform_size", 2.0),
                max_yaw=getattr(self.cfg, "brick_max_yaw", 0.35),
            )
        else:
            # 兜底：非法编号 → 平地
            terrain_utils.pyramid_sloped_terrain(terrain, slope=0.0, platform_size=platform)
        return terrain

    def sequence_terrain(self):
        seq = self.cfg.terrain_sequence
        fallback_id = int(getattr(self.cfg, "sequence_fallback_terrain_id", 5))
        difficulty = float(getattr(self.cfg, "sequence_difficulty", 0.5))

        for k in range(self.cfg.num_sub_terrains):
            i, j = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            # 按行 i 取 terrain_id。
            # 注意：原来这里守卫写的是 `if j < len(seq)` 但取值用 seq[i]，
            # 索引不一致，一旦 num_rows > len(seq) 就会 IndexError，故统一按 i 判断。
            terrain_id = seq[i] if i < len(seq) else fallback_id

            terrain = self.make_terrain_by_id(terrain_id, difficulty)
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

        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *=  -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=0.5)
            terrain_utils.random_uniform_terrain(terrain, min_height=-amplitude, max_height=amplitude, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.30, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 3.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            slope = 0.0  # 强制坡度为0
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)  # 用平滑地形函数生成平地
        elif choice < self.proportions[6]:
            terrain_utils.random_uniform_terrain(terrain, min_height=0.0, max_height=0.23, step=0.005, downsampled_scale=0.2)
        else: 
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)

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

def brick_obstacles_terrain(
    terrain,
    obstacle_height=0.22,
    brick_length=1.0,
    brick_width=0.35,
    num_bricks=45,
    platform_size=2.0,
    max_yaw=0.35,
):
    """Scatter low rectangular brick obstacles on flat terrain.

    The output is still a normal IsaacGym heightfield/trimesh terrain; each
    brick is rasterized into ``terrain.height_field_raw``.
    """
    height = int(obstacle_height / terrain.vertical_scale)
    length_px = max(1, int(brick_length / terrain.horizontal_scale))
    width_px = max(1, int(brick_width / terrain.horizontal_scale))
    platform_px = int(platform_size / terrain.horizontal_scale / 2)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    safe_x1 = center_x - platform_px
    safe_x2 = center_x + platform_px
    safe_y1 = center_y - platform_px
    safe_y2 = center_y + platform_px

    margin = max(length_px, width_px) + 2
    if terrain.length <= 2 * margin or terrain.width <= 2 * margin:
        return

    for _ in range(num_bricks):
        cx = np.random.randint(margin, terrain.length - margin)
        cy = np.random.randint(margin, terrain.width - margin)

        if safe_x1 <= cx <= safe_x2 and safe_y1 <= cy <= safe_y2:
            continue

        yaw = np.random.uniform(-max_yaw, max_yaw)
        if np.random.rand() < 0.5:
            yaw += np.pi / 2.0

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        half_l = length_px / 2.0
        half_w = width_px / 2.0
        radius = int(np.ceil(np.sqrt(half_l ** 2 + half_w ** 2))) + 1

        x1 = max(0, cx - radius)
        x2 = min(terrain.length, cx + radius + 1)
        y1 = max(0, cy - radius)
        y2 = min(terrain.width, cy + radius + 1)

        xs = np.arange(x1, x2) - cx
        ys = np.arange(y1, y2) - cy
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")

        local_x = cos_yaw * grid_x + sin_yaw * grid_y
        local_y = -sin_yaw * grid_x + cos_yaw * grid_y
        mask = (np.abs(local_x) <= half_l) & (np.abs(local_y) <= half_w)

        terrain.height_field_raw[x1:x2, y1:y2][mask] = height

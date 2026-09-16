import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# 路径与评测参数统一由 envs/go2/viz_config.py 管理，本脚本不再写死任何绝对路径
from legged_gym.envs.go2.viz_config import (
    TRAJ_FIG_DIR, TRAJ_FIG_ALGORITHM, TRAJ_FIG_TERRAIN,
    COMPARE_ALGORITHMS, COMPARE_TERRAINS,
    WAYPOINTS, compare_traj_dir, ensure_dir,
)

# ===================== 仅调大字体，不加粗 =====================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 只改字号，不做任何加粗
plt.rcParams['axes.labelsize'] = 17     # x/y 轴标签
plt.rcParams['xtick.labelsize'] = 15    # x刻度数字
plt.rcParams['ytick.labelsize'] = 15    # y刻度数字
plt.rcParams['legend.fontsize'] = 15    # 图例文字
plt.rcParams['axes.titlesize'] = 21     # 标题字号

# 高清保存，不改动线条粗细
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 120
# ======================================================================

def plot_all_saved_trajectories(folder_path, goal_xy_all, save_path=None, title=None):
    # 读取所有保存的轨迹
    traj_files = sorted(glob.glob(os.path.join(folder_path, "traj_*.npy")))
    all_trajs = [np.load(f) for f in traj_files]

    plt.figure(figsize=(10, 8))

    # 绘制机器人轨迹（保持原始粗细，不加粗）
    for traj in all_trajs:
        plt.plot(traj[:, 0], traj[:, 1], 'forestgreen', alpha=0.6, label="Robot Trajectory")

    # 绘制目标点
    if len(goal_xy_all) > 0:
        G = np.array(goal_xy_all)
        plt.scatter(G[:, 0], G[:, 1], c='green', s=60, marker='o', label="Waypoints")

    # 绘制障碍物（保持原始大小样式）
    big_stone_positions = [
        # x = 8
        (8, 1),
        (8, 5),
        (8, 9.4),
        
        # x = 12
        (12, 2),
        (12, 4),
        (12, 5),
        (12, 8),
        (12, 10),
        
        # x = 16
        (16, 1),
        (16, 5),
        (16, 7),
        (16, 9.4),
        
        # x = 20
        (20, 2),
        (20, 4),
        (20, 8),
        (20, 10),
        
        # x = 24
        (24, 1),
        (24, 5),
        (24, 7),
        (24, 9.4),
    ]
    stones = np.array(big_stone_positions, dtype=float)
    plt.scatter(stones[:, 0], stones[:, 1], c='darkred', s=200, marker='s', label="Obstacles")

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    if title:
        plt.title(title)

    # 图例去重
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # 保存高清原图风格
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📸 图片已保存: {save_path}")
    plt.close()

# -----------------------------------------------------------------------------
#  出图对象来自 viz_config.TRAJ_FIG_ALGORITHM / TRAJ_FIG_TERRAIN，改那里即可切换
ALGORITHM = TRAJ_FIG_ALGORITHM      # "dwa"(A*+DWA) / "fmm"(本文) / "rrt"(RRT*)
TERRAIN = TRAJ_FIG_TERRAIN          # "height" / "obstacles" / "slope"

if __name__ == "__main__":
    assert ALGORITHM in COMPARE_ALGORITHMS, f"未知算法: {ALGORITHM}"
    assert TERRAIN in COMPARE_TERRAINS, f"未知地形: {TERRAIN}"

    # 轨迹数据目录：assets/data/compare/<algo>_<terrain>/traj_*.npy
    TRAJECTORY_FOLDER = compare_traj_dir(ALGORITHM, TERRAIN)
    # 汇总图输出：assets/figs/summary/
    SAVE_PATH = os.path.join(ensure_dir(TRAJ_FIG_DIR),
                             f"{ALGORITHM}_{TERRAIN}_trajectories.png")

    plot_all_saved_trajectories(
        folder_path=TRAJECTORY_FOLDER,
        goal_xy_all=WAYPOINTS,
        save_path=SAVE_PATH,
        title=f"{ALGORITHM.upper()}_{TERRAIN.capitalize()}",
    )
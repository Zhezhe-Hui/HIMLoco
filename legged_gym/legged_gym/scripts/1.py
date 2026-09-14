import numpy as np
import matplotlib.pyplot as plt
import glob
import os

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

def plot_all_saved_trajectories(folder_path, goal_xy_all, save_path=None):
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
    plt.title("FMM_Slope")

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
if __name__ == "__main__":
    TRAJECTORY_FOLDER = "/home/hzz/project/HIMLoco/legged_gym/legged_gym/scripts/FMM_Slope"
    
    WAYPOINTS = [
        (11.0, 8.0),
        (15.0, 4.0),
        (19.0, 7.0),
        (23.0, 5.0),
    ]
    
    plot_all_saved_trajectories(
        folder_path=TRAJECTORY_FOLDER,
        goal_xy_all=WAYPOINTS,
        save_path="guijitu"
    )
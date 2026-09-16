import matplotlib.pyplot as plt
import numpy as np
import os

# 输出目录与开关统一由 envs/go2/viz_config.py 管理，本脚本不再往 cwd 里写图
from legged_gym.envs.go2.viz_config import (
    METRICS_FIG_DIR, METRICS_PRINT_STATS, METRICS_SHOW_PLOT, ensure_dir,
)

# ===================== 全局字体&字号配置 =====================
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['axes.titlesize'] = 18     # 图标题大小
plt.rcParams['axes.labelsize'] = 17      # 坐标轴标签大小
plt.rcParams['xtick.labelsize'] = 12     # X 轴刻度数字大小
plt.rcParams['ytick.labelsize'] = 12     # Y 轴刻度数字大小
plt.rcParams['legend.fontsize'] = 13     # 图例大小

def generate_report():
    # ===================== 【统计功能总开关】viz_config.METRICS_PRINT_STATS =====================
    enable_stat_analysis = METRICS_PRINT_STATS
    # 出图目录（assets/figs/metrics/）
    out_dir = ensure_dir(METRICS_FIG_DIR)

    # ===================== 中文字体容错加载 =====================
    possible_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
    if os.path.exists(possible_path):
        import matplotlib.font_manager as fm
        my_font = fm.FontProperties(fname=possible_path)
        plt.rcParams['font.family'] = my_font.get_name()
        plt.rcParams['axes.unicode_minus'] = False
    else:
        try:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
            plt.rcParams['axes.unicode_minus'] = False
            print("尝试通过系统名称加载文泉驿字体")
        except:
            print("未找到中文字体，请确认已安装 fonts-wqy-microhei")

    # ===================== 原始数据集 =====================
    DATA = {
        # ---------------------- FMM ----------------------
        "FMM": {
            "obs": {
                "smooth": [0.2513, 0.2803, 0.2523, 0.2641, 0.2508, 0.2457, 0.2641, 0.2482, 0.2684, 0.2684],
                "dist": [20.950, 21.023, 21.185, 21.009, 21.125, 20.967, 21.142, 21.059, 21.198, 7.533],
                "plan": [3.30, 3.29, 3.26, 3.24, 3.27, 3.29, 3.31, 3.30, 3.28, 3.28]
            },
            "slope": {
                "smooth": [0.1678, 0.1595, 0.1735, 0.1752, 0.1519, 0.1574, 0.1607, 0.1607, 0.1851, 0.1444],
                "dist": [21.211, 22.135, 21.558, 21.516, 22.251, 21.405, 21.251, 7.974, 21.422, 21.354],
                "plan": [3.23, 3.20, 3.21, 3.18, 3.18, 3.21, 3.21, 3.21, 3.19, 3.19]
            },
            "height": {
                "smooth": [0.1595, 0.1538, 0.1647, 0.1722, 0.1703, 0.1607, 0.1834, 0.1554, 0.1831, 0.1688],
                "dist": [20.984, 21.311, 21.219, 21.249, 21.380, 20.973, 21.158, 21.603, 21.192, 20.992],
                "plan": [3.26, 3.26, 3.26, 3.27, 3.27, 3.27, 3.25, 3.25, 3.23, 3.23]
            }
        },
        # ---------------------- RRT ----------------------
        "RRT": {
            "obs": {
                "smooth": [0.3883, 0.3379, 0.3390, 0.3575, 0.3575, 0.3575, 0.3522, 0.3511, 0.3472, 0.3504],
                "dist": [22.626, 21.727, 22.756, 22.273, 5.469, 5.672, 22.266, 22.360, 22.370, 22.201],
                "plan": [31.69, 19.21, 21.60, 18.14, 18.14, 18.14, 19.77, 18.76, 21.88, 19.47]
            },
            "slope": {
                "smooth": [0.4357, 0.4357, 0.4357, 0.3808, 0.3902, 0.3850, 0.3808, 0.3808, 0.3859, 0.3952],
                "dist": [17.503, 22.658, 9.363, 22.864, 22.892, 21.991, 22.864, 8.425, 22.745, 22.867],
                "plan": [49.29, 49.29, 21.84, 29.20, 24.50, 21.85, 21.85, 23.81, 21.98, 21.84]
            },
            "height": {
                "smooth": [0.4775, 0.4844, 0.4806, 0.4806, 0.4806, 0.4806, 0.4723, 0.4800, 0.4806, 0.4762],
                "dist": [23.134, 22.454, 22.992, 15.432, 1.492, 8.346, 23.074, 22.690, 18.744, 22.847],
                "plan": [29.86, 33.22, 24.01, 24.01, 24.01, 24.01, 26.05, 40.07, 24.01, 30.16]
            }
        },
        # ---------------------- A*+DWA ----------------------
        "A*+DWA": {
            "obs": {
                "smooth": [0.2423, 0.2423, 0.2683, 0.2701, 0.2569, 0.2935, 0.2439, 0.2718, 0.2498, 0.2292],
                "dist": [21.217, 16.910, 31.246, 30.452, 30.926, 33.298, 21.504, 31.462, 30.061, 22.041],
                "plan": [53.22, 53.22, 33.92, 33.60, 23.60, 24.51, 49.29, 42.63, 32.31, 47.57]
            },
            "slope": {
                "smooth": [0.1887, 0.1878, 0.3533, 0.2044, 0.1783, 0.1898, 0.2020, 0.1969, 0.2182, 0.1852],
                "dist": [21.327, 21.596, 33.827, 27.831, 21.354, 21.377, 21.419, 21.725, 21.210, 22.109],
                "plan": [38.64, 34.85, 14.97, 38.37, 35.54, 42.62, 36.83, 32.12, 38.73, 30.20]
            },
            "height": {
                "smooth": [0.1790, 0.1710, 0.1790, 0.2519, 0.1841, 0.1785, 0.3500, 0.2123, 0.1685, 0.1742],
                "dist": [21.176, 21.505, 21.105, 29.689, 21.223, 21.084, 27.592, 21.179, 21.300, 21.110],
                "plan": [34.36, 32.51, 35.98, 28.12, 32.92, 32.84, 21.36, 37.74, 37.78, 36.17]
            }
        }
    }

    env_map = {"obs":"Obstacles", "slope":"Slope", "height":"Height"}
    env_list = ["obs", "slope", "height"]
    algo_list = ["FMM", "RRT", "A*+DWA"]
    trials = np.arange(1, 11)

    # ===================== 统计计算逻辑（开关可控） =====================
    if enable_stat_analysis:
        print("="*110)
        print(f"【统计汇总：每组10次实验 均值 | 最大值 | 最小值】")
        print("="*110)
        for env in env_list:
            print(f"\n>>>>>>>>>> 地形: {env_map[env]}")
            for algo in algo_list:
                smooth_arr = DATA[algo][env]["smooth"]
                dist_arr = DATA[algo][env]["dist"]
                plan_arr = DATA[algo][env]["plan"]

                # 计算均值、最大、最小
                stat_smooth = (np.mean(smooth_arr), np.max(smooth_arr), np.min(smooth_arr))
                stat_dist = (np.mean(dist_arr), np.max(dist_arr), np.min(dist_arr))
                stat_plan = (np.mean(plan_arr), np.max(plan_arr), np.min(plan_arr))

                print(f"\n[{algo:6s}]")
                print(f"  轨迹平滑度(rad): 均值={stat_smooth[0]:.4f}, 最大值={stat_smooth[1]:.4f}, 最小值={stat_smooth[2]:.4f}")
                print(f"  路径长度(m):      均值={stat_dist[0]:.3f}, 最大值={stat_dist[1]:.3f}, 最小值={stat_dist[2]:.3f}")
                print(f"  规划耗时(ms):    均值={stat_plan[0]:.2f}, 最大值={stat_plan[1]:.2f}, 最小值={stat_plan[2]:.2f}")
        print("\n"+"="*110)

    # ===================== 批量循环三种地形，自动绘图 =====================
    for current_env in env_list:
        env_name = env_map[current_env]

        smoothness_data = {
            'FMM': DATA['FMM'][current_env]['smooth'],
            'RRT': DATA['RRT'][current_env]['smooth'],
            'A*+DWA': DATA['A*+DWA'][current_env]['smooth']
        }
        distance_data = {
            'FMM': DATA['FMM'][current_env]['dist'],
            'RRT': DATA['RRT'][current_env]['dist'],
            'A*+DWA': DATA['A*+DWA'][current_env]['dist']
        }
        plan_time_data = {
            'FMM': DATA['FMM'][current_env]['plan'],
            'RRT': DATA['RRT'][current_env]['plan'],
            'A*+DWA': DATA['A*+DWA'][current_env]['plan']
        }

        # 图1：轨迹平滑度对比
        plt.figure(figsize=(10, 8))
        plt.plot(trials, smoothness_data['FMM'], 'o-', label='FMM', linewidth=2, markersize=6)
        plt.plot(trials, smoothness_data['RRT'], 'o-', label='RRT', linewidth=2, markersize=6)
        plt.plot(trials, smoothness_data['A*+DWA'], 'o-', label='A* + DWA', linewidth=2, markersize=6)
        plt.title(f"Trajectory Smoothness — {env_name} Terrain")
        plt.xlabel("Trial")
        plt.ylabel("Smoothness / rad")
        plt.xticks(trials)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"smoothness_{env_name}.png"), dpi=300)
        plt.close()

        # 图2：规划耗时对比
        plt.figure(figsize=(10, 8))
        plt.plot(trials, plan_time_data['FMM'], 'o-', label='FMM', linewidth=2, markersize=6)
        plt.plot(trials, plan_time_data['RRT'], 'o-', label='RRT', linewidth=2, markersize=6)
        plt.plot(trials, plan_time_data['A*+DWA'], 'o-', label='A* + DWA', linewidth=2, markersize=6)
        plt.title(f"Average Planning Time — {env_name} Terrain")
        plt.xlabel("Trial")
        plt.ylabel("Time / ms")
        plt.xticks(trials)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"planning_time_{env_name}.png"), dpi=300)
        plt.close()

    print(f"📸 指标对比图已保存到: {out_dir}")
    if METRICS_SHOW_PLOT:
        plt.show()

if __name__ == "__main__":
    generate_report()
# -*- coding: utf-8 -*-
"""
go2 可视化与输出统一配置
========================
把 play.py / play1.py / pathplanner.py 里散落的所有开关集中到这一个文件。
改开关只需编辑本文件，不必再去动 play.py 或 pathplanner.py。

【最快用法】只改下面一行 ACTIVE_PRESET 即可：
    "off"          全部关闭：不截图、不存图、不画叠加（最快、最省显存）
    "viewer_only"  只开 IsaacGym 窗口跟随视角，不存任何图片（默认）
    "paper"        论文出图：存 BEV/FMM 三联图 + trial 总览图 + 轨迹 npy + 汇总图
    "all"          全部打开（最慢，仅调试用）

被 pathplanner.py 与 play.py 通过 `from ... import *` 引入，
因此这里的名字就是原来那些模块级常量，行为完全兼容。
"""

import os

from legged_gym import LEGGED_GYM_ROOT_DIR

# =====================================================================
#  预设选择（只改这一行）
# =====================================================================
#: 可选 "nav_only"(只跑导航) / "viewer_only"(能看机器狗动) / "off" / "paper" / "all"
#:
#: ⛔⛔ 卡死根因（2026-09-16 实测定位，本机向日葵远程桌面 + RTX 4060 8GB + 31GB 内存）⛔⛔
#:   真凶是【每步的相机渲染】render_all_camera_sensors + 把两张 1280x720 图拷回 CPU，
#:   不是 viewer 逐帧重绘，也不是石头/树。关掉它（CAM_READ_ENABLE=False）后
#:   viewer 画面照常刷新、机器狗照常走。
#:
#:   鉴别实验（同样无石头无树、同一 checkpoint、同一 1 trial）：
#:     · viewer_only + CAM_READ_ENABLE=True  -> ~74~139MB/s，step 600~750 到 8GB，被看门狗拦
#:     · viewer_only + CAM_READ_ENABLE=False ->  3.4MB/s，跑完，✅ 成功
#:     · nav_only    + CAM_READ_ENABLE=False ->  3.1MB/s，10 trial 峰值 5.9GB，10/10 ✅
#:   组件级探针进一步排除嫌疑：draw_viewer 2000 步仅 0.07MB/s、
#:   skfmm 3000 步 0.04MB/s、set_camera_location 0MB/s，均非泄漏源。
#:
#:   ⚠ 注意：原代码里 bev.update() 本来就被注释掉了（BEV_ENABLE_UPDATE=False），
#:     深度图读出来从未被用于建图，所以 CAM_READ_ENABLE=False 与原行为【功能等价】，
#:     只是省掉了一次无用的读取。真要启用深度避障时才需要打开它。
#:
#:   结论：
#:     · 远程桌面下跑数据/回归 -> "nav_only"（最快最省，10 trial 约 239s）
#:     · 要肉眼看机器狗走      -> "viewer_only"（10 trial 约 345s，画面正常刷新）
#:       两者都已默认 CAM_READ_ENABLE=False，不会卡死
#:     · 要出论文图（存 RGB/Depth、三联图）-> "paper"，它会自动开 CAM_READ_ENABLE=True；
#:       此时相机泄漏回归，务必坐本机、调小 NUM_TRIALS、别开远程桌面长时间跑
#:     · 任何模式都有内置内存看门狗（见下节 SAFETY_MEM_*）兜底，不会拖死整机
ACTIVE_PRESET = "viewer_only"


# =====================================================================
#  内置内存看门狗（防止再出现“跑一会儿整机卡死”）
# ---------------------------------------------------------------------
#  上面那种泄漏只能靠事后 kill 挽救，所以在评测主循环里内置一道防线：
#  每隔 SAFETY_MEM_CHECK_INTERVAL 步读一次 /proc/self/status 的 VmRSS，
#  超过 SAFETY_MEM_LIMIT_GB 就立刻收尾退出，而不是把内存吃满、
#  触发 swap、把整台机器（含远程桌面）一起拖死。
#  构建完成后的正常基线约 4.8GB，故默认阈值取 8GB：留足余量又不会误杀。
# =====================================================================
SAFETY_MEM_GUARD_ENABLE = True    #: 是否启用内置内存看门狗
SAFETY_MEM_LIMIT_GB = 8.0         #: RSS 超过该值(GB)则中止评测
#: 实测余量（nav_only，无石头无树）：基线 4.87GB，之后约 +4.3MB/s。
#:   10 trial 峰值 5.9GB / 20 trial 6.9GB / 30 trial 7.9GB，8GB 约可撑到 32 trial。
#:   论文用的 10~20 trial 完全够；要跑更多 trial 请把这个值调大（如 12.0）。
SAFETY_MEM_CHECK_INTERVAL = 50    #: 每隔多少步检查一次（<=0 表示每步）


# =====================================================================
#  输出路径（全部统一到仓库根目录的 assets/ 下）
#  ---------------------------------------------------------------------
#  assets/paper/   原论文素材（HIMLoco / H-Infinity 的配图与 PDF）
#  assets/figs/    论文出图
#     ├─ trajectory/  各算法轨迹可视化（ours / fmm / rrt / a_star_dwa / alt）
#     ├─ metrics/     平滑度、规划耗时等指标对比图
#     └─ summary/     汇总轨迹图
#  assets/data/    实验数据
#     ├─ compare/     三算法对比的轨迹 npy（<算法>_<地形>）
#     └─ trajectories/  当前运行产出的轨迹 npy
#  assets/runs/    运行时产出（截图/视频帧，体积大，可随时清空重跑）
#     ├─ bev_fmm/     BEV 高度图、障碍图、FMM 势场三联图
#     ├─ trial_route/ 每个 trial 的俯视总览图
#     ├─ rgb_sav/     RGB / Depth 相机图像
#     ├─ leg_closeup/ play.py 腿部特写截图
#     └─ trial_follow_video/ 跟随视角录屏帧
#  assets/archive/ 旧版压缩包与冗余文件（已不再被代码引用）
# =====================================================================
#: 仓库根目录（viz_config.py 位于 legged_gym/legged_gym/envs/go2/，向上 5 层）
REPO_ROOT = os.path.abspath(
    os.path.join(LEGGED_GYM_ROOT_DIR, os.pardir))
#: 素材与产出的统一根目录
ASSETS_ROOT = os.path.join(REPO_ROOT, "assets")

#: 论文出图目录
FIGS_ROOT = os.path.join(ASSETS_ROOT, "figs")
#: 实验数据目录
DATA_ROOT = os.path.join(ASSETS_ROOT, "data")
#: 运行时产出目录（可随时清空）
RUNS_ROOT = os.path.join(ASSETS_ROOT, "runs")
#: 归档目录（旧压缩包等，代码不再写入）
ARCHIVE_ROOT = os.path.join(ASSETS_ROOT, "archive")
#: 训练日志根目录（train.py 存 model_*.pt 的地方）
LOGS_ROOT = os.path.join(LEGGED_GYM_ROOT_DIR, "logs")


#: 论文出图细分目录
FIGS_TRAJ_DIR = os.path.join(FIGS_ROOT, "trajectory")      #: 各算法轨迹可视化
FIGS_METRICS_DIR = os.path.join(FIGS_ROOT, "metrics")      #: 平滑度 / 规划耗时等指标对比图
FIGS_SUMMARY_DIR = os.path.join(FIGS_ROOT, "summary")      #: 汇总轨迹图

#: 三算法对比实验数据（目录名规则：<算法>_<地形>，全小写）
COMPARE_DATA_DIR = os.path.join(DATA_ROOT, "compare")
COMPARE_ALGORITHMS = ("dwa", "fmm", "rrt")                 #: A*+DWA / 本文 FMM / RRT*
COMPARE_TERRAINS = ("height", "obstacles", "slope")        #: 三种地形


def compare_traj_dir(algorithm, terrain):
    """返回“某算法 + 某地形”的对比轨迹 npy 目录（assets/data/compare/<algo>_<terrain>）。"""
    return os.path.join(COMPARE_DATA_DIR,
                        "%s_%s" % (algorithm.lower(), terrain.lower()))


def ensure_dir(path):
    """建目录并原样返回，便于 savefig / np.save 直接串起来用。"""
    os.makedirs(path, exist_ok=True)
    return path


# =====================================================================
#  一、相机图像保存（RGB / Depth）  —— play1.py 走 pathplanner 这条链
# =====================================================================
SAVE_CAMERA_IMAGES = False          #: 总开关：保存相机图像到文件
SAVE_RGB_IMAGE = False              #: 是否保存 RGB
SAVE_DEPTH_IMAGE = False            #: 是否保存深度图
SAVE_CAMERA_IMAGE_INTERVAL = 50     #: 每隔多少步存一张
SAVE_CAMERA_IMAGE_DIR = os.path.join(RUNS_ROOT, "rgb_sav")


# =====================================================================
#  二、BEV / FMM 三联图保存（高度图 + 障碍图 + 势场）
# =====================================================================
SAVE_BEV_FMM_IMAGES = False             #: 总开关：保存 BEV/FMM 三联图
SAVE_BEV_FMM_INTERVAL = 5               #: 每隔多少步存一组
SAVE_BEV_FMM_MAX_PER_TRIAL = 800        #: 单个 trial 最多存多少组
SAVE_BEV_FMM_DIR = os.path.join(RUNS_ROOT, "bev_fmm")


# =====================================================================
#  三、trial 总览截图（固定俯视相机，叠加规划路线与实际轨迹）
# =====================================================================
SAVE_TRIAL_ROUTE_IMAGE = False                  #: 总开关：每个 trial 结束后存一张总览图
SAVE_TRIAL_ROUTE_DIR = os.path.join(RUNS_ROOT, "trial_route")
SAVE_TRIAL_ROUTE_LOCAL_FMM_SEGMENTS = False     #: 总览图里是否叠加过程中采样的局部 FMM 段
SAVE_TRIAL_ROUTE_LOCAL_FMM_INTERVAL = 10        #: 局部 FMM 段的采样间隔（步）


# =====================================================================
#  四、轨迹数据与汇总图
# =====================================================================
SAVE_TRAJECTORY_NPY = False                     #: 是否把每个 trial 的真实轨迹存成 .npy
SAVE_TRAJECTORY_DIR = os.path.join(DATA_ROOT, "trajectories")
SAVE_SUMMARY_TRAJECTORY_PLOT = False            #: 所有 trial 跑完后是否画汇总轨迹图
SAVE_SUMMARY_TRAJECTORY_PLOT_DIR = FIGS_SUMMARY_DIR


# =====================================================================
#  五、IsaacGym viewer 实时显示（只动视角，不存文件）
# =====================================================================
LIVE_SHOW_CAMERA_IMAGES = False         #: 实时弹出 RGB/Depth 的 matplotlib 窗口
LIVE_CAMERA_IMAGE_INTERVAL = 5          #: 实时相机窗口刷新间隔（步）
LIVE_SHOW_BEV_FMM_WINDOW = False        #: 实时弹出 BEV/FMM 三联图窗口
LIVE_BEV_FMM_WINDOW_INTERVAL = 5        #: 实时三联图刷新间隔（步）

LIVE_DRAW_VIEWER_OVERLAY = True         #: 在 IsaacGym viewer 里画叠加物（下面几项的总开关）
LIVE_DRAW_WAYPOINTS = True              #: 画目标点
LIVE_DRAW_TRAJECTORY = False            #: 画已走过的轨迹
LIVE_DRAW_FMM_PATH = True               #: 画 FMM 预测路径
LIVE_FOLLOW_VIEWER_CAMERA = True        #: viewer 相机跟随机器狗（适合直接录屏）
LIVE_HOLD_FINAL_ROUTE = False           #: 跑完整条路线后停留几秒看最终路线

# ---------- ⚡ 性能开关（这几项是“整机卡死”的真正来源，务必先看这里） ----------
#: 每步是否让 IsaacGym 重绘整个 viewer 场景（env.enable_viewer_sync）。
#: ⚠ 这是最大的性能杀手：env.step() 内部每步都调 render()，开启时会执行
#:   step_graphics + draw_viewer，把场景里【所有】actor 的图形体重画一遍。
#:   985 碎石 + 4 棵高面数树 + 远程桌面(向日葵)合成 → 每步几万条线段，整机锁死。
#:   设为 False：窗口不再逐帧刷新，但【导航与指标完全正常】，深度相机照常出图。
#:   只在需要肉眼看/录屏时才开 True（此时建议同时把石头树关掉）。
LIVE_VIEWER_SYNC = True
#: 每隔多少步重画一次 viewer 叠加物（1 = 每步，最卡；10 = 省 10 倍绘制开销）
LIVE_DRAW_INTERVAL = 1
#: 轨迹球最多画多少个。原实现每步重画【全部】历史轨迹点，轨迹越长每步越慢
#: （1500 步 → 每步画 ~187 个球 × 72 条线），是典型的 O(N²) 卡顿。
#: 超过该数量会自动加大抽样间隔，把球数压到上限内。0 = 不限制。
LIVE_DRAW_TRAJ_MAX_POINTS = 300
#: 同上，限制 FMM 预测路径球数量（0 = 不限制）
LIVE_DRAW_FMM_PATH_MAX_POINTS = 200

# ---------- 深度相机读取（导航感知链路的入口） ----------
#: 每步是否读深度相机（render_all_camera_sensors + 两张 1280x720 图 GPU→CPU 拷贝）。
#: ⚠ 这是“跑一会儿整机卡死”的真凶（实测 74~139MB/s 泄漏，详见文件头），
#:   因此【默认 False】。只有满足下面任一条件才需要设 True：
#:     1) BEV_ENABLE_UPDATE=True（深度图喂给 BEV 建图 → 上层真的在避障）
#:     2) SAVE_CAMERA_IMAGES / SAVE_RGB_IMAGE / SAVE_DEPTH_IMAGE=True（要存图）
#:     3) LIVE_SHOW_CAMERA_IMAGES / SAVE_BEV_FMM_IMAGES=True（要实时窗口/三联图）
#:   原代码 bev.update() 本就被注释掉，深度图读出来从未用于建图，
#:   所以默认 False 与原行为【功能等价】，导航与指标完全不受影响。
CAM_READ_ENABLE = False
# ---------- viewer 叠加物的绘制样式（只影响显示，不影响控制/指标） ----------
#: 目标点线框球半径 [m]
DRAW_WAYPOINT_RADIUS_M = 0.15
#: 目标点球体离地净空 [m]（_get_draw_position 用，避免陷进地形）
DRAW_WAYPOINT_CLEARANCE_M = 0.08
#: 已走轨迹球半径 [m]
DRAW_TRAJ_RADIUS_M = 0.05
#: 轨迹球离地净空 [m]
DRAW_TRAJ_CLEARANCE_M = 0.06
#: 每隔多少个轨迹点画一个球（越大越省开销）
DRAW_TRAJ_STEP_SKIP = 8
#: FMM 预测路径球半径 [m]
DRAW_FMM_PATH_RADIUS_M = 0.04
#: FMM 预测路径球离地净空 [m]
DRAW_FMM_PATH_CLEARANCE_M = 0.06
#: 每隔多少个 FMM 路径点画一个球
DRAW_FMM_PATH_STEP_SKIP = 10
#: 计算叠加物 z 高度时的默认离地净空 [m]
DRAW_DEFAULT_CLEARANCE_M = 0.05
#: viewer 里绘制/等待循环的帧间隔 [s]
LIVE_DRAW_SLEEP_S = 0.02

# =====================================================================
#  六、IsaacGym viewer 跟随相机机位参数（单位：米）
# =====================================================================
TRIAL_FOLLOW_CAMERA_BACK_M = 2.5            #: 往后退多少
TRIAL_FOLLOW_CAMERA_RIGHT_M = 0.9           #: 往右偏多少
TRIAL_FOLLOW_CAMERA_HEIGHT_M = 0.6          #: 相机离地高度
TRIAL_FOLLOW_CAMERA_LOOKAHEAD_M = 1.0       #: 注视点往前多少
TRIAL_FOLLOW_CAMERA_LOOKAT_HEIGHT_M = 0.35  #: 注视点离地高度


# =====================================================================
#  七、画图样式（只影响出图，不影响控制）
# =====================================================================
BEV_FMM_SMOOTH_PATH = False                   #: 三联图里是否对路径做平滑显示
BEV_FMM_SHOW_POTENTIAL = False                #: 三联图是否显示第三张 FMM 势场图
TRIAL_ROUTE_SHOW_TARGET_ORDER_LINE = False    #: 总览图是否画“起点->目标点->终点”连线
TRIAL_ROUTE_SMOOTH_TARGET_LINE = False        #: 总览图目标点连线是否平滑显示
TRIAL_ROUTE_HIDE_SMALL_STONES_IN_CAMERA = False  #: 拍总览图时临时隐藏小石头（只留大石头和树）
TRIAL_ROUTE_OVERLAY_Z_OFFSET = 0.08           #: 叠加线抬高多少米，避免被地面遮挡

# 总览图固定相机参数
TRIAL_ROUTE_CAMERA_MIN_HEIGHT = 8.0           #: 最小俯拍高度（会按路线范围自动增高）
TRIAL_ROUTE_CAMERA_FOV_DEG = 90.0             #: 视场角
TRIAL_ROUTE_CAMERA_X_RANGE = (3.0, 27.0)      #: 横轴范围（世界坐标 x）
TRIAL_ROUTE_CAMERA_Y_RANGE = None             #: None = 按地形宽度自动取 y 范围
TRIAL_ROUTE_CAMERA_BACK_Y_M = -10             #: 相机沿 y 后退（轻微倾斜俯视，避免正俯视变黑）
TRIAL_ROUTE_CAMERA_LOOKAT_Z = 0.35            #: 注视点高度


# =====================================================================
#  八、play.py（底层步态单独 play，不走规划器）专用开关
# =====================================================================
LEG_CLOSEUP_CAPTURE = False                     #: 腿部特写自动截图总开关（原来默认 True，会一运行就存图）
LEG_CLOSEUP_DIR = os.path.join(RUNS_ROOT, "leg_closeup")
LEG_CLOSEUP_START_STEP = 20                     #: 从第几步开始截
LEG_CLOSEUP_FRAME_INTERVAL = 3                  #: 每隔多少步截一张
LEG_CLOSEUP_MAX_FRAMES = 150                    #: 最多截多少张

EXPORT_POLICY = True                            #: 是否导出 JIT 策略（真机部署用）
RECORD_FRAMES = False                           #: play.py 里按帧存图（需自己建 exported/frames 目录）
MOVE_CAMERA = False                             #: play.py 里相机是否持续平移


# =====================================================================
#  九、离线出图脚本（scripts/1.py 轨迹图、scripts/a.py 指标图）
# =====================================================================
#: 轨迹汇总图输出目录（scripts/1.py）
TRAJ_FIG_DIR = FIGS_SUMMARY_DIR
#: 指标对比图输出目录（scripts/a.py）
METRICS_FIG_DIR = FIGS_METRICS_DIR
#: scripts/1.py 默认画哪个算法 / 哪种地形
TRAJ_FIG_ALGORITHM = "fmm"        #: "dwa" / "fmm" / "rrt"
TRAJ_FIG_TERRAIN = "slope"        #: "height" / "obstacles" / "slope"
#: scripts/a.py：是否在终端打印均值/最值统计表
METRICS_PRINT_STATS = True
#: scripts/a.py：画完是否弹窗显示（批量出图时建议 False，免得卡在 plt.show）
METRICS_SHOW_PLOT = False


# =====================================================================
#  十、play1.py 评测流程参数
# =====================================================================
#: 并行环境数。
#: ⚠ go2_config 里 stone.num_stones=1000 是"每个 env"的石头数，
#:   actor 创建在 for i in range(num_envs) 内，所以显存随 num_envs 线性暴涨。
#:   RTX 4060 (8GB) 实测：16 可跑，32 起 PhysX 报 fail to allocate memory。
#:   纯看步态可设 1~4；要跑论文级评测建议先把 num_stones 调小再加大本值。
NUM_ENVS = 1

# =====================================================================
NUM_TRIALS = 10        #: 跑多少次独立导航试验（论文表2/表3 用的是 10 或 20）
RENDER = True           #: 是否开 IsaacGym viewer 窗口
PLANNER_MODE = "fmm"    #: 高层规划器："fmm"(本文方法) / "rrt"(RRT*) / "astar_dwa"(A*+DWA)
#: 导航目标点（世界坐标），论文用的是 4 个点：A/B/C + 终点
WAYPOINTS = [
    (11.0, 8.0),
    (15.0, 4.0),
    (19.0, 7.0),
    (23.0, 5.0),
]


# =====================================================================
#  十·B、play1.py 评测时对 env_cfg 的覆盖开关
# ---------------------------------------------------------------------
#  原 play1.py 里这几行是写死在脚本里的：
#      env_cfg.noise.add_noise = False
#      env_cfg.domain_rand.randomize_friction = False
#      env_cfg.domain_rand.push_robots = False
#      env_cfg.domain_rand.disturbance = False
#      env_cfg.domain_rand.randomize_payload_mass = False
#      env_cfg.commands.heading_command = False
#  导航评测要可复现，所以默认全关（保持原行为）；
#  想做“带扰动的鲁棒性对比实验”时把对应项改成 True 即可，不必动脚本。
# =====================================================================
EVAL_OVERRIDE_ADD_NOISE = False                #: 观测噪声
EVAL_OVERRIDE_RANDOMIZE_FRICTION = False       #: 地面摩擦随机化
EVAL_OVERRIDE_PUSH_ROBOTS = False              #: 随机推力扰动
EVAL_OVERRIDE_DISTURBANCE = False              #: 外力干扰
EVAL_OVERRIDE_RANDOMIZE_PAYLOAD_MASS = False   #: 负载质量随机化
EVAL_OVERRIDE_HEADING_COMMAND = False          #: 速度指令按朝向自动转向

#: play1.py 按此表逐项覆盖 env_cfg；键为 (cfg 段名, 属性名)
EVAL_OVERRIDES = (
    (("noise", "add_noise"), EVAL_OVERRIDE_ADD_NOISE),
    (("domain_rand", "randomize_friction"), EVAL_OVERRIDE_RANDOMIZE_FRICTION),
    (("domain_rand", "push_robots"), EVAL_OVERRIDE_PUSH_ROBOTS),
    (("domain_rand", "disturbance"), EVAL_OVERRIDE_DISTURBANCE),
    (("domain_rand", "randomize_payload_mass"), EVAL_OVERRIDE_RANDOMIZE_PAYLOAD_MASS),
    (("commands", "heading_command"), EVAL_OVERRIDE_HEADING_COMMAND),
)

#: 被 play1.py 注释掉的 terrain 覆盖项，留成开关方便做地形消融实验
EVAL_TERRAIN_OVERRIDE_ENABLE = False   #: True 时用下面几个值覆盖 env_cfg.terrain
EVAL_TERRAIN_NUM_ROWS = 10
EVAL_TERRAIN_NUM_COLS = 8
EVAL_TERRAIN_CURRICULUM = True
EVAL_TERRAIN_MAX_INIT_LEVEL = 9
EVAL_TERRAIN_MESH_TYPE = "plane"       #: 'plane' / 'trimesh'


# =====================================================================
#  十·C、加载哪个训练好的策略模型（play1.py / play.py）
# ---------------------------------------------------------------------
#  原代码 train_cfg.runner.load_run 保持默认 -1，而 helpers.get_load_path
#  在 -1 时是把 logs/<experiment_name>/ 下的 run 目录名做“字符串排序”取最后一个。
#  这会踩两个坑：
#    1) 目录名按字母序排，Sep15_* 永远排在 Dec10_* 后面，跨月份直接选错；
#    2) 任何一次 train 冒烟测试留下的 3~60 迭代垃圾 run，都会顶掉真正训好的模型，
#       表现为“机器狗走两步就摔 / 成功率 0”。
#  所以这里把模型来源显式配置出来，不再依赖目录名排序。
#  取值规则（交给 get_load_path）：
#    POLICY_LOAD_RUN  = -1 沿用原行为（字符串序最后一个 run）
#                     = "Sep15_12-14-31_"  logs/<experiment_name>/ 下的 run 目录名
#                     = "/abs/path/to/run" 绝对路径（os.path.join 会直接采用）
#    POLICY_CHECKPOINT= -1 取该 run 里编号最大的 model_*.pt
#                     = 1200 取 model_1200.pt
# =====================================================================
POLICY_EXPERIMENT_NAME = "rough_go2"   #: logs/ 下的实验名（对应 runner.experiment_name）
#: 留空 = 默认 logs/<experiment_name>；也可填绝对路径的 log_root
POLICY_LOG_ROOT = os.path.join(LOGS_ROOT, POLICY_EXPERIMENT_NAME)
#: 该实验下当前唯一训练到 1200 迭代的 run（其余都是 2~60 迭代的冒烟测试）
POLICY_LOAD_RUN = os.path.join(LOGS_ROOT, "Dec10_20-00-07_")
#: -1 = 取该 run 里编号最大的 model_*.pt；填数字则取 model_<N>.pt
POLICY_CHECKPOINT = 1200


# =====================================================================
#  十一、场景与障碍物（石头 / 树 / 地形）
# ---------------------------------------------------------------------
#  这一节控制“仿真场景里生成什么”，由 go2_config.py 在类定义时读取，
#  因此对 train.py / play.py / play1.py 全部生效，改这里即可，不必动 go2_config.py。
#
#  ⚠ 显存关键：石头 actor 是在 `for i in range(num_envs)` 内逐个创建的，
#    所以总 actor 数 = num_envs × num_stones，显存随之线性暴涨。
#    RTX 4060 (8GB) 实测：num_stones=1000 时 num_envs≤16 可跑，32 起 PhysX OOM。
#    把 SCENE_NUM_SMALL_STONES 调成 0（只留 15 个大石头）后 num_envs 可上到 256。
# =====================================================================

# ---------- 树 / 静态障碍物（deadwood、oak_tree、snow_tree） ----------
SCENE_ENABLE_TREES = True             #: 总开关：是否生成 static_obstacles.assets 里的树
#: 只生成指定名字的树；None 或空列表 = 全部生成（树名见 go2_config.static_obstacles.assets）
SCENE_TREE_WHITELIST = None

# ---------- 石头（大小石头数量互相独立，不再共用一个 num_stones） ----------
SCENE_ENABLE_STONES = True            #: 总开关：关掉则大小石头全部不生成
SCENE_ENABLE_BIG_STONES = True        #: 是否生成固定位置的大石头（导航实验的主要障碍）
SCENE_NUM_SMALL_STONES = 985          #: 随机撒布的小碎石数量（0 = 只留大石头）
SCENE_BIG_STONE_SCALE = 2             #: 大石头 URDF 等比放大倍数
SCENE_SMALL_STONE_SCALE = (0.1, 0.3)  #: 小碎石随机缩放范围 (min, max)
SCENE_STONE_SPAWN_X = (12.0, 36.0)    #: 小碎石随机撒布范围（世界坐标 x）
SCENE_STONE_SPAWN_Y = (0.0, 12.0)     #: 小碎石随机撒布范围（世界坐标 y）
SCENE_STONE_DENSITY = 1000.0          #: 石头密度（仅小碎石为动态时有意义）

# ---------- 石头性能开关（显存 / 卡顿） ----------
#: 小碎石是否静态化（fix_base_link + disable_gravity）。
#: ⚠ 这是消除“卡顿”的关键开关：动态 mesh 刚体是 PhysX 里最贵的组合。
#:   实测 8env×985碎石：动态 26.35ms/step -> 静态 12.81ms/step（2.06x 加速）。
#:   代价：碎石不会被踢飞、不参与动力学（导航实验里碎石本就是固定障碍，影响可忽略）。
#:   若需要碎石被踢飞的真实动力学，设为 False。
SCENE_SMALL_STONES_STATIC = True
#: 静态化时小碎石中心离地形的高度。动态时改用 SCENE_STONE_SPAWN_HEIGHT 抛下落体。
#: 实测碎石落地稳定后中心 z ≈ 地形高度，故取 0.01 略微抬起避免与地面共面。
SCENE_SMALL_STATIC_Z_OFFSET = 0.01
#: 动态（非静态）模式下小碎石的抛下高度，与原实现一致
SCENE_STONE_SPAWN_HEIGHT = 0.3
#: 把连续随机缩放量化成 N 档。PhysX 为每种 actor scale 生成独立碰撞几何，
#: 连续 scale 会导致 985 个石头各建一份；量化后可复用缓存。0 = 不量化。
#: 实测对构建时间影响很小（瓶颈在树），但能减少显存碎片，建议保留 4~8。
SCENE_SMALL_STONE_SCALE_STEPS = 4

#: 碰撞体模式："mesh"（三角面网格，最贴外形）/ "vhacd"（凸分解）/ None（IsaacGym 默认）
#: ⚠ 实测石头的 vhacd 无收益（构建时间 43.2s vs 43.8s、GPU 2610 vs 2632MiB 基本持平），
#:   因为石头只有 442 面，本身就便宜。保持 "mesh" 即可。
SCENE_SMALL_STONE_COLLISION = "mesh"
SCENE_BIG_STONE_COLLISION = "mesh"

# ---------- 地形 ----------
#: 地形 id 对照（实现见 legged_gym/utils/terrain.py:make_terrain_by_id）
TERRAIN_IDS = {
    "smooth_slope": 0,   # 光滑斜坡
    "slope":        1,   # 粗糙斜坡
    "stairs_down":  2,   # 下楼梯
    "stairs_up":    3,   # 上楼梯
    "obstacles":    4,   # 离散障碍
    "flat":         5,   # 平地
    "height":       6,   # 离散高度
    "brick":        7,   # 砖块长条障碍
}
#: 地形预设："custom" 用下面手写的序列；或填 TERRAIN_IDS 里的名字（如 "slope"）生成单一地形。
#: 对应论文三种实验地形：height / obstacles / slope。
SCENE_TERRAIN_PRESET = "custom"
#: SCENE_TERRAIN_PRESET="custom" 时使用的地形序列（原 go2_config 默认值，勿随意缩短）
SCENE_TERRAIN_SEQUENCE_CUSTOM = [4, 1, 4, 1, 1, 1]
# ---------- 地形形状（对应 utils/terrain.py:make_terrain_by_id）----------
#: sequence_terrain 的固定难度，直接影响坡度/台阶/障碍高度
TERRAIN_SEQUENCE_DIFFICULTY = 0.5
TERRAIN_SLOPE_SCALE = 0.4               #: 坡度 = difficulty × 该系数
TERRAIN_STEP_WIDTH = 0.50               #: 楼梯台阶宽度 [m]
TERRAIN_STEP_HEIGHT_SCALE = 0.18        #: 台阶高度 = difficulty × 该系数
TERRAIN_PLATFORM_SIZE = 3.0             #: 默认中央平台尺寸 [m]
TERRAIN_PLATFORM_SIZE_ROUGH_SLOPE = 0.5 #: 粗糙斜坡的平台尺寸 [m]
TERRAIN_AMPLITUDE_MIN = 0.01            #: 起伏幅度基准 [m]
TERRAIN_AMPLITUDE_SCALE = 0.07          #: 起伏幅度 = 基准 + 该系数×difficulty
#: 离散障碍(id=4)
TERRAIN_DISCRETE_HEIGHT_FIXED = 0.01
TERRAIN_DISCRETE_RECT_MIN = 2
TERRAIN_DISCRETE_RECT_MAX = 3
TERRAIN_DISCRETE_NUM_RECTS = 50
#: 离散高度(id=6)，对应论文的 height 地形
TERRAIN_HEIGHT_FIELD_MIN = 0.0
TERRAIN_HEIGHT_FIELD_MAX = 0.08
TERRAIN_HEIGHT_FIELD_STEP = 0.005
TERRAIN_HEIGHT_FIELD_DOWNSAMPLE = 0.2
#: 粗糙斜坡(id=1)的随机起伏
TERRAIN_ROUGH_MIN_HEIGHT = -0.05
TERRAIN_ROUGH_MAX_HEIGHT = 0.05
TERRAIN_ROUGH_STEP = 0.005
TERRAIN_ROUGH_DOWNSAMPLE = 0.2
#: 离散障碍高度的 difficulty 换算
TERRAIN_DISCRETE_OBSTACLE_BASE = 0.05
TERRAIN_DISCRETE_OBSTACLE_SCALE = 0.1
#: 地形边界墙（防止机器人跑出地图）
TERRAIN_WALL_ENABLE = True
TERRAIN_WALL_THICKNESS_M = 1.0
TERRAIN_WALL_HEIGHT_M = 4.0
TERRAIN_WALL_MARGIN_M = 1.0
#: 地形列数不足时兜底用的地形 id（5 = 平地）
TERRAIN_SEQUENCE_FALLBACK_ID = 5

#: 单一地形时序列的填充长度。
#: ⚠ 必须 ≥ terrain.num_rows（当前 3）：terrain.py:sequence_terrain 用 seq[row] 取值，
#:   长度不够会 IndexError。取 6 与原设置长度一致。
SCENE_TERRAIN_PAD = 6

# ---------- 大石头固定位置（世界坐标，导航实验的障碍布局） ----------
#: 与论文实验布局一致；改这里即可重排大石头。
SCENE_BIG_STONE_POSITIONS = [
    (8, 1), (8, 5), (8, 9.4),
    (12, 2), (12, 10),
    (16, 1), (16, 5), (16, 7), (16, 9.4),
    (20, 2), (20, 4), (20, 8), (20, 10),
    (24, 7), (24, 9.4),
]

# =====================================================================
#  场景预设（一键切换整套场景配置）
# =====================================================================
SCENE_PRESETS = {
    # 论文原始设置：15 个大石头 + 985 个随机碎石 + 4 棵树（= 原 num_stones 1000）
    "paper_full": dict(
        SCENE_ENABLE_STONES=True, SCENE_ENABLE_BIG_STONES=True,
        SCENE_NUM_SMALL_STONES=985, SCENE_ENABLE_TREES=True,
        SCENE_TERRAIN_PRESET="custom",
    ),
    # 导航评测推荐：只留 15 个大石头 + 树，显存骤降，num_envs 可开大
    "nav_big_only": dict(
        SCENE_ENABLE_STONES=True, SCENE_ENABLE_BIG_STONES=True,
        SCENE_NUM_SMALL_STONES=0, SCENE_ENABLE_TREES=True,
        SCENE_TERRAIN_PRESET="custom",
    ),
    # 只有大石头，不要树
    "stones_only": dict(
        SCENE_ENABLE_STONES=True, SCENE_ENABLE_BIG_STONES=True,
        SCENE_NUM_SMALL_STONES=0, SCENE_ENABLE_TREES=False,
        SCENE_TERRAIN_PRESET="custom",
    ),
    # 空场景：无任何障碍物，用于单独调试底层步态
    "empty": dict(
        SCENE_ENABLE_STONES=False, SCENE_ENABLE_BIG_STONES=False,
        SCENE_NUM_SMALL_STONES=0, SCENE_ENABLE_TREES=False,
        SCENE_TERRAIN_PRESET="flat",
    ),
    # ⚡ 调试/回归专用：关掉大树 + 大小石头，但【保留 custom 地形】。
    #   与 "empty" 的区别：empty 会把地形换成平地，导航路线就没意义了；
    #   这个只砍掉障碍物，地形、出生点、waypoints 全部不变，
    #   所以能用来快速验证代码改动是否跑通，又不影响导航逻辑本身。
    #   实测收益：省掉 4 棵树 186 万面的碰撞 mesh cooking（构建 ~40s 里的 28.6s）
    #   和 985 个碎石 actor，构建与每步耗时都大幅下降。
    "no_obstacles": dict(
        SCENE_ENABLE_STONES=False, SCENE_ENABLE_BIG_STONES=False,
        SCENE_NUM_SMALL_STONES=0, SCENE_ENABLE_TREES=False,
        SCENE_TERRAIN_PRESET="custom",
    ),
}
#: 只改这一行即可切换场景；设为 None 表示不套用预设，直接用上面的单项开关
ACTIVE_SCENE_PRESET = "no_obstacles"


def apply_scene_preset(name):
    """把场景预设写进本模块全局命名空间（用法同 apply_preset）。"""
    if name is None:
        return None
    if name not in SCENE_PRESETS:
        raise KeyError("未知场景预设: %r，可选: %s" % (name, sorted(SCENE_PRESETS.keys())))
    for key, value in SCENE_PRESETS[name].items():
        globals()[key] = value
    globals()["ACTIVE_SCENE_PRESET"] = name
    # 预设里可能改了地形/石头，必须重算这两个派生值
    globals()["SCENE_TERRAIN_SEQUENCE"] = resolve_terrain_sequence()
    (globals()["SCENE_NUM_BIG_STONES"], globals()["SCENE_NUM_SMALL_STONES_TOTAL"],
     globals()["SCENE_NUM_STONES"],
     globals()["SCENE_BIG_STONE_INDICES"]) = resolve_stone_counts()
    return name


def resolve_terrain_sequence(preset=None, custom=None, pad=None):
    """把 SCENE_TERRAIN_PRESET 解析成 terrain_sequence 列表。"""
    preset = SCENE_TERRAIN_PRESET if preset is None else preset
    custom = SCENE_TERRAIN_SEQUENCE_CUSTOM if custom is None else custom
    pad = SCENE_TERRAIN_PAD if pad is None else pad
    if preset == "custom":
        return list(custom)
    if preset not in TERRAIN_IDS:
        raise KeyError("未知地形预设: %r，可选: %s 或 'custom'"
                       % (preset, sorted(TERRAIN_IDS.keys())))
    return [TERRAIN_IDS[preset]] * pad


def resolve_stone_counts(enable=None, big=None, small=None, positions=None):
    """返回 (num_big_stones, num_small_stones, num_stones, big_stone_indices)。

    大小石头数量已解耦：底层按“先大后小”的顺序创建 actor，
    大石头恒占 actor 索引 1..num_big（见 legged_robot.py 的创建循环），
    big_stone_indices 仅为兼容旧代码而保留。
    """
    enable = SCENE_ENABLE_STONES if enable is None else enable
    big = SCENE_ENABLE_BIG_STONES if big is None else big
    small = SCENE_NUM_SMALL_STONES if small is None else small
    positions = SCENE_BIG_STONE_POSITIONS if positions is None else positions
    if not enable:
        return 0, 0, 0, []
    num_big = len(positions) if big else 0
    num_small = max(0, int(small)) if small is not None else 0
    return num_big, num_small, num_big + num_small, list(range(num_big))


def resolve_tree_assets(all_assets, whitelist=None):
    """按 SCENE_TREE_WHITELIST 过滤 static_obstacles.assets。"""
    whitelist = SCENE_TREE_WHITELIST if whitelist is None else whitelist
    if not whitelist:
        return list(all_assets)
    keep = set(whitelist)
    return [a for a in all_assets if a.get("name") in keep]


apply_scene_preset(ACTIVE_SCENE_PRESET)

#: go2_config.py 直接取用的最终值（已解析，勿在此之后再改开关）
SCENE_TERRAIN_SEQUENCE = resolve_terrain_sequence()
(SCENE_NUM_BIG_STONES, SCENE_NUM_SMALL_STONES_TOTAL,
 SCENE_NUM_STONES, SCENE_BIG_STONE_INDICES) = resolve_stone_counts()


def check_consistency(verbose=True):
    """校验必须互相一致的感知/相机参数，返回“真错误”列表。

    这几组量原本分散在 legged_robot.py（相机 fov/机位）、pathplanner.py
    （跟随机位、深度裁剪）和 BEVMapper（fov、cam_height、max_depth）里，
    靠“数值碰巧相同”才正确。改成集中配置后，在这里统一校验。

    输出分两类，避免把预期配置也报成“未通过”：
      errors —— 真的互相矛盾/会算错，必须改；
      notes  —— 只是提醒当前行为（如 nav_only 主动跳过深度读取），可忽略。
    返回值仍只含 errors，保持向后兼容；notes 仅在 verbose 时打印，
    也可用 check_consistency_with_notes() 一并取到。
    """
    problems = []
    notes = []
    if abs(CAM_HORIZONTAL_FOV - BEV_HFOV_DEG) > 1e-6:
        problems.append(
            "CAM_HORIZONTAL_FOV(%s) 必须等于 BEV_HFOV_DEG(%s)，否则深度图反投影到 BEV 会错位"
            % (CAM_HORIZONTAL_FOV, BEV_HFOV_DEG))
    if abs(CAM_FOLLOW_HEIGHT - BEV_CAM_HEIGHT) > 1e-6:
        problems.append(
            "CAM_FOLLOW_HEIGHT(%s) 必须等于 BEV_CAM_HEIGHT(%s)，BEV 反投影按该高度算地面距离"
            % (CAM_FOLLOW_HEIGHT, BEV_CAM_HEIGHT))
    if CAM_ENABLE is False and (BEV_X or BEV_Y):
        problems.append("CAM_ENABLE=False 时 BEV 建图拿不到深度图，导航会退化")
    if CAM_READ_ENABLE is False:
        need_depth = (BEV_ENABLE_UPDATE or SAVE_CAMERA_IMAGES
                      or SAVE_RGB_IMAGE or SAVE_DEPTH_IMAGE
                      or LIVE_SHOW_CAMERA_IMAGES or SAVE_BEV_FMM_IMAGES)
        if need_depth:
            problems.append(
                "CAM_READ_ENABLE=False 但开了 BEV_ENABLE_UPDATE / 存图 / 实时相机窗口，"
                "这些都需要深度图；请设 CAM_READ_ENABLE=True，否则拿到的是空图")
        elif not BEV_ENABLE_UPDATE:
            notes.append(
                "CAM_READ_ENABLE=False（已跳过深度读取）：仅当 BEV_ENABLE_UPDATE=False "
                "且不需要存/看相机图时才可这样设，导航与指标不受影响")
    if LIVE_VIEWER_SYNC is False and (LIVE_DRAW_VIEWER_OVERLAY or LIVE_FOLLOW_VIEWER_CAMERA):
        notes.append(
            "LIVE_VIEWER_SYNC=False 时 viewer 窗口不逐帧刷新，"
            "LIVE_DRAW_*/跟随相机 看不到效果（想肉眼看请设 LIVE_VIEWER_SYNC=True）")
    if LIVE_DRAW_INTERVAL is not None and int(LIVE_DRAW_INTERVAL) < 1:
        problems.append("LIVE_DRAW_INTERVAL 必须 >= 1（当前 %s）" % LIVE_DRAW_INTERVAL)
    if BEV_ENABLE_UPDATE is False:
        notes.append(
            "BEV_ENABLE_UPDATE=False：深度图未喂给 BEVMapper，障碍图恒为空，"
            "上层规划器实际不避障（保持原代码行为；如需启用深度感知请设为 True）")
    if SCENE_ENABLE_STONES and SCENE_NUM_STONES <= 0:
        problems.append("SCENE_ENABLE_STONES=True 但石头总数为 0（检查大/小石头开关）")
    if len(SCENE_TERRAIN_SEQUENCE) < 3:
        problems.append(
            "SCENE_TERRAIN_SEQUENCE 长度(%d) 必须 >= terrain.num_rows(3)，否则 sequence_terrain 会越界"
            % len(SCENE_TERRAIN_SEQUENCE))
    check_consistency._last_notes = notes
    if verbose:
        if problems:
            print("[viz_config] ⚠ 配置一致性检查未通过:")
            for item in problems:
                print("   - " + item)
        else:
            print("[viz_config] 配置一致性检查通过 ✓")
        for item in notes:
            print("[viz_config] 提示: " + item)
    return problems


def check_consistency_with_notes():
    """返回 (errors, notes)。errors 同 check_consistency() 的返回值。"""
    errors = check_consistency(verbose=False)
    return errors, list(getattr(check_consistency, "_last_notes", []))


def scene_summary():
    """一行场景摘要，启动时打印确认。"""
    if not SCENE_ENABLE_STONES:
        stones = "石头=关"
    else:
        stones = "石头=%d(大%d+小%d,%s)" % (
            SCENE_NUM_STONES, SCENE_NUM_BIG_STONES, SCENE_NUM_SMALL_STONES_TOTAL,
            "静态" if SCENE_SMALL_STONES_STATIC else "动态")
    return "scene=%s | %s | 树=%s | 地形=%s" % (
        ACTIVE_SCENE_PRESET or "custom", stones,
        "开" if SCENE_ENABLE_TREES else "关",
        SCENE_TERRAIN_PRESET)


# =====================================================================
#  十一·B、机器人出生点（导航起点）
# ---------------------------------------------------------------------
#  原代码把出生点写死成 (6.0, 9.0, 0.4)，并注释掉了按 env_origins 分布的逻辑，
#  所以无论 num_envs 多大，所有机器人都从同一个世界坐标出发。
#  这对导航评测是必要的（起点一致才可复现），但训练时会让所有 env 挤在一点。
# =====================================================================
#: False = 用 SPAWN_FIXED_POSITION 固定出生点（导航评测，原行为）
#: True  = 按课程/地形算出的 env_origins 分布（训练时更合理）
SPAWN_USE_ENV_ORIGINS = False
#: 固定出生点（世界坐标 x, y, z）。⚠ 必须与 WAYPOINTS 所在区域相容。
SPAWN_FIXED_POSITION = (6.0, 9.0, 0.4)
#: SPAWN_USE_ENV_ORIGINS=True 时，xy 随机抖动幅度 [m]
SPAWN_ORIGIN_XY_JITTER = 1.0
#: 重置时给的随机初速度幅度（0 = 静止起步）
SPAWN_INIT_VELOCITY_RANGE = 1.0


# =====================================================================
#  十二、终止判定（摔倒 / 超时 / 掉落 / 卡住）
# ---------------------------------------------------------------------
#  对应 legged_robot.py 的 check_termination()。
#  这些阈值原本是写死在 legged_robot.py 里的魔法数（-0.5 / 0.3 / 0.05 / 1.0），
#  现在集中到这里，go2_config.termination 继承 TERMINATION_CFG。
#  ⚠ 这一节会影响训练与评测结果，调参前请先想清楚，别当成纯显示开关。
# =====================================================================
TERM_ENABLE_COLLISION = True       #: 终止部位碰撞即判摔倒
TERM_COLLISION_FORCE = 1.0         #: 接触力阈值 [N]
TERM_ENABLE_TIMEOUT = True         #: episode 超时终止
TERM_ENABLE_CLIFF_FALL = True      #: 掉落悬崖终止
TERM_CLIFF_FALL_HEIGHT = -0.5      #: base 低于该高度判掉落 [m]
TERM_ENABLE_STUCK = True           #: 卡住终止（导航评测很依赖它，否则会一直空转到超时）
TERM_STUCK_CMD_SPEED_MIN = 0.3     #: 命令线速度大于该值才算“应该动”[m/s]
TERM_STUCK_ACTUAL_SPEED_MAX = 0.05 #: 实际线速度小于该值才算“没动”[m/s]
TERM_STUCK_TIME_S = 1.0            #: 持续多久判定为卡住 [s]


class TERMINATION_CFG:
    """把上面的 TERM_* 开关打包成 config 类，供 go2_config.termination 继承。

    legged_robot.py 通过 cfg.termination.<字段> 读取，字段名必须与之一致。
    """
    enable_collision = TERM_ENABLE_COLLISION
    collision_force_threshold = TERM_COLLISION_FORCE
    enable_timeout = TERM_ENABLE_TIMEOUT
    enable_cliff_fall = TERM_ENABLE_CLIFF_FALL
    cliff_fall_height = TERM_CLIFF_FALL_HEIGHT
    enable_stuck = TERM_ENABLE_STUCK
    stuck_cmd_speed_min = TERM_STUCK_CMD_SPEED_MIN
    stuck_actual_speed_max = TERM_STUCK_ACTUAL_SPEED_MAX
    stuck_time_s = TERM_STUCK_TIME_S


# =====================================================================
#  十三、树 / 静态障碍物性能（构建耗时的真正瓶颈）
# ---------------------------------------------------------------------
#  ⚠ 实测：整个场景构建 43.8s 里，load_asset 占 40.7s（93%），
#     其中 4 棵树的 PhysX mesh cooking 就吃掉 28.6s：
#         deadwood(66万面) 12.8s | snow_tree(47万面) 9.1s | oak_tree(26万面) 6.7s
#     而 985 个碎石 stone.urdf(442面) 只要 0.01s，create_actor 8040 次仅 0.1s。
#     所以“构建慢”是树的高模 mesh 造成的，与石头数量基本无关。
# =====================================================================
#: 同一 URDF 被多个 spec 引用时复用已加载 asset（go2_config 里 snow_tree 出现两次，
#: 原来会重复 cooking 一次，白丢 ~9s）。实测 43.8s -> 34.3s，行为完全等价。
SCENE_TREE_REUSE_DUPLICATE_ASSETS = True
#: 树的 vhacd 凸分解。⚠ 实测默认参数反而更慢（33.6s -> 100.0s），保持 None 关闭。
SCENE_TREE_VHACD = None
#: 只想快速调试步态/规划时，可用 SCENE_ENABLE_TREES=False 跳过 28.6s 的树加载。


# =====================================================================
#  十四、高层规划器（BEV 建图 / FMM / RRT* / A*+DWA）
# ---------------------------------------------------------------------
#  原来这些数值全部写死在 pathplanner.py 的 BEVMapper(...) / FMMGradientController(...)
#  构造调用里，改一个阈值就要翻代码。现在集中到这里，
#  pathplanner.py 只负责把它们透传给各控制器。
# =====================================================================

# ---------- 深度相机传感器（legged_robot 创建，BEV 建图的数据源）----------
CAM_ENABLE = True               #: 是否在 env0 上创建相机
CAM_WIDTH = 1280                #: 图像宽 [px]
CAM_HEIGHT = 720                #: 图像高 [px]
#: 水平视场角 [deg]。⚠ 必须等于 BEV_HFOV_DEG，否则深度图反投影到 BEV 会错位。
CAM_HORIZONTAL_FOV = 90.0
#: 初始机位（仅第一帧生效，之后每步按 base 朝向重写）
CAM_INIT_FORWARD = 0.3
CAM_INIT_LOOKAT_FORWARD = 1.3
CAM_INIT_HEIGHT = 0.6

# ---------- 感知模块每步的相机跟随机位（相对 base）----------
#: ⚠ CAM_FOLLOW_HEIGHT 应与 BEV_CAM_HEIGHT 一致：BEV 反投影假设相机在该高度。
CAM_FOLLOW_FORWARD = 0.3        #: 相机在 base 前方多远 [m]
CAM_FOLLOW_HEIGHT = 0.25        #: 相机相对 base 的高度 [m]
CAM_FOLLOW_LOOKAT_FORWARD = 1.0 #: 注视点在 base 前方多远 [m]

# ---------- BEV 高度图建图（深度图 -> 障碍栅格）----------
BEV_X = 6.0                     #: 前方感知范围 [m]
BEV_Y = 6.0                     #: 左右感知范围 [m]
BEV_RES = 0.05                  #: 栅格分辨率 [m/cell]
BEV_CAM_HEIGHT = 0.25           #: 深度相机离地高度 [m]
BEV_HFOV_DEG = 90.0             #: 相机水平视场角 [deg]（需与 legged_robot 相机 fov 一致）
BEV_MAX_DEPTH = 6.0             #: 最大有效深度 [m]，超过视为无效
BEV_MIN_DEPTH = 0.05            #: 最小有效深度 [m]，过近的点丢弃
BEV_V_START_RATIO = 0.55        #: 深度图纵向采样起始比例（跳过上半张天空区域）
BEV_HEIGHT_RANGE_THRESH = 0.2   #: 核心阈值：单栅格内 max_z-min_z 超过它即判为障碍 [m]
BEV_MIN_POINTS = 8              #: 单栅格至少多少像素才可信（抗噪）

# ---------- BEV 更新总开关（⚠ 重要，见下方说明）----------
#: 是否把深度图喂给 BEVMapper（即调用 self.bev.update(depth_pos)）。
#: ⚠⚠ 原代码里这一行是被注释掉的（pathplanner.py 的 _update_camera_and_bev 中
#:    "# self.bev.update(depth_pos)"），且全仓库再无其它调用点，
#:    意味着 BEV 障碍图**恒为全 0**：上层规划器看不到任何障碍，FMM 在空障碍图上
#:    退化成“直线朝 waypoint 走”。
#:    实测对比（1 env，朝大石头走 50 步）：
#:        不调用 update -> 障碍格数 0；调用 update -> 障碍格数 21（深度有效像素 18.9 万）
#: 默认 False = 完全保持你现在的行为与既有实验数值。
#: 若要恢复“深度图像 -> BEV -> 避障”这条论文主链路，改成 True（会改变所有导航指标）。
BEV_ENABLE_UPDATE = False

# ---------- 障碍记忆 / 时间融合（TFAM）----------
#: 给规划器的障碍图用“即时帧”还是“即时 ∪ 历史记忆”。
#: True  = 时序融合（记忆叠加，遮挡后仍记得障碍）
#: False = 只用当前帧（原代码的实际行为，注释里那行 return occ_for_planner 被注释掉了）
BEV_ENABLE_MEMORY_FUSION = False
BEV_MEMORY_MAX_AGE = 30         #: 记忆寿命 [帧]，约 1 秒（dt≈0.03）

# ---------- FMM 梯度跟随（本文方法）----------
FMM_MAX_WZ = 1.0                #: 角速度上限 [rad/s]
FMM_YAW_K = 2.0                 #: 期望方向角 -> wz 的比例增益
FMM_LOOKAHEAD_M = 0.6           #: 前视取梯度距离 [m]，太近会导致 dc≈0
FMM_INFLATE_RADIUS_M = 0.25     #: 障碍膨胀半径 [m]，防止擦边撞
FMM_GOAL_MIN_FORWARD_M = 0.10   #: goal 在身后时钳到前方的最小值 [m]

# ---------- RRT*（对比算法）----------
RRT_MAX_WZ = 1.0
RRT_YAW_K = 2.0
RRT_LOOKAHEAD_M = 0.6
RRT_INFLATE_RADIUS_M = 0.25

# ---------- A* + DWA（对比算法）----------
DWA_MAX_VX = 1.0                #: 线速度上限 [m/s]
DWA_MAX_WZ = 1.2                #: 角速度上限 [rad/s]
DWA_MAX_ACC_VX = 2.0            #: 线加速度上限 [m/s^2]
DWA_MAX_ACC_WZ = 3.0            #: 角加速度上限 [rad/s^2]
DWA_DT = 0.1                    #: 预测步长 [s]
DWA_PREDICT_TIME = 1.2          #: 预测时域 [s]
DWA_INFLATE_RADIUS_M = 0.25     #: 撞墙时可上调至 0.3

# ---------- 底层下发的速度指令 ----------
#: play1.py 导航时下发的 vx。原代码写死 1.0，只把 FMM 算出的 wz 叠加上去。
NAV_FORWARD_VX = 1.0
#: 到达 waypoint 的判定半径 [m]
NAV_GOAL_REACH_THRESH_M = 0.3
#: 跑完整条路线后停留几秒（便于截图/录屏）
NAV_FINAL_HOLD_SECONDS = 5.0

# ---------- 评测指标口径 ----------
#: 平滑度统计的有效区间（原代码用 0 < current_smooth < 1.0 过滤无效路径）
METRIC_SMOOTH_VALID_MIN = 0.0
METRIC_SMOOTH_VALID_MAX = 1.0
#: 失败且无有效路径时记的平滑度兜底值（原代码写死 1.5）
METRIC_SMOOTH_FALLBACK = 1.5
#: 路径点少于该数量时不计算平滑度（原代码写死 5）
METRIC_SMOOTH_MIN_POINTS = 5
#: 路径总长度（米）小于该值时视为“几乎没动”，不计算平滑度（原代码写死 0.01）
METRIC_SMOOTH_MIN_PATH_LEN_M = 0.01


# =====================================================================
#  十五、play.py（底层步态）速度指令与腿部特写相机机位
# =====================================================================
#: play.py 默认下发的速度指令 [m/s, m/s, rad/s]
PLAY_VEL_X = 1.0
PLAY_VEL_Y = 0.0
PLAY_VEL_YAW = 0.0
#: 腿部特写相机相对 base 的机位（前/左/上，单位 m）
LEG_CAMERA_FORWARD_M = 1.4
LEG_CAMERA_LEFT_M = 0.5
LEG_CAMERA_HEIGHT_M = 0.45
#: 腿部特写注视点（前/上，单位 m）
LEG_CAMERA_LOOKAT_FORWARD_M = 0.10
LEG_CAMERA_LOOKAT_HEIGHT_M = 0.15

# ---------- play.py 日志与运行步数（原代码写死在脚本里） ----------
#: 记录哪只机器狗的状态曲线（多 env 时的索引）
PLAY_LOG_ROBOT_INDEX = 0
#: 记录哪个关节的状态曲线
PLAY_LOG_JOINT_INDEX = 1
#: 记录多少步状态后画曲线（0 = 不画，省时间省内存）
PLAY_LOG_STATE_STEPS = 100
#: 总运行步数 = max_episode_length × 该倍数（多跑一点确保 episode 收尾）
PLAY_TOTAL_STEPS_MULTIPLIER = 1.1
#: 是否在结束时画 Logger 状态曲线（关掉可省掉 matplotlib 开销）
PLAY_PLOT_STATES = True


# =====================================================================
#  预设定义（覆盖上面的默认值）
# =====================================================================
PRESETS = {
    # 只跑导航：关掉一切渲染/出图/数据记录，机器正常导航出指标，最省资源。
    # ⚠ 远程桌面(向日葵/VNC)下请优先用这个，viewer 逐帧重绘会把整机拖死。
    "nav_only": dict(
        SAVE_CAMERA_IMAGES=False, SAVE_RGB_IMAGE=False, SAVE_DEPTH_IMAGE=False,
        SAVE_BEV_FMM_IMAGES=False,
        SAVE_TRIAL_ROUTE_IMAGE=False, SAVE_TRIAL_ROUTE_LOCAL_FMM_SEGMENTS=False,
        SAVE_TRAJECTORY_NPY=False, SAVE_SUMMARY_TRAJECTORY_PLOT=False,
        LIVE_SHOW_CAMERA_IMAGES=False, LIVE_SHOW_BEV_FMM_WINDOW=False,
        LIVE_DRAW_VIEWER_OVERLAY=False, LIVE_DRAW_WAYPOINTS=False,
        LIVE_DRAW_TRAJECTORY=False, LIVE_DRAW_FMM_PATH=False,
        LIVE_FOLLOW_VIEWER_CAMERA=False, LIVE_HOLD_FINAL_ROUTE=False,
        LIVE_VIEWER_SYNC=False, LIVE_DRAW_INTERVAL=1,
        CAM_READ_ENABLE=False,
        LEG_CLOSEUP_CAPTURE=False, RECORD_FRAMES=False, MOVE_CAMERA=False,
        METRICS_PRINT_STATS=True,
        RENDER=False,
    ),
    # 全部关闭：不存任何文件、不画任何叠加。跑得最快、显存最省。
    "off": dict(
        SAVE_CAMERA_IMAGES=False, SAVE_RGB_IMAGE=False, SAVE_DEPTH_IMAGE=False,
        SAVE_BEV_FMM_IMAGES=False,
        SAVE_TRIAL_ROUTE_IMAGE=False, SAVE_TRIAL_ROUTE_LOCAL_FMM_SEGMENTS=False,
        SAVE_TRAJECTORY_NPY=False, SAVE_SUMMARY_TRAJECTORY_PLOT=False,
        LIVE_SHOW_CAMERA_IMAGES=False, LIVE_SHOW_BEV_FMM_WINDOW=False,
        LIVE_DRAW_VIEWER_OVERLAY=False, LIVE_DRAW_WAYPOINTS=False,
        LIVE_DRAW_TRAJECTORY=False, LIVE_DRAW_FMM_PATH=False,
        LIVE_FOLLOW_VIEWER_CAMERA=False, LIVE_HOLD_FINAL_ROUTE=False,
        LIVE_VIEWER_SYNC=False, LIVE_DRAW_INTERVAL=1,
        CAM_READ_ENABLE=False,
        LEG_CLOSEUP_CAPTURE=False, RECORD_FRAMES=False,
        RENDER=False,
    ),
    # 只看窗口：IsaacGym 跟随视角 + 画目标点和 FMM 路径，不存任何文件。
    # CAM_READ_ENABLE=False 是刻意的：相机每步渲染是真凶级内存泄漏（见文件头），
    # 关掉后画面照常刷新、机器狗照常走，且原代码本就没用深度图建图，功能等价。
    "viewer_only": dict(
        SAVE_CAMERA_IMAGES=False, SAVE_BEV_FMM_IMAGES=False,
        SAVE_TRIAL_ROUTE_IMAGE=False, SAVE_TRAJECTORY_NPY=False,
        SAVE_SUMMARY_TRAJECTORY_PLOT=False,
        LIVE_SHOW_CAMERA_IMAGES=False, LIVE_SHOW_BEV_FMM_WINDOW=False,
        LIVE_DRAW_VIEWER_OVERLAY=True, LIVE_DRAW_WAYPOINTS=True,
        LIVE_DRAW_TRAJECTORY=True, LIVE_DRAW_FMM_PATH=True,
        LIVE_FOLLOW_VIEWER_CAMERA=True, LIVE_HOLD_FINAL_ROUTE=False,
        LIVE_VIEWER_SYNC=True, LIVE_DRAW_INTERVAL=1,
        CAM_READ_ENABLE=False,
        LEG_CLOSEUP_CAPTURE=False, RECORD_FRAMES=False,
        RENDER=True,
    ),
    # 论文出图：存三联图 + 总览图 + 轨迹 npy + 汇总图，但不开实时弹窗（省时间）。
    # ⚠ 这个预设必须读相机（要存 RGB/Depth 和三联图），所以 CAM_READ_ENABLE=True，
    #   相机泄漏会回归：请坐本机跑、调小 NUM_TRIALS、别开远程桌面长时间挂着。
    "paper": dict(
        SAVE_CAMERA_IMAGES=False, SAVE_RGB_IMAGE=True, SAVE_DEPTH_IMAGE=True,
        SAVE_CAMERA_IMAGE_INTERVAL=50,
        SAVE_BEV_FMM_IMAGES=True, SAVE_BEV_FMM_INTERVAL=5,
        BEV_FMM_SHOW_POTENTIAL=False,
        SAVE_TRIAL_ROUTE_IMAGE=True, SAVE_TRIAL_ROUTE_LOCAL_FMM_SEGMENTS=False,
        SAVE_TRAJECTORY_NPY=True, SAVE_SUMMARY_TRAJECTORY_PLOT=True,
        LIVE_SHOW_CAMERA_IMAGES=False, LIVE_SHOW_BEV_FMM_WINDOW=False,
        LIVE_DRAW_VIEWER_OVERLAY=True, LIVE_DRAW_WAYPOINTS=True,
        LIVE_DRAW_TRAJECTORY=True, LIVE_DRAW_FMM_PATH=True,
        LIVE_FOLLOW_VIEWER_CAMERA=True, LIVE_HOLD_FINAL_ROUTE=False,
        LIVE_VIEWER_SYNC=True, LIVE_DRAW_INTERVAL=1,
        CAM_READ_ENABLE=True,
        LEG_CLOSEUP_CAPTURE=False, RECORD_FRAMES=False,
        RENDER=True,
    ),
    # 全开：调试用，最慢。同样需要相机（要存 RGB/Depth + 三联图）。
    "all": dict(
        SAVE_CAMERA_IMAGES=True, SAVE_RGB_IMAGE=True, SAVE_DEPTH_IMAGE=True,
        SAVE_BEV_FMM_IMAGES=True, BEV_FMM_SHOW_POTENTIAL=True,
        SAVE_TRIAL_ROUTE_IMAGE=True, SAVE_TRIAL_ROUTE_LOCAL_FMM_SEGMENTS=True,
        SAVE_TRAJECTORY_NPY=True, SAVE_SUMMARY_TRAJECTORY_PLOT=True,
        LIVE_SHOW_CAMERA_IMAGES=True, LIVE_SHOW_BEV_FMM_WINDOW=True,
        LIVE_DRAW_VIEWER_OVERLAY=True, LIVE_DRAW_WAYPOINTS=True,
        LIVE_DRAW_TRAJECTORY=True, LIVE_DRAW_FMM_PATH=True,
        LIVE_FOLLOW_VIEWER_CAMERA=True, LIVE_HOLD_FINAL_ROUTE=True,
        LEG_CLOSEUP_CAPTURE=True, RECORD_FRAMES=False,
        LIVE_VIEWER_SYNC=True, LIVE_DRAW_INTERVAL=1,
        CAM_READ_ENABLE=True,
        RENDER=True,
    ),
}

def apply_preset(name):
    """把预设写进本模块的全局命名空间。

    因为 pathplanner.py / play.py 是在 import 时通过 `import *` 取走这些名字，
    所以必须在模块加载阶段就完成覆盖（见文件末尾）。
    """
    if name not in PRESETS:
        raise KeyError("未知预设: %r，可选: %s" % (name, sorted(PRESETS.keys())))
    for key, value in PRESETS[name].items():
        globals()[key] = value
    return name


apply_preset(ACTIVE_PRESET)


def ensure_dirs():
    """按当前开关，预先把需要的输出目录建好。"""
    needed = []
    if SAVE_BEV_FMM_IMAGES:
        needed.append(SAVE_BEV_FMM_DIR)
        for sub in ("height", "obstacles", "potential"):
            needed.append(os.path.join(SAVE_BEV_FMM_DIR, sub))
    if SAVE_TRIAL_ROUTE_IMAGE:
        needed.append(SAVE_TRIAL_ROUTE_DIR)
    if SAVE_CAMERA_IMAGES:
        needed.append(SAVE_CAMERA_IMAGE_DIR)
    if SAVE_TRAJECTORY_NPY:
        needed.append(SAVE_TRAJECTORY_DIR)
    if SAVE_SUMMARY_TRAJECTORY_PLOT:
        needed.append(SAVE_SUMMARY_TRAJECTORY_PLOT_DIR)
    if LEG_CLOSEUP_CAPTURE:
        needed.append(LEG_CLOSEUP_DIR)
    for path in needed:
        os.makedirs(path, exist_ok=True)
    return needed


def summary():
    """返回一行人类可读的当前配置摘要，方便启动时打印确认。"""
    on = [name for name, value in sorted(globals().items())
          if name.isupper() and value is True]
    check_consistency(verbose=True)
    return "preset=%s | 已开启: %s\n%s" % (
        ACTIVE_PRESET, ", ".join(on) if on else "无", scene_summary())

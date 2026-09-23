# -*- coding: utf-8 -*-
"""
go2 可视化与输出统一配置
========================
把 play.py / play1.py / pathplanner.py 里散落的所有开关集中到这一个文件。
改开关只需编辑本文件，不必再去动 play.py 或 pathplanner.py。

【手动运行命令】以下命令均从仓库根目录执行，可直接复制到终端。

    cd /home/fabu/HIMLoco
    conda activate himloco
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

    # 1. 肉眼查看推荐：IsaacGym 窗口 + 双石窄门 + MPPI，只跑 1 次。
    #    注意：需要窗口时不要加 --headless。
    HIMLOCO_PRESET=viewer_only \
    HIMLOCO_SCENE_PRESET=narrow_gate \
    HIMLOCO_PLANNER_MODE=mppi \
    HIMLOCO_NUM_TRIALS=1 \
    python legged_gym/scripts/play1.py --task=go2

    # 2. 随机野外走廊测试：关闭 viewer、随机石头和树，适合跑数据。
    #    视觉导航不要添加 --headless：本项目在该参数下不会创建深度相机。
    HIMLOCO_PRESET=nav_only \
    HIMLOCO_SCENE_PRESET=random_corridor \
    HIMLOCO_PLANNER_MODE=mppi \
    HIMLOCO_SCENE_SEED=20260922 \
    HIMLOCO_NUM_TRIALS=10 \
    python legged_gym/scripts/play1.py --task=go2

    # 3. FMM 对照：与上条保持相同场景和 seed，只替换规划器。
    HIMLOCO_PRESET=nav_only \
    HIMLOCO_SCENE_PRESET=random_corridor \
    HIMLOCO_PLANNER_MODE=fmm \
    HIMLOCO_SCENE_SEED=20260922 \
    HIMLOCO_NUM_TRIALS=10 \
    python legged_gym/scripts/play1.py --task=go2

【常用命令参数】把需要的变量放在 python 命令前；未写的使用本文件默认值。

    HIMLOCO_PRESET=viewer_only       # 开窗口观察，不保存论文图片
    HIMLOCO_PRESET=nav_only          # 关闭 viewer，适合批量实验；仍不要加 --headless
    HIMLOCO_PRESET=paper             # 保存轨迹、统计结果及论文图片
    HIMLOCO_SCENE_PRESET=narrow_gate # 固定双石窄门，检查对准和侧擦
    HIMLOCO_SCENE_PRESET=random_corridor  # 随机大石头、树和离散高度地形
    HIMLOCO_SCENE_PRESET=blocker     # 固定挡路障碍，用于基础避障回归
    HIMLOCO_SCENE_PRESET=empty       # 平地且无障碍，仅检查步态/控制
    HIMLOCO_PLANNER_MODE=mppi        # MPPI 局部规划器（当前推荐）
    HIMLOCO_PLANNER_MODE=fmm         # FMM 基线，用于 A/B 对照
    HIMLOCO_NUM_TRIALS=10            # 本进程实验次数；肉眼查看建议设为 1
    HIMLOCO_SCENE_SEED=20260922      # 随机种子；A/B 对照必须保持一致
    HIMLOCO_TEMPORAL_MEMORY=1        # 1 开启时序障碍记忆，0 关闭做消融
    HIMLOCO_MPPI_SAMPLES=384         # MPPI 采样数；增大更慢，通常也更稳定
    HIMLOCO_FMM_SOFT_COST=1          # FMM 使用连续安全代价；0 为硬膨胀基线
    HIMLOCO_DEBUG_START_STEPS=30     # 输出起步阶段调试日志；0 表示关闭

    # 单行写法示例（效果与多行命令相同）：
    HIMLOCO_PRESET=viewer_only HIMLOCO_SCENE_PRESET=narrow_gate HIMLOCO_PLANNER_MODE=mppi HIMLOCO_NUM_TRIALS=1 python legged_gym/scripts/play1.py --task=go2

⚠⚠ 场景预设（ACTIVE_SCENE_PRESET）必须在【启动进程前】改好，不能在运行时调用
   apply_scene_preset() 切换：go2_config.py 里 `num_stones = VIZ.SCENE_NUM_STONES`
   这类写法在类定义求值时就把值固化了，而 go2_config 由 envs/__init__.py 在
   import 时加载。运行时再切只会改 viz_config 自己的全局，石头/树/地形仍按旧值
   创建，实验会在“空场景”里静默跑完、结论全错（本仓库 2026-09-16 踩过这个坑）。
   现在 apply_scene_preset() 检测到这种情况会打警告；可视化类开关
   （ACTIVE_PRESET / SAVE_* / LIVE_*）不受此限制，运行时改没问题。

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


def _env_bool(name, default):
    """Read an optional boolean experiment override without ambiguous truthiness."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError("%s must be one of 1/0, true/false, yes/no, on/off" % name)


def _env_int(name, default):
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else int(default)


def _env_float(name, default):
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else float(default)

# =====================================================================
#  预设选择（只改这一行）
# =====================================================================
#: 可选 "nav_only"(只跑导航) / "viewer_only"(能看机器狗动) / "off" / "paper" / "all"
#:
#: ⛔⛔ 卡死根因（2026-09-16 实测定位，本机向日葵远程桌面 + RTX 4060 8GB + 31GB 内存）⛔⛔
#:   真凶是【每步的相机渲染】render_all_camera_sensors + 把两张 1280x720 图拷回 CPU，
#:   不是 viewer 逐帧重绘，也不是石头/树。
#:   （当时的临时对策是 CAM_READ_ENABLE=False，但那等于放弃避障；
#:     现已改为“低分辨率 + 只读 depth + 感知降频”三重压制，见下方“现在的默认配置”。）
#:
#:   鉴别实验（同样无石头无树、同一 checkpoint、同一 1 trial）：
#:     · viewer_only + CAM_READ_ENABLE=True  -> ~74~139MB/s，step 600~750 到 8GB，被看门狗拦
#:     · viewer_only + CAM_READ_ENABLE=False ->  3.4MB/s，跑完，✅ 成功
#:     · nav_only    + CAM_READ_ENABLE=False ->  3.1MB/s，10 trial 峰值 5.9GB，10/10 ✅
#:   组件级探针进一步排除嫌疑：draw_viewer 2000 步仅 0.07MB/s、
#:   skfmm 3000 步 0.04MB/s、set_camera_location 0MB/s，均非泄漏源。
#:
#:   ⚠ 历史遗留：原代码里 bev.update() 本来就被注释掉了（BEV_ENABLE_UPDATE=False），
#:     深度图读出来从未被用于建图，所以上层规划器其实【一直没在避障】，
#:     FMM 在空障碍图上退化成“直线朝 waypoint 走”。
#:
#: ✅ 现在的默认配置是【真避障】：BEV_ENABLE_UPDATE=True，因此相机读取会自动打开
#:    （见 CAM_READ_ENABLE=None 的自动推导）。为了不让泄漏回归，配套做了三重压制：
#:      1) CAM_WIDTH/HEIGHT 降到 320x180   —— 单步拷贝量降到原来的 6.2%
#:      2) CAM_READ_RGB=False（自动推导）  —— 只拷 depth，不拷没人用的 RGB，再省一半
#:      3) PERCEPT_UPDATE_INTERVAL=4       —— 50Hz 主循环里每 4 步才渲染+建图一次，
#:                                             再降到 1/4（12.5Hz 感知，8cm 位移<2 栅格）
#:    三者叠加 ≈ 原来的 0.8%，即 1280x720 时的 7.37MB/步 -> 约 0.06MB/步。
#:
#:   结论：
#:     · 跑数据/回归 -> "nav_only"（无 viewer，最快最省）
#:     · 要肉眼看机器狗走 -> "viewer_only"（默认，画面正常刷新且真在避障）
#:     · 要出论文图（存 RGB/Depth、三联图）-> "paper"，它会自动开 CAM_READ_RGB=True，
#:       此时单步拷贝量翻倍，务必坐本机、调小 NUM_TRIALS、别开远程桌面长时间跑
#:     · 任何模式都有内置内存看门狗（见下节 SAFETY_MEM_*）兜底，不会拖死整机
#:
#:  ---------------------------------------------------------------------
#:  相机几何标定结论（2026-09-16，全部实测，非推断）
#:  ---------------------------------------------------------------------
#:  1) 深度语义：IsaacGym IMAGE_DEPTH 是【轴向深度】（沿光轴）。地面模型
#:     h/tan(θ) 误差 +0.6%，h/sin(θ) 误差 -1~-9.5%。故 BEV 反投影用严格针孔
#:     （BEV_USE_AXIAL_GEOM=True），像素级地面判据也改用 tan（旧 sin 是
#:     SPEED_MOD_USE_PIXEL_JUDGE 误报 99% 的根因）。
#:  2) 横向符号：单石头定向标定（石头放机器人左侧 1.0m）命中图像左半
#:     u∈[0,120] => 成像【不镜像】，du>0 = 机器人右侧。而 FMM/路径回投影
#:     约定“列号增大 = 左”，故反投影必须 y_left = -du*depth
#:     （BEV_LATERAL_SIGN=-1）。旧 +du 使障碍图左右镜像、FMM 朝障碍一侧转。
#:  3) 相机离地高度：policy 驱动实测 base 离地 = 0.254(足端)/0.278(terrain 真值)，
#:     旧值 0.369 无法复现；反投影残差拟合独立确认 h_off=0.25、h≈0.528。
#:     故 BEV_CAM_HEIGHT=0.528、ROBOT_BASE_HEIGHT_M=0.278。
#:  4) 近距盲区：平视时石头顶只比相机高 0.138m，落在 ROI 上沿外，1.1m 内必漏检
#:     （几何下界）。俯角可推近盲区，但俯角会让【最近可见地面】近至 0.35m，
#:     与机身自遮挡（前腿/胸口 0.35~0.5m）同带，产生 clearance 间歇尖峰。
#:     对策：相机前移 0.5m + BEV_MIN_RANGE_M=0.2（自遮挡带挡在掩蔽外、石头带保留）。
#:     pitch=25 + 前移机位实测不再顶死（blocker 进度 50%）。
#:  5) 天空污染：上游把深度 clip 到 BEV_MAX_DEPTH，天空恰好=6.0，`d>max_depth`
#:     判不掉；平视时反投影 x=6 出界无害，俯角时 x≈4.7 进界且 z≈-3.6m 成假障碍。
#:     故 BEV_CLIP_INVALID_EPS 必须 >0（尤其开俯角时）。
#:  6) 地形边界墙：heightfield 四周有 4m 高边界墙（y=11.4~11.9 等），起点 (6,9)
#:     距左墙仅 2.4m，落在 BEV 横向 ±3m 内。它是真地形不是 bug，但解释了
#:     一部分 wz 偏置；镜像修复后 FMM 会正确地把它当左墙并往右让。
#:  7) 航点管理三处修复（2026-09-16 wz/轨迹日志实测）：
#:     a) 越过航点平面即前进（NAV_ADVANCE_ON_CROSS）：绕过石头冲过航点后
#:        goal 被钳成纯横向 -> wz 饱和原地打转、0.25m 阈值永不满足。
#:     b) 中间航点宽阈值（NAV_MID_GOAL_REACH_THRESH_M=1.0）：goal 落进最小
#:        转弯圆(v/wz_max=1.0m)内会触发纯追踪极限环（wz=-1.0 连续 90 步、
#:        clearance=3.0 无障、全速跑圈）。
#:     c) 速度-转向耦合（SPEED_TURN_*）：FMM 假设全向点机器人，而盲走底层
#:        转弯半径=vx/wz；wz=±1+vx=1 时轨迹是 1m 圆弧、弯向障碍顶死。
#:        急转压 vx 到 0.3 后不再顶死（终止从 environment_termination 变 timeout），
#:        代价是变慢；blocker 八石窄廊仍 0/10，属上层-底层能力边界，如实报告。
#:  8) 障碍白名单（2026-09-17，BEV_OBSTACLE_ACTORS_ONLY=True）：
#:     BEV 判据不区分来源，no_obstacles 场景里稳定检出 ~55 格边界墙（4m 高、
#:     起点距左墙 2.4m、在 BEV ±3m 内），FMM 常年往右推、观感像“绕地形离散障碍”。
#:     实测地形凸起 hr≈0.01m 本就不会被检出，被检出的只有墙与 actor。
#:     白名单后轨迹各段横向偏离 0.22~0.53m（近似直线）。
#:     ⚠ 连带效应（已定位并修复）：occ≡0 后 clearance 恒 3.0，刹车也一起失效。
#:       故刹车改用【原始几何图】而非白名单图（BEV_CLEARANCE_USE_RAW_MAP=True）。
#:       曾以为"急转画 1m 圆弧打滑顶死"是主因，加了「急转无条件降速档」，
#:       实测证明它是净负收益（no_obstacles 0.90->0.00~0.10、stones_only 0.80->0.00），
#:       已降级为默认关闭的开关 SPEED_TURN_HARD_ENABLE=False。真正的死锁根因见第 9~10 条。
#:  9) 底层随机指令覆盖（2026-09-17 定位的【分层架构真 bug】）：
#:     go2_config.commands.resampling_time 原为 10s，policy dt=0.005*4=0.02s，
#:     于是 _post_physics_step_callback 每【500 步】调 _resample_commands，
#:     把上层规划器刚下发的 commands 重采样成随机值
#:     (lin_vel_x/y∈[-1,1]、ang_vel_yaw∈[-3.14,3.14])，直接覆盖规划器指令。
#:     执行顺序是 callback(随机覆盖) -> check_termination(用随机指令判卡住)
#:     -> compute_observations(策略当步观测到的也是随机指令)。
#:     铁证：诊断曾打印"指令 1.16/1.19/1.25 m/s"，而 NAV_FORWARD_VX=1.0，
#:     规划器根本不可能下发这么大的值。
#:     修复：EVAL_COMMAND_RESAMPLING_TIME=1e6（评测期等效关闭，见该处注释）。
#:  10) 停滞死锁与逃逸（2026-09-17 停滞取证探针实测）：
#:     修掉第 9 条后成功率反而从 0.58 掉到 0.17 —— 因为那个 bug 的随机转向恰好
#:     充当了【隐式逃逸机制】把机器狗从死锁里摇出来；即历史高分建立在 bug 之上。
#:     探针取证：机器狗【原地踏步死锁】（位置钉在 (16.16,5.02) 连续 50+ 步，
#:     实际 vx 在 ±0.1 抖动、yawrate 完全不跟踪 wz=-0.54、BEV 障碍图全零），
#:     上层认为"前方通畅"于是永远重复同一条指令 -> 死锁到 timeout。
#:     已排除的假设：不是撞障碍（接触力仅 46~96N）、不是陡坡（全域坡度中位 0.0%、
#:     p99 仅 5.5%；离散障碍地形 id=4 实际用 discrete_obstacle_height_fixed=0.01m）。
#:     修复：上层自己做停滞检测 + 倒退/原地转脱困（STALL_*，实测 0.07 -> 0.53）。
#:  11) 诊断/日志的测量污染（2026-09-17 修复）：
#:     _diagnose_termination 与 state_log 都在 env.step() 【返回后】才读 env.commands，
#:     而 step 内部 reset_idx 已无条件把 commands 重新随机化，读到的是重置后的噪声。
#:     故改为显式传当步真实下发的 cmd_vx/wz。另外 state_log.append 在主循环里被写了
#:     【两遍】（每步重复记录，第一份还是被污染的那份），已删掉重复块。
#:     注：base_lin_vel/base_ang_vel 在 reset_idx 之前算好且未重算，读它们仍有效，
#:     所以"实际 0.001、yawrate≈0"是真实值 —— 机器狗确实物理停住了。
#:
#:  📊 定稿实测（2026-09-17，各 15 trial，第 9~11 条修复后）：
#:        no_obstacles 0.53 / stones_only 0.27 / blocker_single 0.40 / blocker_two 0.33
#:     逃逸开 0.53 vs 关 0.07（差 0.46，远超二项噪声 ±0.12）；倒退段消融 0.25 vs 0.08。
#:     stones_only 失败主因是 base 接触力 512N 的真实碰撞（base_z=0.40m、姿态正常，
#:     胸口撞 0.76m 高石头侧面），不是判据误报；碰撞阈值放宽到 50N 可升到 0.42，
#:     但那属于改变成功判据定义，需与导师确认，见 TERM_COLLISION_FORCE 注释。
#:     ⚠ 方法论教训：10 trial 的二项标准差达 ±0.31，此前 0.90/0.80 与 0.40/0.30 的
#:       差异大多落在噪声内，同配置重复实测出现过 0.90 与 0.17 两端。凡关键结论一律
#:       用 ≥15 trial 并配开/关对照，不要靠单点数字定标。
ACTIVE_PRESET = "viewer_only"
#: 允许用环境变量在启动前覆盖上面的预设（与场景预设同机制），方便做 A/B 与回归：
#:       HIMLOCO_PRESET=nav_only python play1.py --task=go2
#:   留空/不设 = 完全按 ACTIVE_PRESET 走。可选值见 PRESETS 的键。
_PRESET_ENV = os.environ.get("HIMLOCO_PRESET", "").strip()
if _PRESET_ENV:
    ACTIVE_PRESET = _PRESET_ENV


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
LIVE_FOLLOW_VIEWER_CAMERA = False       #: viewer 相机跟随机器狗（适合直接录屏；默认关，用固定机位）
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
#:   所以设 False 与原行为【功能等价】，导航与指标完全不受影响。
#:
#: 取值：
#:   None = 自动（推荐）：只要有消费方需要深度图就自动开启，见 resolve_cam_read()。
#:          这样就不必在每个 PRESETS 里手工维护这个耦合关系。
#:   True / False = 手动强制（设 False 但有消费方时 check_consistency 会报错）
CAM_READ_ENABLE = None

#: 是否每步把 RGB 图也拷回 CPU（depth 之外的那一张）。
#: ⚠ 原代码无条件拷 depth+rgb 两张，但 RGB 只有 SAVE_RGB_IMAGE /
#:   LIVE_SHOW_CAMERA_IMAGES 才用得上。导航避障只需要 depth，
#:   白白多拷一张 = 泄漏量直接翻倍（320x180 float32 单张 0.23MB/步）。
#: 取值同 CAM_READ_ENABLE：None=自动（按消费方推导），True/False=手动强制。
CAM_READ_RGB = None

# ---------- 感知降频（每 N 步才渲染并更新一次 BEV）----------
#: 主循环是 50Hz（控制 dt≈0.02s），而深度渲染 + BEV 反投影是整条链路里最贵的
#: 两步。机器狗 1.0m/s 时每步只前进 2cm = 0.4 个 BEV 栅格（BEV_RES=0.05m），
#: 每步重建一次障碍图属于严重过采样。
#: 设为 N：每 N 步才读一次深度并 bev.update()，中间步直接复用上一次的障碍图
#: （BEVMapper 的 min_z/max_z 在两次 update 之间保持不变，规划器照常工作）。
#: 收益是【线性】的：GPU→CPU 拷贝量与 BEV 反投影耗时都除以 N。
#:   N=1  -> 每步更新（原始行为，0.46MB/步）
#:   N=4  -> 12.5Hz 感知，8cm 位移 < 2 栅格，避障精度基本无损（0.115MB/步）
#: ⚠ 不要设太大：N=10 时每步 20cm，接近 BEV 栅格的 4 倍，近距障碍可能来不及反应。
PERCEPT_UPDATE_INTERVAL = 4
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
#: 跑多少次独立导航试验。环境变量便于做小样本冒烟与批量 A/B，不改文件默认值：
#:   HIMLOCO_NUM_TRIALS=3 python play1.py --task=go2
#: 注意：本项目在 --headless 下不创建深度相机，视觉导航实验不能添加该参数。
NUM_TRIALS = int(os.environ.get("HIMLOCO_NUM_TRIALS", "10"))
NAV_EPISODE_LENGTH_S = 60.0  #: 24m 随机走廊含绕障，30~45s 容易把正常绕行误判超时
RENDER = True           #: 是否开 IsaacGym viewer 窗口
PLANNER_MODE = os.environ.get("HIMLOCO_PLANNER_MODE", "mppi").strip().lower()
#: 高层规划器："fmm"(基线) / "mppi"(FMM走廊+足迹MPPI) / "rrt" / "astar_dwa"
#: 导航目标点（世界坐标），论文用的是 4 个点：A/B/C + 终点
WAYPOINTS = [
    (11.0, 8.0),
    (15.0, 4.0),
    (19.0, 7.0),
    (23.0, 5.0),
]
#: 随机直线走廊实验：每个 trial 根据实际出生 y 生成唯一目标 (goal_x, start_y)。
NAV_RANDOM_STRAIGHT_ROUTE = False
NAV_STRAIGHT_GOAL_X = 30.0
#: 起步诊断日志步数；仅调试使用，环境变量覆盖，默认不打印。
NAV_DEBUG_START_STEPS = _env_int("HIMLOCO_DEBUG_START_STEPS", 0)
#: 固定世界坐标模式下的 trial 级出生点随机化；random_corridor 预设会开启。
SPAWN_RANDOMIZE_EACH_TRIAL = False
SPAWN_RANDOM_X_RANGE = (6.0, 6.0)
SPAWN_RANDOM_Y_RANGE = (3.0, 9.0)


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
#: 【指令重采样周期覆盖】评测期禁止底层随机重采样覆盖上层规划器的指令。
#:
#:   ⚠⚠ 这是分层架构里的一个【真 bug】，2026-09-17 定位，是此前所有 A/B 矩阵
#:   成功率剧烈波动（同配置 0.90 / 0.40 反复横跳）的根源：
#:
#:   go2_config.commands.resampling_time 原值 10s，policy dt = sim_dt(0.005)
#:   × decimation(4) = 0.02s，于是 legged_robot._post_physics_step_callback 里
#:       env_ids = (episode_length_buf % (10/0.02) == 0)
#:   每【500 步 = 10s】触发一次 _resample_commands()，把 commands 重采样为
#:       lin_vel_x ∈ [-1.0, 1.0]、lin_vel_y ∈ [-1.0, 1.0]、ang_vel_yaw ∈ [-3.14, 3.14]
#:   —— 直接【覆盖掉上层规划器刚下发的 cmd_vx / wz】。
#:
#:   更严重的是执行顺序（legged_robot.post_physics_step）：
#:       _post_physics_step_callback()   <- 这里随机覆盖 commands
#:       check_termination()             <- 卡住判据用的已是【随机】指令
#:       compute_observations()          <- 策略当步观测到的也是【随机】指令
#:   后果有二：
#:     1) 机器狗每 10s 被喂一次随机速度指令（可能是倒退、或 ±3.14rad/s 原地打转），
#:        底层据此动作 -> 摔倒 / 撞障碍 / 冲出航点，与上层规划完全脱钩；
#:     2) “卡住”终止判据 cmd_speed 用随机指令，产生大量【假卡住】误判。
#:
#:   实测铁证：终止诊断打印的“指令 1.18 / 1.25 m/s”【超过】NAV_FORWARD_VX=1.0，
#:   而规划器根本不可能下发这么大的值 -> 只能是重采样来的随机指令。
#:
#:   评测/导航时上层规划器应当【独占】速度指令权，故默认把周期设为极大值
#:   （等效关闭；reset_idx 里的初始重采样仍保留，下一步即被规划器覆盖）。
#:   ⚠ 不能用 1e9：legged_robot 里除数是 int(resampling_time/dt)=5e10，
#:     超过 int32 上限 2.1e9（episode_length_buf 是 int32），存在溢出风险。
#:     1e6s -> 5e7 步，远大于 max_episode_length(~1000 步)，等效关闭且安全。
#:   训练阶段请沿用 go2_config 的 10s 随机指令课程，不要改这里以外的东西。
EVAL_COMMAND_RESAMPLING_TIME = 1e6

#: play1.py 按此表逐项覆盖 env_cfg；键为 (cfg 段名, 属性名)
EVAL_OVERRIDES = (
    (("noise", "add_noise"), EVAL_OVERRIDE_ADD_NOISE),
    (("domain_rand", "randomize_friction"), EVAL_OVERRIDE_RANDOMIZE_FRICTION),
    (("domain_rand", "push_robots"), EVAL_OVERRIDE_PUSH_ROBOTS),
    (("domain_rand", "disturbance"), EVAL_OVERRIDE_DISTURBANCE),
    (("domain_rand", "randomize_payload_mass"), EVAL_OVERRIDE_RANDOMIZE_PAYLOAD_MASS),
    (("commands", "heading_command"), EVAL_OVERRIDE_HEADING_COMMAND),
    # 见上方长注释：阻止底层每 500 步随机覆盖上层规划器的速度指令
    (("commands", "resampling_time"), EVAL_COMMAND_RESAMPLING_TIME),
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
#: 随仿真进程创建大石头和树的位置。GPU PhysX 不支持启动后移动静态刚体，
#: 因此新布局必须通过重新启动 play1.py 生成，不能在同一进程的 reset 中搬动。
SCENE_RANDOMIZE_OBSTACLES_EACH_TRIAL = False
SCENE_RANDOM_OBSTACLE_X = (12.0, 24.0)
SCENE_RANDOM_OBSTACLE_Y = (0.0, 12.0)
#: scale=2 大石头实际宽约 1.0m；1.65m 中心距保留约 0.65m 净缝。
#: 原 1.4m 可能只剩 0.4m，低于 Go2 腿部动态包络，物理上不可稳定通过。
SCENE_RANDOM_OBSTACLE_MIN_SEPARATION_M = 1.65
SCENE_RANDOM_TREE_Z_OFFSET = 0.02
SCENE_RANDOM_CORRIDOR_BIG_STONE_COUNT = 20
#: 可用 HIMLOCO_SCENE_SEED 覆盖；相同 seed 可让不同算法复用完全相同的布局。
SCENE_RANDOM_SEED_BASE = _env_int("HIMLOCO_SCENE_SEED", 20260922)

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
TERRAIN_ROUGH_MAX_HEIGHT = 0.1
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
#:
#: ⚠ 实测发现（2026-09-16）：这套布局下 15 块石头只有 (16,5) 一块真正压在
#:   waypoint 直线上（距路径 0.20m），其余 14 块都在 1.34m 以外；
#:   而底层策略本来就能跨过孤立石头，实测各组轨迹离它最近仍有 0.95~1.45m。
#:   结果就是【这个场景根本不需要避障】：不避障（BEV_ENABLE_UPDATE=False）反而
#:   10/10 全过，开避障后 wz 指令变复杂、盲走底层更容易失稳，成功率更低。
#:   要做有意义的避障对照实验，请用下面的 SCENE_BLOCKER_STONE_POSITIONS。
SCENE_PAPER_STONE_POSITIONS = [
    (8, 1), (8, 5), (8, 9.4),
    (12, 2), (12, 10),
    (16, 1), (16, 5), (16, 7), (16, 9.4),
    (20, 2), (20, 4), (20, 8), (20, 10),
    (24, 7), (24, 9.4),
]

#: 【新增】把大石头直接压在 waypoint 连线上的布局，用于避障对照实验。
#: 由 WAYPOINTS（起点 (6,9) + 4 个目标点）的每段按 t=0.4 / 0.7 插值算出，
#: 保证石头【必然挡路】——不绕行就一定撞。这才测得出上层规划的价值。
#: 与 SCENE_BIG_STONE_POSITIONS 二选一：把选用的赋给前者即可（见 SCENE_PRESETS）。
#: 【论文难度梯度用】blocker 布局的子集：只保留第 1 段（seg0→1）上的挡路石头。
#:   用于“单石挡路 / 双石挡路 / 八石窄廊”的难度梯度对照实验。
SCENE_BLOCKER_SINGLE_POSITIONS = [
    (8.0, 8.6),
]
SCENE_BLOCKER_TWO_POSITIONS = [
    (8.0, 8.6), (9.5, 8.3),
]
# 两块 scale=2 大石头的实际 y 向宽度约 0.96m。下面中心距 1.70m，
# 碰撞几何之间净缝约 0.74m，缝中心严格位于 y=6.0。
SCENE_NARROW_GATE_POSITIONS = [
    (16.0, 4.6905), (16.0, 6.3905),
]
SCENE_BLOCKER_STONE_POSITIONS = [
    (8.0, 8.6), (9.5, 8.3),      # seg0→1
    (12.6, 6.4), (13.8, 5.2),    # seg1→2
    (16.6, 5.2), (17.8, 6.1),    # seg2→3
    (20.6, 6.2), (21.8, 5.6),    # seg3→4
]
#: 当前实际使用的大石头布局（None = 用论文原始布局 SCENE_PAPER_STONE_POSITIONS）。
#: 设为 "blocker" 走 SCENE_BLOCKER_STONE_POSITIONS；也可直接填一个坐标列表。
#: ⚠ 切换由 apply_scene_preset() 统一处理，下面这个派生值不要在别处手改。
SCENE_STONE_LAYOUT = None

# =====================================================================
#  场景预设（一键切换整套场景配置）
# =====================================================================
SCENE_PRESETS = {
    # 论文原始设置：15 个大石头 + 985 个随机碎石 + 4 棵树（= 原 num_stones 1000）
    "paper_full": dict(
        SCENE_STONE_LAYOUT=None,
        SCENE_ENABLE_STONES=True, SCENE_ENABLE_BIG_STONES=True,
        SCENE_NUM_SMALL_STONES=985, SCENE_ENABLE_TREES=True,
        SCENE_TERRAIN_PRESET="custom",
    ),
    # 导航评测推荐：只留 15 个大石头 + 树，显存骤降，num_envs 可开大
    "nav_big_only": dict(
        SCENE_STONE_LAYOUT=None,
        SCENE_ENABLE_STONES=True, SCENE_ENABLE_BIG_STONES=True,
        SCENE_NUM_SMALL_STONES=0, SCENE_ENABLE_TREES=True,
        SCENE_TERRAIN_PRESET="custom",
    ),
    # 只有大石头，不要树
    "stones_only": dict(
        SCENE_STONE_LAYOUT=None,
        SCENE_ENABLE_STONES=True, SCENE_ENABLE_BIG_STONES=True,
        SCENE_NUM_SMALL_STONES=0, SCENE_ENABLE_TREES=False,
        SCENE_TERRAIN_PRESET="custom",
    ),
    # 空场景：无任何障碍物，用于单独调试底层步态
    "empty": dict(
        SCENE_STONE_LAYOUT=None,
        SCENE_ENABLE_STONES=False, SCENE_ENABLE_BIG_STONES=False,
        SCENE_NUM_SMALL_STONES=0, SCENE_ENABLE_TREES=False,
        SCENE_TERRAIN_PRESET="flat",
    ),
    # 【新增·避障对照实验专用】8 块大石头直接压在 waypoint 连线上。
    #   论文原始布局（paper_full / nav_big_only / stones_only）里 15 块石头只有
    #   1 块真正挡路，其余都在路径 1.34m 以外 —— 那个场景下【不避障也能全过】，
    #   测不出上层规划的价值。这个布局下不绕行就一定撞，才能做出有意义的对照。
    "blocker": dict(
        SCENE_ENABLE_STONES=True, SCENE_ENABLE_BIG_STONES=True,
        SCENE_NUM_SMALL_STONES=0, SCENE_ENABLE_TREES=False,
        SCENE_TERRAIN_PRESET="custom", SCENE_STONE_LAYOUT="blocker",
    ),
    # 【难度梯度】单石挡路：只有 1 块石头压在 waypoint 第一段上。
    "blocker_single": dict(
        SCENE_ENABLE_STONES=True, SCENE_ENABLE_BIG_STONES=True,
        SCENE_NUM_SMALL_STONES=0, SCENE_ENABLE_TREES=False,
        SCENE_TERRAIN_PRESET="custom", SCENE_STONE_LAYOUT="blocker_single",
    ),
    # 【难度梯度】双石挡路：第一段上 2 块石头，形成窄缝。
    "blocker_two": dict(
        SCENE_ENABLE_STONES=True, SCENE_ENABLE_BIG_STONES=True,
        SCENE_NUM_SMALL_STONES=0, SCENE_ENABLE_TREES=False,
        SCENE_TERRAIN_PRESET="custom", SCENE_STONE_LAYOUT="blocker_two",
    ),
    # 可复现双石窄门：从 (6,6) 直行到 (30,6)，用于单独验证对准与侧擦。
    "narrow_gate": dict(
        SCENE_ENABLE_STONES=True, SCENE_ENABLE_BIG_STONES=True,
        SCENE_NUM_SMALL_STONES=0, SCENE_ENABLE_TREES=False,
        SCENE_TERRAIN_PRESET="flat", SCENE_STONE_LAYOUT="narrow_gate",
        SCENE_RANDOMIZE_OBSTACLES_EACH_TRIAL=False,
        NAV_RANDOM_STRAIGHT_ROUTE=True,
        SPAWN_RANDOMIZE_EACH_TRIAL=True,
        SPAWN_RANDOM_X_RANGE=(6.0, 6.0),
        SPAWN_RANDOM_Y_RANGE=(6.0, 6.0),
    ),
    # 同上但保留树（更接近野外，代价是构建变慢、显存变高）
    "blocker_trees": dict(
        SCENE_ENABLE_STONES=True, SCENE_ENABLE_BIG_STONES=True,
        SCENE_NUM_SMALL_STONES=0, SCENE_ENABLE_TREES=True,
        SCENE_TERRAIN_PRESET="custom", SCENE_STONE_LAYOUT="blocker",
    ),
    # 随机直线走廊：起点 (6, y), y∈[3,9]；终点 (30, y)；中段随机大石头+树。
    # 障碍区域为世界坐标 x∈[12,24], y∈[0,12]，地形只用离散高度，不含斜坡。
    "random_corridor": dict(
        SCENE_ENABLE_STONES=True, SCENE_ENABLE_BIG_STONES=True,
        SCENE_NUM_SMALL_STONES=0, SCENE_ENABLE_TREES=True,
        SCENE_TERRAIN_PRESET="height", SCENE_STONE_LAYOUT="random_corridor",
        SCENE_RANDOMIZE_OBSTACLES_EACH_TRIAL=True,
        NAV_RANDOM_STRAIGHT_ROUTE=True,
        SPAWN_RANDOMIZE_EACH_TRIAL=True,
    ),
    # ⚡ 调试/回归专用：关掉大树 + 大小石头，但【保留 custom 地形】。
    #   与 "empty" 的区别：empty 会把地形换成平地，导航路线就没意义了；
    #   这个只砍掉障碍物，地形、出生点、waypoints 全部不变，
    #   所以能用来快速验证代码改动是否跑通，又不影响导航逻辑本身。
    #   实测收益：省掉 4 棵树 186 万面的碰撞 mesh cooking（构建 ~40s 里的 28.6s）
    #   和 985 个碎石 actor，构建与每步耗时都大幅下降。
    "no_obstacles": dict(
        SCENE_STONE_LAYOUT=None,
        SCENE_ENABLE_STONES=False, SCENE_ENABLE_BIG_STONES=False,
        SCENE_NUM_SMALL_STONES=0, SCENE_ENABLE_TREES=False,
        SCENE_TERRAIN_PRESET="custom",
    ),
}
#: 只改这一行即可切换场景；设为 None 表示不套用预设，直接用上面的单项开关
ACTIVE_SCENE_PRESET = "random_corridor"

#: ⚠ 场景预设只能在【进程启动、go2_config 尚未 import】时生效，见
#:   _warn_scene_preset_too_late()。为方便做 A/B 对照实验（同一份代码跑不同场景，
#:   不必反复改文件），这里允许用环境变量在 import 阶段覆盖上面这行：
#:       HIMLOCO_SCENE_PRESET=blocker python play1.py --task=go2
#:   因为读取发生在模块加载时（早于 go2_config 固化类属性），所以是真正生效的。
#:   留空/不设 = 完全按文件里的 ACTIVE_SCENE_PRESET 走。
_SCENE_PRESET_ENV = os.environ.get("HIMLOCO_SCENE_PRESET", "").strip()
if _SCENE_PRESET_ENV:
    ACTIVE_SCENE_PRESET = _SCENE_PRESET_ENV

#: viz_config 自身是否已加载完毕（用于抑制模块初始化期间的误报告警）
_VIZ_MODULE_READY = False


def _warn_scene_preset_too_late(name):
    """若 go2_config 已被 import，提醒运行时切换场景预设不会生效。

    实测陷阱（2026-09-16）：go2_config.py 里 `num_stones = VIZ.SCENE_NUM_STONES`
    这类写法是在【类定义求值时】就把值固化成类属性了，而 go2_config 由
    envs/__init__.py 在 import 时加载。因此一旦 `from legged_gym.envs import *`
    跑过，再调 apply_scene_preset() 只改了 viz_config 自己的全局，
    go2_config / legged_robot 拿到的仍是旧值 —— 石头/树根本没被创建，
    实验会在“空场景”里静默跑完，结论全错。
    正确做法：改本文件的 ACTIVE_SCENE_PRESET，然后重新启动进程。
    """
    import sys
    # viz_config 自身在模块加载时会调一次 apply_scene_preset(ACTIVE_SCENE_PRESET)。
    # 那时 go2_config 往往正处于“已开始 import 但类还没定义完”的中间态，
    # 直接按 sys.modules 判断会误报（且读不到 Go2RoughCfg，打出 num_stones=-1）。
    if not _VIZ_MODULE_READY:
        return False
    mod = sys.modules.get("legged_gym.envs.go2.go2_config")
    cfg_cls = getattr(mod, "Go2RoughCfg", None) if mod is not None else None
    stone_cfg = getattr(cfg_cls, "stone", None)
    if stone_cfg is None:
        return False        # go2_config 还没定义完，不算“太晚”
    try:
        cur = int(getattr(stone_cfg, "num_stones", -1))
    except Exception:
        cur = -1
    print(
        "\n[viz_config] ⚠⚠ apply_scene_preset(%r) 在 go2_config 已 import 之后调用，"
        "【不会生效】！\n"
        "   go2_config 里 num_stones 等是类定义求值时就固化的类属性，\n"
        "   现在它仍是 num_stones=%d，而本模块已解析为 num_stones=%d。\n"
        "   结果：场景（石头/树/地形）实际按【旧配置】创建，实验可能在空场景里\n"
        "   静默跑完、结论全错。\n"
        "   ✅ 正确做法：改本文件顶部的 ACTIVE_SCENE_PRESET = %r，然后重启进程。\n"
        % (name, cur, SCENE_NUM_STONES, name),
        file=sys.stderr)
    return True


def apply_scene_preset(name):
    """把场景预设写进本模块全局命名空间（用法同 apply_preset）。

    ⚠ 只能在【进程启动、go2_config 尚未 import】时调用才有效，
      见 _warn_scene_preset_too_late()。正常运行请改 ACTIVE_SCENE_PRESET。
    """
    if name is None:
        return None
    if name not in SCENE_PRESETS:
        raise KeyError("未知场景预设: %r，可选: %s" % (name, sorted(SCENE_PRESETS.keys())))
    for key, value in SCENE_PRESETS[name].items():
        globals()[key] = value
    globals()["ACTIVE_SCENE_PRESET"] = name
    # 预设里可能改了地形/石头布局，必须重算这几个派生值。
    # SCENE_BIG_STONE_POSITIONS 是【当前生效】的坐标，go2_config / legged_robot
    # 都直接读它，所以布局切换只需覆盖这一个名字，下游代码无需改动。
    globals()["SCENE_BIG_STONE_POSITIONS"] = resolve_stone_layout()
    globals()["SCENE_TERRAIN_SEQUENCE"] = resolve_terrain_sequence()
    (globals()["SCENE_NUM_BIG_STONES"], globals()["SCENE_NUM_SMALL_STONES_TOTAL"],
     globals()["SCENE_NUM_STONES"],
     globals()["SCENE_BIG_STONE_INDICES"]) = resolve_stone_counts()
    # 放在派生值算完之后，告警里才能打印出“本模块已解析为”的真实数字
    _warn_scene_preset_too_late(name)
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


def resolve_stone_layout(layout=None):
    """按 SCENE_STONE_LAYOUT 选出实际使用的大石头坐标列表。

    这样切换“论文原始布局 / 挡路布局”不必手改 SCENE_BIG_STONE_POSITIONS，
    场景预设也能直接把 layout 写进去（见 SCENE_PRESETS 的 blocker_* 项）。
    """
    layout = SCENE_STONE_LAYOUT if layout is None else layout
    if layout is None or layout == "paper":
        return list(SCENE_PAPER_STONE_POSITIONS)
    if layout == "blocker":
        return list(SCENE_BLOCKER_STONE_POSITIONS)
    if layout == "blocker_single":
        return list(SCENE_BLOCKER_SINGLE_POSITIONS)
    if layout == "blocker_two":
        return list(SCENE_BLOCKER_TWO_POSITIONS)
    if layout == "narrow_gate":
        return list(SCENE_NARROW_GATE_POSITIONS)
    if layout == "random_corridor":
        # 这里只提供 actor 数量；legged_robot 在 PhysX actor 创建前按 seed 重新采样。
        # GPU pipeline 启动后不能移动静态碰撞体，因此同一进程内布局保持不变。
        count = int(SCENE_RANDOM_CORRIDOR_BIG_STONE_COUNT)
        return [(12.5 + (i % 4) * 2.5, 1.5 + (i // 4) * 7.0)
                for i in range(count)]
    if isinstance(layout, (list, tuple)):
        return list(layout)
    raise KeyError("未知石头布局: %r（可选 None/'paper'/'blocker' 或坐标列表）" % (layout,))


def resolve_stone_counts(enable=None, big=None, small=None, positions=None):
    """返回 (num_big_stones, num_small_stones, num_stones, big_stone_indices)。

    大小石头数量已解耦：底层按“先大后小”的顺序创建 actor，
    大石头恒占 actor 索引 1..num_big（见 legged_robot.py 的创建循环），
    big_stone_indices 仅为兼容旧代码而保留。
    """
    enable = SCENE_ENABLE_STONES if enable is None else enable
    big = SCENE_ENABLE_BIG_STONES if big is None else big
    small = SCENE_NUM_SMALL_STONES if small is None else small
    positions = resolve_stone_layout() if positions is None else positions
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
_VIZ_MODULE_READY = True

#: go2_config.py 直接取用的最终值（已解析，勿在此之后再改开关）
SCENE_BIG_STONE_POSITIONS = resolve_stone_layout()
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
    if PLANNER_MODE not in ("fmm", "mppi", "rrt", "astar_dwa"):
        problems.append("PLANNER_MODE=%r 不受支持" % PLANNER_MODE)
    if MPPI_HORIZON < 2 or MPPI_NUM_SAMPLES < 2:
        problems.append("MPPI_HORIZON 和 MPPI_NUM_SAMPLES 必须都 >= 2")
    if MPPI_FOOTPRINT_LENGTH_M <= 0.0 or MPPI_FOOTPRINT_WIDTH_M <= 0.0:
        problems.append("MPPI 机身足迹长宽必须 > 0")
    if TEMPORAL_MEMORY_MAX_AGE_STEPS < 1:
        problems.append("TEMPORAL_MEMORY_MAX_AGE_STEPS 必须 >= 1")
    if abs(CAM_HORIZONTAL_FOV - BEV_HFOV_DEG) > 1e-6:
        problems.append(
            "CAM_HORIZONTAL_FOV(%s) 必须等于 BEV_HFOV_DEG(%s)，否则深度图反投影到 BEV 会错位"
            % (CAM_HORIZONTAL_FOV, BEV_HFOV_DEG))
    # ⚠ 原校验写的是 “CAM_FOLLOW_HEIGHT 必须等于 BEV_CAM_HEIGHT”，这是【语义错误】：
    #   CAM_FOLLOW_HEIGHT 是相机相对 base 的偏移，BEV_CAM_HEIGHT 是相机【离地】高度，
    #   两者根本不是一个量。正确的关系是
    #       BEV_CAM_HEIGHT ≈ base 离地高度 + CAM_FOLLOW_HEIGHT
    #   （因为 pathplanner 里 cam_pos.z = base_pos[2] + CAM_FOLLOW_HEIGHT，
    #     而 base_pos[2] 本身已含 base 离地高度。）
    #   实测 Go2 站立 base 离地 0.369±0.003m，CAM_FOLLOW_HEIGHT=0.25，
    #   故相机真实离地 0.619m，而旧配置写 0.25m -> 系统性偏差 +0.369m。
    #   这个偏差正是“像素级地面判据在无障碍地形上误报 48%”的原因：
    #   判据用 cam_height/sin(俯角) 反推地面深度，cam_height 偏小 59% 会严重失真。
    cam_h_expect = ROBOT_BASE_HEIGHT_M + CAM_FOLLOW_HEIGHT
    # ROBOT_BASE_HEIGHT_M 见下方定义（policy 驱动实测 0.278，地形真值基准）
    if abs(cam_h_expect - BEV_CAM_HEIGHT) > BEV_CAM_HEIGHT_TOL:
        problems.append(
            "BEV_CAM_HEIGHT(%s) 与“base离地(%s)+相机偏移(%s)=%s”不符，超出容差 %s；"
            "反投影会把地面算错高度、像素级判据大面积误报"
            % (BEV_CAM_HEIGHT, ROBOT_BASE_HEIGHT_M, CAM_FOLLOW_HEIGHT,
               cam_h_expect, BEV_CAM_HEIGHT_TOL))
    if CAM_ENABLE is False and (BEV_X or BEV_Y):
        problems.append("CAM_ENABLE=False 时 BEV 建图拿不到深度图，导航会退化")
    # CAM_READ_ENABLE 已由 resolve_cam_read() 在 apply_preset 里解析成 True/False；
    # 消费方判定复用 cam_read_consumers()，避免两处逻辑分叉。
    if CAM_READ_ENABLE is False:
        consumers = cam_read_consumers()
        if consumers:
            problems.append(
                "CAM_READ_ENABLE=False 但这些开关需要深度图: %s；"
                "请设 CAM_READ_ENABLE=None(自动) 或 True，否则拿到的是空图、避障静默失效"
                % ", ".join(consumers))
        else:
            notes.append(
                "CAM_READ_ENABLE=False（已跳过深度读取）：当前无任何消费方需要深度图，"
                "属预期省资源配置，导航与指标不受影响")
    # RGB 与 depth 分开判断：读 depth 却强制读 RGB 只是浪费，反之则会拿到空图
    if CAM_READ_ENABLE is False and CAM_READ_RGB is True:
        problems.append(
            "CAM_READ_RGB=True 但 CAM_READ_ENABLE=False：整段相机读取已被跳过，"
            "RGB 也拿不到；请设 CAM_READ_ENABLE=None(自动) 或 True")
    if CAM_READ_ENABLE is True and CAM_READ_RGB is False:
        rgb_need = cam_rgb_consumers()
        if rgb_need:
            problems.append(
                "CAM_READ_RGB=False 但这些开关需要 RGB 图: %s；"
                "会存出/显示空图。请设 CAM_READ_RGB=None(自动) 或 True"
                % ", ".join(rgb_need))
        else:
            notes.append(
                "CAM_READ_RGB=False（只读 depth，不拷 RGB）：当前无任何消费方需要 RGB，"
                "属预期省资源配置 —— 每步少拷一张 float32，泄漏量减半")
    if SPEED_MOD_ENABLE:
        if not (0.0 < SPEED_MOD_MIN_VX <= NAV_FORWARD_VX):
            problems.append(
                "SPEED_MOD_MIN_VX(%s) 必须落在 (0, NAV_FORWARD_VX=%s] 内："
                "为 0 会原地卡死，大于上限则减速无意义"
                % (SPEED_MOD_MIN_VX, NAV_FORWARD_VX))
        if SPEED_MOD_STOP_DIST >= SPEED_MOD_SLOW_DIST:
            problems.append(
                "SPEED_MOD_STOP_DIST(%s) 必须小于 SPEED_MOD_SLOW_DIST(%s)，"
                "否则减速区间为空/为负" % (SPEED_MOD_STOP_DIST, SPEED_MOD_SLOW_DIST))
        if SPEED_MOD_USE_PIXEL_JUDGE:
            notes.append(
                "SPEED_MOD_USE_PIXEL_JUDGE=True：该判据的地面模型用了 sin(θ)（斜距），"
                "而 IMAGE_DEPTH 是轴向深度、应为 tan(θ)，实测在无障碍地形上误报 99%。"
                "启用前请先修正模型并用 probe_pix_falsepos.py 复测误报率")
        if not (0.0 < SPEED_MOD_GROUND_RATIO <= 1.0):
            problems.append(
                "SPEED_MOD_GROUND_RATIO(%s) 必须落在 (0,1]：它是“实测深度 < 地面深度的该倍数”"
                "的比较系数" % SPEED_MOD_GROUND_RATIO)
        if SPEED_MOD_MAX_RANGE > BEV_X:
            notes.append(
                "SPEED_MOD_MAX_RANGE(%sm) 超过 BEV 前向范围 BEV_X(%sm)，"
                "超出部分永远扫不到" % (SPEED_MOD_MAX_RANGE, BEV_X))
        if SPEED_MOD_ENABLE and not BEV_ENABLE_UPDATE:
            problems.append(
                "SPEED_MOD_ENABLE=True 但 BEV_ENABLE_UPDATE=False：障碍图恒为空，"
                "前向净空永远等于最大扫描距离，速度调制等于没生效")
    if FMM_TURN_HYSTERESIS_ENABLE:
        if not (0.0 <= FMM_TURN_HYSTERESIS_THRESH < 1.0):
            problems.append(
                "FMM_TURN_HYSTERESIS_THRESH(%s) 必须落在 [0,1)：dir_c 是归一化方向分量"
                % FMM_TURN_HYSTERESIS_THRESH)
        if int(FMM_TURN_HOLD_STEPS) < 0:
            problems.append("FMM_TURN_HOLD_STEPS 必须 >= 0（当前 %s）" % FMM_TURN_HOLD_STEPS)
    if FMM_OPEN_SPACE_YAW_DEADBAND_RAD < 0.0:
        problems.append(
            "FMM_OPEN_SPACE_YAW_DEADBAND_RAD 必须 >= 0（当前 %s）"
            % FMM_OPEN_SPACE_YAW_DEADBAND_RAD)
    if FMM_USE_SOFT_COST:
        if FMM_HARD_RADIUS_M < 0.0:
            problems.append("FMM_HARD_RADIUS_M 必须 >= 0（当前 %s）" % FMM_HARD_RADIUS_M)
        if FMM_SOFT_CLEARANCE_M < FMM_HARD_RADIUS_M:
            problems.append(
                "FMM_SOFT_CLEARANCE_M(%s) 必须 >= FMM_HARD_RADIUS_M(%s)"
                % (FMM_SOFT_CLEARANCE_M, FMM_HARD_RADIUS_M))
        if FMM_SOFT_COST_WEIGHT < 0.0:
            problems.append("FMM_SOFT_COST_WEIGHT 必须 >= 0（当前 %s）" % FMM_SOFT_COST_WEIGHT)
        if FMM_SOFT_COST_POWER <= 0.0:
            problems.append("FMM_SOFT_COST_POWER 必须 > 0（当前 %s）" % FMM_SOFT_COST_POWER)
    if PERCEPT_UPDATE_INTERVAL is not None:
        try:
            interval = int(PERCEPT_UPDATE_INTERVAL)
        except (TypeError, ValueError):
            problems.append("PERCEPT_UPDATE_INTERVAL 必须是整数（当前 %r）"
                            % (PERCEPT_UPDATE_INTERVAL,))
        else:
            if interval < 1:
                problems.append("PERCEPT_UPDATE_INTERVAL 必须 >= 1（当前 %d）" % interval)
            elif interval > 10:
                notes.append(
                    "PERCEPT_UPDATE_INTERVAL=%d 偏大：机器狗 1.0m/s 下每次感知间隔已移动 "
                    "%.2fm（≈%.0f 个 BEV 栅格），近距障碍可能来不及反应，建议 <= 10"
                    % (interval, interval * 0.02 * NAV_FORWARD_VX,
                       interval * 0.02 * NAV_FORWARD_VX / BEV_RES))
    if LIVE_VIEWER_SYNC is False and (LIVE_DRAW_VIEWER_OVERLAY or LIVE_FOLLOW_VIEWER_CAMERA):
        notes.append(
            "LIVE_VIEWER_SYNC=False 时 viewer 窗口不逐帧刷新，"
            "LIVE_DRAW_*/跟随相机 看不到效果（想肉眼看请设 LIVE_VIEWER_SYNC=True）")
    if LIVE_DRAW_INTERVAL is not None and int(LIVE_DRAW_INTERVAL) < 1:
        problems.append("LIVE_DRAW_INTERVAL 必须 >= 1（当前 %s）" % LIVE_DRAW_INTERVAL)
    if BEV_ENABLE_STEP_JUDGE:
        if BEV_STEP_THRESH is not None and float(BEV_STEP_THRESH) <= 0:
            problems.append(
                "BEV_STEP_THRESH(%s) 必须 > 0（或设 None 复用 BEV_HEIGHT_RANGE_THRESH）"
                % BEV_STEP_THRESH)
        if int(BEV_STEP_SPAN) < 1:
            problems.append("BEV_STEP_SPAN 必须 >= 1（当前 %s）" % BEV_STEP_SPAN)
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
    if NAV_GOAL_REACH_THRESH_M <= 0.0:
        problems.append("NAV_GOAL_REACH_THRESH_M 必须 > 0")
    if NAV_GOAL_SLOW_DIST_M <= NAV_GOAL_REACH_THRESH_M:
        problems.append(
            "NAV_GOAL_SLOW_DIST_M(%s) 必须大于到达阈值(%s)"
            % (NAV_GOAL_SLOW_DIST_M, NAV_GOAL_REACH_THRESH_M))
    if NAV_RANDOM_STRAIGHT_ROUTE:
        if not SPAWN_RANDOMIZE_EACH_TRIAL:
            problems.append("随机直线路线已开启，但 SPAWN_RANDOMIZE_EACH_TRIAL=False")
        if NAV_STRAIGHT_GOAL_X <= SPAWN_RANDOM_X_RANGE[1]:
            problems.append("直线目标 x 必须位于出生区域前方")
    if SCENE_RANDOMIZE_OBSTACLES_EACH_TRIAL:
        if SCENE_RANDOM_OBSTACLE_X[0] >= SCENE_RANDOM_OBSTACLE_X[1]:
            problems.append("随机障碍 x 范围必须递增")
        if SCENE_RANDOM_OBSTACLE_Y[0] >= SCENE_RANDOM_OBSTACLE_Y[1]:
            problems.append("随机障碍 y 范围必须递增")
        if SCENE_RANDOM_OBSTACLE_MIN_SEPARATION_M < 0.0:
            problems.append("随机障碍最小间距必须 >= 0")
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
    route = ("直线(出生x%s/y%s -> 目标x%.1f)" % (
                SPAWN_RANDOM_X_RANGE, SPAWN_RANDOM_Y_RANGE, NAV_STRAIGHT_GOAL_X)
             if NAV_RANDOM_STRAIGHT_ROUTE else "固定航点")
    return "scene=%s | %s | 树=%s | 地形=%s | 路线=%s" % (
        ACTIVE_SCENE_PRESET or "custom", stones,
        "开" if SCENE_ENABLE_TREES else "关",
        SCENE_TERRAIN_PRESET, route)


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
SPAWN_INIT_VELOCITY_RANGE = 0.0


# =====================================================================
#  十二、终止判定（摔倒 / 超时 / 掉落 / 卡住）
# ---------------------------------------------------------------------
#  对应 legged_robot.py 的 check_termination()。
#  这些阈值原本是写死在 legged_robot.py 里的魔法数（-0.5 / 0.3 / 0.05 / 1.0），
#  现在集中到这里，go2_config.termination 继承 TERMINATION_CFG。
#  ⚠ 这一节会影响训练与评测结果，调参前请先想清楚，别当成纯显示开关。
# =====================================================================
TERM_ENABLE_COLLISION = True       #: 终止部位碰撞即判摔倒
#: 接触力阈值 [N]。监视刚体是 base（机身，见 go2_config.terminate_after_contacts_on）。
#:   ⚠ 1.0N 是 legged_gym 上游默认值，但它意味着【机身蹭到任何东西即判摔倒】。
#:     实测（2026-09-17 消融，stones_only 12 trial）：
#:        阈值 1.0N -> 0.25     阈值 50N -> 0.42
#:     且日志里出现过"接触力2.4N>1.0N"就终止 —— 2.4N 对 ~15kg 的 Go2 而言
#:     只是轻微蹭碰（真摔倒通常是数百 N 量级），判为摔倒偏严。
#:   保持 1.0 的理由：与上游/多数文献口径一致，成功率数字更保守可信。
#:   若要改用 50，属于【改变成功判据定义】，会影响论文表格里所有成功率，
#:   需要与导师确认后再动，不要为了刷指标悄悄放宽。
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
#: ⚠ 分辨率与内存泄漏强相关：每步要把 depth+rgb 两张 float32 图拷回 CPU，
#:   单步内存 = W*H*8 字节。1280x720 时高达 7.37MB/步，1000 步就是 7.4GB，
#:   这正是“跑一会儿整机卡死”的根因（详见文件头实测记录）。
#:   实测 BEV 障碍检出能力（同一场景、真实仿真深度图、15 个大石头）：
#:       1280x720 -> 障碍栅格 71~72，7.37MB/步   (100%)
#:        640x360 -> 障碍栅格 66~68，1.84MB/步   ( 25%)
#:        320x180 -> 障碍栅格 60~61，0.46MB/步   ( 6.2%)
#:   BEV 栅格只有 120x120（= BEV_X / BEV_RES），320x180 已远超其需求：
#:   障碍检出仅降 15%，内存降 94% —— 性价比最高，故取 320x180。
#:   若要出论文用的高清 RGB/Depth 图，临时改回 1280x720 并调小 NUM_TRIALS，
#:   且不要开远程桌面长时间跑。
CAM_WIDTH = 320                 #: 图像宽 [px]
CAM_HEIGHT = 180                #: 图像高 [px]
#: 水平视场角 [deg]。⚠ 必须等于 BEV_HFOV_DEG，否则深度图反投影到 BEV 会错位。
CAM_HORIZONTAL_FOV = 90.0
#: 初始机位（仅第一帧生效，之后每步按 base 朝向重写）
CAM_INIT_FORWARD = 0.3
CAM_INIT_LOOKAT_FORWARD = 1.3
CAM_INIT_HEIGHT = 0.6

# ---------- 感知模块每步的相机跟随机位（相对 base）----------
#: ⚠ 这是相机【离地】高度，必须等于 ROBOT_BASE_HEIGHT_M + CAM_FOLLOW_HEIGHT。
#:   原值 0.25 是直接抄了 CAM_FOLLOW_HEIGHT，但那是“相对 base 的偏移”而非“离地高度”，
#:   导致反投影系统性偏低 0.369m（见 check_consistency 的推导）。
#:   影响范围：
#:     · BEV 同格高度差判据 max_z-min_z：常数偏移会抵消，所以【基本不受影响】
#:       （这与之前“斜坡误判率 0%”的实测结论一致，不必怀疑那条结论）；
#:     · 反投影出的绝对 z、以及像素级地面判据 cam_height/sin(俯角)：【直接算错】。
#: Go2 站立时 base 中心离地高度 [m]。
#: ⚠ 实测值（2026-09-16，用足端最低点当地面基准，站稳后 60 帧）：0.369 ± 0.003 m。
#:   注意 base_pos[2] 打印出来是 ~1.25m，那是【绝对世界 z】（terrain 中央平台抬高了出生点），
#:   不能直接当离地高度用 —— 这是之前几何推导跑偏的原因。
ROBOT_BASE_HEIGHT_M = 0.278
#: ⚠ 2026-09-16 复测修正（policy 驱动，150 帧）：base 离地 = 0.254±0.007(足端最低点基准)
#:   / 0.278±0.007(terrain heightfield 真值基准)。旧值 0.369 无法复现（当时用了零关节
#:   动作或别的基准）。反投影残差拟合独立确认 h_off=0.25、相机离地≈0.528。
#: BEV_CAM_HEIGHT 与推导值的允许偏差 [m]。base 离地会随步态小幅波动（±0.003），
#: 且不同 checkpoint/地形略有差异，故留 0.02m 容差；超差才报错。
BEV_CAM_HEIGHT_TOL = 0.02

CAM_FOLLOW_FORWARD = 0.3        #: 相机在 base 前方多远 [m]
#: 相机相对 base 的高度偏移 [m]。⚠ 这不是离地高度，别和 BEV_CAM_HEIGHT 混为一谈：
#:   pathplanner 里 cam_pos.z = base_pos[2] + 本值，而 base_pos[2] 已含 base 离地高度，
#:   所以相机【真实离地】= ROBOT_BASE_HEIGHT_M + 本值。
CAM_FOLLOW_HEIGHT = 0.25
CAM_FOLLOW_LOOKAT_FORWARD = 1.0 #: 注视点在 base 前方多远 [m]

# ---------- BEV 高度图建图（深度图 -> 障碍栅格）----------
BEV_X = 6.0                     #: 前方感知范围 [m]
BEV_Y = 6.0                     #: 左右感知范围 [m]
BEV_RES = 0.05                  #: 栅格分辨率 [m/cell]
BEV_CAM_HEIGHT = 0.528          #: 深度相机【离地】高度 [m] = base离地0.278(地形基准) + 偏移0.25
#: ⚠ 2026-09-16 复测修正：旧值 0.619 基于“base 离地 0.369”，但 policy 驱动下
#:   实测 base 离地 = 0.254(足端基准)/0.278(terrain 真值基准)，0.369 无法复现。
#:   反投影残差拟合（反投影世界z - terrain真值）也确认 h_off=0.25 正确、h≈0.528。
#:   h 是地面模型的分母，错 0.09m(17%) 会直接造成近距误报/漏检。
BEV_HFOV_DEG = 90.0             #: 相机水平视场角 [deg]（需与 legged_robot 相机 fov 一致）
BEV_MAX_DEPTH = 6.0             #: 最大有效深度 [m]，超过视为无效
BEV_MIN_DEPTH = 0.05            #: 最小有效深度 [m]，过近的点丢弃
#: 深度图纵向采样起始比例：BEV 建图只用图像【下 (1-该值) 部分】，跳过上半张天空。
#:
#: ⚠ 这个值【不要随便放宽】（2026-09-16 实测回归）：
#:   几何上相机是平视的（CAM_FOLLOW_LOOKAT_FORWARD 与相机同高），地平线在 v_ratio=0.50，
#:   所以 0.55 = 只看【低于相机高度 0.25m 的地面】。大石头高出地面 0.76m，
#:   其顶部仰角 9.6°(3m)~26.9°(1m)，落在 v_ratio 0.04~0.34 —— 会被这个 ROI 裁掉。
#:   把 ROI 放宽到 0.10 确实能让 BEV 看见石头（近距 occ 从 0 提升到 65+），
#:   但代价是把地平线以上的地形起伏/远景也纳入，障碍图被污染：
#:     · blocker 场景（8 块挡路石头）: v=0.55 与 v=0.10 都因近距盲区失败
#:     · 论文场景 no_obstacles（纯地形）: v=0.55 -> 0.90，v=0.10 -> 0.70  ⬇
#:   即“放宽 ROI”在 BEV 栅格判据上是【净负收益】，所以维持 0.55。
#:
#: ✅ 近距直立障碍的盲区改由【像素级地面比较判据】单独解决，见 SPEED_MOD_ROI_START
#:    与 BEVMapper.forward_obstacle_distance()：两个判据目标不同
#:    （BEV 栅格图 -> 喂 FMM 做转向；像素级净空 -> 决定要不要刹车），
#:    因此各用各的最优 ROI，互不干扰。
BEV_V_START_RATIO = 0.55        #: BEV 建图纵向采样起始比例（见上方回归结论，勿放宽）

# ---------- 相机几何标定结论（2026-09-16 实测，全部带开关可回退）----------
#: BEV 横向符号。实测（单石头定向标定：石头放机器人左侧 1.0m，命中图像左半 u∈[0,120]）
#:   相机成像【不镜像】：du>0 = 图像右 = 机器人右侧。而 FMM / 路径回投影整条链
#:   约定“列号增大 = 左”，故反投影必须取 y_left = -du*depth。
#:   原实现 y = +du*depth 使障碍图【左右镜像】，FMM 会朝障碍一侧转向（真 bug）。
#:   -1.0 = 修正（默认）；+1.0 = 回退原行为。
BEV_LATERAL_SIGN = -1.0
#: 是否按【轴向深度】做严格针孔反投影。
#:   实测 IsaacGym IMAGE_DEPTH 是轴向深度（沿光轴）：地面模型 h/tan(θ) 误差 +0.6%，
#:   而 h/sin(θ)（欧氏）误差 -1~-9.5%。True = 严格针孔（默认）；False = 原近似。
BEV_USE_AXIAL_GEOM = True
#: 深度被上游 clip 到 BEV_MAX_DEPTH 的像素（天空/超远地面恰好等于 6.0）视为无效。
#:   原 `d > max_depth` 判不掉“恰好等于 max_depth”的像素。平视时它们反投影 x=6.0
#:   刚好出界、危害不大；但【有俯角时】x = 6*(cos p - dv*sin p) 会落进 BEV 范围内，
#:   且 z ≈ -3.6m，单格高度差爆表 -> 假障碍。故俯角功能必须配合本开关。
#:   （放宽 ROI 后 occ 暴涨的主因是地平线以上的远景山丘，仍由 BEV_V_START_RATIO 控制。）
BEV_CLIP_INVALID_EPS = 1e-3
#: 光轴俯角 [deg]，向下为正。平视时近距直立障碍落在 ROI 上沿之外（1.1m 内必漏检，
#:   几何下界）；下压光轴把盲区下界推近。0 = 平视（原行为）。
#:   ⚠ 改此值后 BEV_V_START_RATIO 的几何含义随之变化（ROI 相对光轴），需重新回归。
BEV_CAM_PITCH_DEG = 0.0
#: 近距掩蔽 [m]：BEV 建图丢弃前向距离小于该值的像素。俯角机位下最近可见地面
#:   可近至 0.35m，混有掠射角噪声/机身自遮挡，实测把 clearance 打成间歇尖峰、
#:   速度调制反复砍速导致顶死。0 = 不掩蔽（平视时最近可见地面 0.9m，无影响）。
BEV_MIN_RANGE_M = 0.0
#: 障碍判据一（原实现）：单栅格内 max_z-min_z 超过它即判为障碍 [m]
BEV_HEIGHT_RANGE_THRESH = 0.2
#: 障碍判据二（跨格台阶，⚠ 修复“正对垂直墙面漏检”的关键）
#:   判据一【只看同一格内部】的高度差。正对的墙面/大石头上，所有像素深度几乎相同，
#:   反投影后挤在 ix 固定的少数几格里，格内高度差被摊薄到 ≈0，于是墙检不出来。
#:   实测（8 块 0.76m 高、1.0m 宽的挡路大石头）：
#:       距石 2.04m -> occ=140 格；1.10m -> occ=0（失明）
#:       前向净空随之从 0.55m 跳回最大值 3.00m，规划器判定“前方无障”，
#:       wz≈0，机器狗刹住后直直顶着石头磨到 timeout，成功率 0/10。
#:   判据二把本格 max_z 与【更近 BEV_STEP_SPAN 格】的 min_z 相比：
#:       墙面：max_z≈0.76 vs 墙前地面 min_z≈-0.08 -> 台阶 0.84 > 0.2，检出 ✓
#:       坡地：每格抬升仅 bev_res×坡度（10% 坡 = 0.005m/格，30% = 0.015m/格），
#:             远小于 0.2m，不误判 ✓
#:   ⚠ 这比“放宽 BEV_V_START_RATIO”稳得多：放宽 ROI 会把地形起伏整片判成障碍，
#:     实测论文场景成功率 0.90 -> 0.70（见 BEV_V_START_RATIO 的回归记录）。
#: ⛔ 默认 False：这个判据【离线单测未通过，增益为 0】（2026-09-16）。
#:   设想是“把本格 max_z 与更近一格的 min_z 相比，墙面会呈现 0.84m 的台阶”。
#:   但真实约束是：BEV_V_START_RATIO=0.55 只保留【低于相机高度】的像素，
#:   而相机真实离地 0.619m、大石头顶离地 0.757m —— 石头高出相机的部分
#:   根本不在这个 ROI 里，反投影出的 max_z 最多只有 0.13m（实测 hr 最大 0.125~0.194），
#:   远低于 0.2m 阈值，所以两个判据都拿不到有效数据。
#:   离线单测（/tmp/unit_step_judge.py，合成墙面/坡地）：
#:       wall@2.0 原判据=0 格，新判据=0 格，增益 +0
#:       block@1.0 原判据=0 格，新判据=0 格，增益 +0
#:       slope30 两者都 0 格（不误判，但也说明它没在工作）
#:   端到端实测（blocker 场景 8 块挡路石头，10 trial）：成功率仍 0.00。
#:   => 结论：近距直立障碍的漏检【不是判据不够，而是 ROI 根本没覆盖到】。
#:      真正的修法是把相机俯角向下压（让石头整体进画面）或抬高相机，
#:      而不是在地面 ROI 里换判据。见 CAM_PITCH_DEG 的实验记录。
BEV_ENABLE_STEP_JUDGE = False   #: 是否叠加跨格台阶判据（关掉 = 原实现行为）
BEV_STEP_THRESH = None          #: 台阶阈值 [m]，None = 复用 BEV_HEIGHT_RANGE_THRESH
BEV_STEP_SPAN = 1               #: 与前方第几格比较（1 = 相邻格，对坡度最不敏感）
BEV_MIN_POINTS = 8              #: 单栅格至少多少像素才可信（抗噪）

# ---------- BEV 更新总开关（⚠ 重要，见下方说明）----------
#: 是否把深度图喂给 BEVMapper（即调用 self.bev.update(depth_pos)）。
#: ⚠⚠ 原代码里这一行是被注释掉的（pathplanner.py 的 _update_camera_and_bev 中
#:    "# self.bev.update(depth_pos)"），且全仓库再无其它调用点，
#:    意味着 BEV 障碍图**恒为全 0**：上层规划器看不到任何障碍，FMM 在空障碍图上
#:    退化成“直线朝 waypoint 走”。
#:    实测对比（1 env，朝大石头走 50 步）：
#:        不调用 update -> 障碍格数 0；调用 update -> 障碍格数 21（深度有效像素 18.9 万）
#: 设 False = 保持原代码行为与既有实验数值（但上层等于没在避障）。
#:
#: ✅ 现已默认 True：这条链路才是论文“上层深度图像规划”的本体。
#:    关掉时 FMM 在空障碍图上退化成直线，机器狗从没“看见”过任何障碍，
#:    所谓“避障效果不好”其实是“没有避障”。
#:    已用真实仿真深度图验证可正常检出障碍（320x180 下 60~61 个障碍栅格，
#:    平地/10%上坡/下坡的误判率均为 0%，因为判据是同格内 max_z-min_z 而非绝对高度）。
#:    注意：打开后必须配合 CAM_READ_ENABLE=True（预设已自动处理），
#:    且相机内存泄漏会回归，务必保持上面的低分辨率。
BEV_ENABLE_UPDATE = True

# ---------- 障碍白名单（只绕 actor 障碍）----------
#: 障碍图是否只保留【actor 障碍】（大石头 / 树 / 后续新增的动态障碍）。
#:   ⚠ 2026-09-17 实测：BEV 判据不区分来源，no_obstacles 场景里仍稳定检出 ~55 格，
#:     世界坐标聚类全在 terrain 边界墙（4m 高，起点距左墙 2.4m，在 BEV ±3m 内），
#:     FMM 常年往右推，观感像“在绕地形离散障碍”。地形凸起 hr≈0.01m 本就不会被检出。
#:   True = 只绕 actor（用户语义）；False = 原行为（墙/地形起伏也进障碍图）。
BEV_OBSTACLE_ACTORS_ONLY = True
#: 每个 actor 的掩膜半径 [m]。大石头 scale=2 实测 footprint 约 1.0~1.2m 宽，
#:   取 0.8 留膨胀余量；太小会漏掉石头边缘、太大会把窄廊堵死。
BEV_ACTOR_MASK_RADIUS_M = 0.8
#: 【刹车净空】用哪张障碍图算 forward_clearance（决定要不要减速）。
#:   True  = 原始几何图（含边界墙 + 地形台阶）；False = 白名单图（同转向用图）。
#:   ⚠ 默认 True，2026-09-17 实测定标：白名单把 no_obstacles 障碍图归零后
#:     clearance 恒为 3.0，速度调制彻底失效，机器狗在 0.1m 地形台阶上以 1.0m/s
#:     全速冲撞打滑卡死（成功率 0.90 -> 0.40，失败全部是"指令 0.9~1.2 /
#:     实际 0.000"）。
#:   语义区分：转向问"要不要绕开它"（地形台阶四足就该踩过去，不绕），
#:             刹车问"脚下这段地面会不会绊倒我"（台阶必须降速通过）。
#:     两者是不同问题，因此各用各的图。
BEV_CLEARANCE_USE_RAW_MAP = True

# ---------- 障碍记忆 / 时间融合（TFAM）----------
#: 给规划器的障碍图用“即时帧”还是“即时 ∪ 历史记忆”。
#: True  = 时序融合（记忆叠加，遮挡后仍记得障碍）
#: False = 只用当前帧（原代码的实际行为，注释里那行 return occ_for_planner 被注释掉了）
#:
#: ⚠⚠ 实测结论（2026-09-16，blocker 场景 8 块挡路石头、10 trial）：
#:   打开后 occ 从正常的 20~100 暴涨到 421~929（BEV 总共才 14400 格），
#:   历史障碍把整张局部图填满、可通行区域被彻底封死，FMM 找不到通路，
#:   结果 10/10 全部 Timeout、成功率 0.00（关闭时至少能检出并减速）。
#:   根因在 bev_height_mapper.shift_with_motion()：它按【世界位移】平移 occ_memory，
#:   但 occ_memory 存的是【机器人坐标系】的栅格，机器人原地转身（yaw 变化）时
#:   位移≈0、不做任何平移，于是同一块石头在每个朝向都被重复“写入”一次，
#:   记忆越叠越满。要真正启用时序融合，得先把记忆图改成世界坐标系对齐
#:   （或按 yaw 做旋转重采样），并重新调 BEV_MEMORY_MAX_AGE。
#:   在修好之前【保持 False】。
BEV_ENABLE_MEMORY_FUSION = False
BEV_MEMORY_MAX_AGE = 30         #: 记忆寿命 [帧]，约 1 秒（dt≈0.03）

# ---------- 世界坐标时序障碍记忆（MPPI方法）----------
#: 与上面的旧机器人坐标栅格记忆不同，本模块保存世界坐标障碍点，并按当前完整
#: SE(2) 位姿重投影；因此原地转向不会把同一障碍重复涂满局部地图。
TEMPORAL_MEMORY_ENABLE = _env_bool("HIMLOCO_TEMPORAL_MEMORY", True)
TEMPORAL_MEMORY_MAX_AGE_STEPS = 50    #: 50 Hz 下保留 1 秒，覆盖转身盲区且限制边缘累积
TEMPORAL_MEMORY_VOXEL_M = 0.10
TEMPORAL_MEMORY_MAX_POINTS = 12000

# ---------- FMM 梯度跟随（本文方法）----------
FMM_MAX_WZ = 1.0                #: 角速度上限 [rad/s]
FMM_YAW_K = 2.0                 #: 期望方向角 -> wz 的比例增益
FMM_LOOKAHEAD_M = 0.6           #: 前视取梯度距离 [m]，太近会导致 dc≈0
FMM_INFLATE_RADIUS_M = 0.25     #: 障碍膨胀半径 [m]，防止擦边撞
FMM_GOAL_MIN_FORWARD_M = 0.10   #: goal 在身后时钳到前方的最小值 [m]

# ---------- 双层代价场（第4章第一阶段）----------
#: False = 原始固定膨胀 FMM 基线；True = 硬碰撞层 + 连续软安全代价层。
#: 硬层决定“绝对不能走”，软层只让贴近障碍的路径变贵，因此不会像扩大固定膨胀
#: 那样直接封死本来可通行的窄通道。
FMM_USE_SOFT_COST = True
#: A/B 实验覆盖：0=固定膨胀基线，1=双层代价场。留空时使用上面的默认值。
FMM_USE_SOFT_COST = _env_bool("HIMLOCO_FMM_SOFT_COST", FMM_USE_SOFT_COST)
#: 硬碰撞半径 [m]。当前仍是圆形近似；下一阶段将替换为 Go2 矩形 footprint 轨迹包络。
FMM_HARD_RADIUS_M = 0.20
#: 距障碍小于该距离开始产生软代价 [m]。
FMM_SOFT_CLEARANCE_M = 0.65
#: 软代价强度。0 等价于只有硬碰撞层；值越大越偏好高净空路线。
FMM_SOFT_COST_WEIGHT = 4.0
#: 软代价曲线指数。2.0 让远端影响平缓、靠近硬边界时快速增强。
FMM_SOFT_COST_POWER = 2.0

# ---------- 转向决策锁存（⚠ 修复 wz 逐步振荡导致的原地打转） ----------
#: 问题（2026-09-16 实测定位）：当 goal 与机器人连线【正对】一个障碍时，
#:   左右绕行代价相同，FMM 距离场的横向梯度在 0 附近抖动，其符号每步翻转，
#:   于是 wz 在 +max_wz / -max_wz 之间来回跳，机器狗原地打转直至顶死在障碍上。
#:   实测：wz 连续 170 步在 ±1.0 之间振荡，位移停在 x=7.31 不再前进，
#:   最终 stuck / timeout，成功率 0/10（这是“感知修好了仍全失败”的真正原因）。
#: 解法：横向证据不足时不做新决策，沿用上一次已确立的转向符号（滞回），
#:   并保证该符号至少维持若干步，让机器狗把绕行动作【做完整】再重新评估。
#: 取值：
#:   FMM_TURN_HYSTERESIS_THRESH  横向梯度 |dir_c| 小于该值视为“证据不足”。
#:       dir_c 是归一化后的方向分量，取值 [-1,1]；0.05 约对应 2.9° 的方向偏差，
#:       既能压住鞍点抖动，又不会钝化正常避障响应。设 0 = 关闭该判据。
#:   FMM_TURN_HOLD_STEPS  转向符号最少保持的控制步数。主循环 50Hz，
#:       40 步 = 0.8s，用于密集障碍中的绕行方向承诺；太长会钝化重新决策。设 0 = 关闭。
#: ⚠ 两者都为 0 时锁存完全关闭（= 原始行为）。
FMM_TURN_HYSTERESIS_ENABLE = True
FMM_TURN_HYSTERESIS_THRESH = 0.05   #: 横向梯度“证据不足”阈值（归一化分量）
FMM_TURN_HOLD_STEPS = 40            #: 转向符号最少保持步数（50Hz 下 40 步 = 0.8s）
#: 用户观察 12.5Hz 规划在近距离窄缝中的响应偏慢，恢复为每个 50Hz 控制步重规划。
#: 深度图仍是 12.5Hz，但机器人位姿和局部目标每步更新；角速度限幅继续抑制跳变。
FMM_REPLAN_INTERVAL_STEPS = 1
#: 沿 FMM 完整路径取前视点，帮助机身先对准窄缝中心线。
FMM_USE_PATH_LOOKAHEAD = True
#: 空地图不跟踪离散 FMM 前视点，而是直接跟踪连续目标方向；小于该角度保持直行。
#: 0.02 rad 约 1.1°：足以过滤厘米级定位噪声，同时在步态航向漂移肉眼可见前开始纠偏。
FMM_OPEN_SPACE_YAW_DEADBAND_RAD = 0.02
#: 限制 wz 变化率，避免一次重规划就从全左打到全右。
FMM_MAX_WZ_ACCEL = 3.0              #: [rad/s^2]
#: 当前 BEV 内不存在到局部目标的连通路径时，停止前进并朝较空一侧原地搜索。
FMM_STOP_IF_UNREACHABLE = True
FMM_UNREACHABLE_TURN_WZ = 0.6       #: [rad/s]

# ---------- 窄通道对准与降速 ----------
#: 路径前方 NARROW_PASSAGE_LOOKAHEAD_M 内的最小原始净空低于阈值时进入窄通道模式。
NARROW_PASSAGE_ENABLE = True
NARROW_PASSAGE_LOOKAHEAD_M = 1.5
NARROW_PASSAGE_CLEARANCE_M = 0.60   #: 路径中心到最近障碍表面的距离 [m]
NARROW_PASSAGE_MAX_VX = 0.35        #: 对准并穿越时的速度上限 [m/s]
NARROW_PASSAGE_MAX_WZ = 0.60        #: 防止缝内急转造成后腿/机身侧擦 [rad/s]

# ---------- FMM走廊 + 矩形足迹 MPPI ----------
MPPI_REPLAN_INTERVAL_STEPS = 4      #: 12.5 Hz采样优化；控制指令仍以50 Hz下发
MPPI_HORIZON = 20
MPPI_NUM_SAMPLES = _env_int("HIMLOCO_MPPI_SAMPLES", 384)
MPPI_ROLLOUT_DT = 0.10              #: 预测2秒
MPPI_MAX_VX = 1.0
MPPI_MIN_VX = 0.10
MPPI_MAX_WZ = 1.0
MPPI_FOOTPRINT_LENGTH_M = 0.72      #: Go2动态机身/腿部包络长度
MPPI_FOOTPRINT_WIDTH_M = 0.42       #: Go2动态机身/腿部包络宽度
MPPI_FOOTPRINT_MARGIN_M = 0.05
MPPI_TEMPERATURE = 8.0
MPPI_NOISE_VX = 0.22
MPPI_NOISE_WZ = 0.55

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
#: play1.py 导航时下发的 vx 上限（无障碍时按此速度走）。原代码写死 1.0。
NAV_FORWARD_VX = 1.0

# ---------- 前向净空速度调制（⚠ 修复“开了避障成功率反而下降”的关键） ----------
#: 背景（2026-09-16 实测定位）：原代码只把 FMM 算出的 wz 叠加上去，
#:   vx 恒为 NAV_FORWARD_VX=1.0 —— 上层【只转向、从不减速】。
#:   而大石头是静态刚体（fix_base_link=True），终止判据又是
#:   terminate_after_contacts_on=["base"]（机身碰到就算摔）。
#:   结果：机器狗以 1.0m/s 全速撞上不可动的石头，实测接触力高达 985N/645N，
#:   角速度误差飙到 9~12rad/s（剧烈翻滚），直接判 environment termination。
#:   对照实验（15 大石头、同 checkpoint、同 waypoint、10 trial）：
#:       BEV_ENABLE_UPDATE=False（不避障，直线穿）-> 10/10 成功
#:       BEV_ENABLE_UPDATE=True （避障但不减速）  ->  5/10 成功
#:   即“看得见障碍却仍全速撞上去”比“根本不看”更容易失败 —— 缺的就是这一环。
#:
#: 方案：每步沿机器人正前方的一个锥形窗口扫描障碍栅格，得到【前向净空距离】，
#:   再按梯形曲线把 vx 从 NAV_FORWARD_VX 平滑降到 SPEED_MOD_MIN_VX：
#:       净空 >= SPEED_MOD_SLOW_DIST  -> 全速
#:       净空 <= SPEED_MOD_STOP_DIST  -> 最低速（不停死，留给 wz 绕过去）
#:       中间线性过渡；并做加速度限幅，避免 vx 阶跃把盲走底层踢失稳。
#: 为什么“最低速”而不是“停死”：vx=0 会让机器狗原地卡到超时，
#:   而且 stuck 终止判据是 cmd_speed>0.3 才触发（见 TERM_STUCK_CMD_SPEED_MIN），
#:   把下限设在 0.3 以下还能顺带避免“慢速通行被误判为卡住”。
SPEED_MOD_ENABLE = True         #: 前向净空速度调制总开关
# ---------- 停滞逃逸（上层脱困，修复"原地踏步死锁到 timeout"）----------
#: 总开关。
#:   ⚠ 2026-09-17 停滞取证探针实测定位（/tmp/pitch/logs/stall_probe.pkl）：
#:     no_obstacles 场景里机器狗会【原地踏步死锁】直到 timeout。取证数据：
#:       step 1303~1353 位置死锁在 (16.16, 5.02)，纹丝不动；
#:       cmd=(vx 1.00, wz -0.54) 上层持续下发"前进+右转"；
#:       实际 vx = +0.105 / -0.060 / +0.015 / -0.021 正负抖动（原地踏步）；
#:       实际 yawrate = +0.05 / -0.17 / -0.18 完全没跟踪 wz=-0.54；
#:       pitch = -0.42rad(-24°)，terrain z=+0.750，站高 0.29m 正常；
#:       接触力 46~96N（不是撞障碍），BEV 障碍图全零。
#:     根因：BEV 只看 actor 障碍（白名单），地形坑洼不进障碍图 -> 上层认为
#:     "前方通畅" -> 永远重复同一条指令 -> 底层盲走策略在坑洼里出不来 -> 死锁。
#:     此前 no_obstacles 时高时低（0.17~0.90）的真凶也是它：底层随机指令重采样
#:     bug（resampling_time=10s）会偶发注入 wz∈[-3.14,3.14] 的随机转向，
#:     恰好充当了【隐式逃逸机制】把机器狗从死锁里摇出来；
#:     修掉那个 bug（EVAL_COMMAND_RESAMPLING_TIME=1e6）后逃逸能力消失，
#:     成功率从 0.58 掉到 0.17 —— 即历史高分建立在 bug 之上。
#:   True = 上层自己做停滞检测 + 逃逸动作（正解）；False = 原行为（死锁到 timeout）。
#:
#:   ✅ 实测定标（2026-09-17，各 15 trial，no_obstacles 论文场景）：
#:        逃逸开 0.53  vs  逃逸关 0.07   —— 差异 0.46，远超二项噪声(±0.12)，
#:        失败模式从"卡住(指令1.00实际0.002)"彻底转为"碰撞/timeout"。
#:   ✅ 消融（stones_only，12 trial）：
#:        有倒退段 0.25  vs  只原地转不倒退 0.08
#:        => 倒退是逃逸的【必要动作】。曾担心"BEV 只有前向视野、倒退是盲退会
#:           倒进石头"，实测证伪：盲退反而显著更好（0.25 vs 0.08）。
STALL_ESCAPE_ENABLE = True
#: 停滞判据：过去 STALL_WINDOW_STEPS 步内实际走过的【路程】小于 STALL_MIN_TRAVEL_M
#:   即判为停滞。用路程而不是净位移：原地踏步时净位移≈0、路程有抖动，
#:   路程判据更稳（探针实测停滞段 vx 在 ±0.1 间抖动，净位移 0.01m）。
#:   ⚠ 窗口必须【短于】环境的 TERM_STUCK_TIME_S(=1.0s=50步)，否则环境先判失败、
#:     轮不到逃逸；故取 30 步(0.6s)。
STALL_WINDOW_STEPS = 30         #: 停滞观察窗口 [step]
STALL_MIN_TRAVEL_M = 0.05       #: 窗口内路程低于该值算停滞 [m]
#: 逃逸动作：先倒退，再原地转，各持续若干步。
#:   倒退是四足脱困的标准手法（离开卡住脚的那个坑洼），
#:   之后原地转把朝向换一个方向，避开刚才走不通的那条线。
STALL_REVERSE_STEPS = 25        #: 倒退步数（0.5s）
STALL_REVERSE_VX = -0.4         #: 倒退速度 [m/s]（负值）
STALL_TURN_STEPS = 35           #: 原地转向步数（0.7s）
STALL_TURN_WZ = -0.9            #: 逃逸转向角速度 [rad/s]（符号由 STALL_TURN_ALTERNATE 决定）
#: 每次逃逸交替转向方向（True）还是固定方向（False）。交替可避免反复转回同一个死锁位。
STALL_TURN_ALTERNATE = True
#: 逃逸期间忽略 FMM 的 wz 与速度调制，直接用逃逸指令（必须如此，否则上层会把它拉回去）。
STALL_OVERRIDE_PLANNER = True
#: 一个 trial 内最多触发几次逃逸，超过就放弃（防止无限逃逸刷满 timeout 反而更慢）。
STALL_MAX_ESCAPES = 6
SPEED_MOD_MIN_VX = 0.25         #: 逼近障碍时的最低前进速度 [m/s]（0 = 会原地卡死）
SPEED_MOD_SLOW_DIST = 2.0       #: 净空小于此距离开始减速 [m]
SPEED_MOD_STOP_DIST = 0.7       #: 净空小于此距离降到最低速 [m]
SPEED_MOD_CONE_HALF_ANGLE = 25.0  #: 前向扫描锥的半角 [deg]（覆盖机身宽度+膨胀余量）
#: 像素级前向净空判据的“地面比”：相机平视时每一行 v 对应一个俯角 θ，
#: 该行若是地面，深度应为 cam_height/sin(θ)；实测深度小于该值的 GROUND_RATIO 倍，
#: 就认为这个像素打在【比地面更近的直立物】上。
#: 取值权衡：越接近 1 越灵敏但越容易把坡地/地面起伏误判成障碍；
#:   0.85 = 允许地面深度有 15% 的估计误差（深度噪声 + 地形起伏），
#:   离线单测：10% 上坡不误判（命中 0 px），墙面 0.3~3.0m 全部测准（误差 <0.05m）。
SPEED_MOD_GROUND_RATIO = 0.85
#: 像素级前向净空判据【专用】的纵向 ROI 起始比例（与 BEV_V_START_RATIO 解耦）。
#: ⚠ 只有这个判据用宽 ROI；BEV 栅格图仍用 0.55（放宽会污染障碍图、拉低成功率）。
SPEED_MOD_ROI_START = 0.10
#: 是否启用【像素级地面比较】作为前向净空的第二数据源。
#:
#: ⛔ 默认 False：这个判据【未通过验证，当前不可用】（2026-09-16 实测）。
#:   思路是“相机平视时每行 v 对应俯角 θ，若该行是地面则深度应为 cam_height/f(θ)，
#:   实测深度显著更近就说明打在了直立物上”。但实现里 f(θ) 用错了：
#:       写的是 cam_height / sin(θ)  -> 得到的是【斜距】
#:       而 IsaacGym 的 IMAGE_DEPTH 是【轴向深度】（z-buffer，沿光轴的距离），
#:       地面在轴向深度上应满足 cam_height / tan(θ)。
#:   两者相差 1/cos(θ)，于是判据退化成“俯角 > 31.8° 的像素一律算障碍”，
#:   与真实场景无关。实测误报率（no_obstacles 场景，无任何石头/树）：
#:       cam_height=0.25(旧错值) -> 误报 48%
#:       cam_height=0.619(已修正) -> 误报 99%（净空恒报 0.6~0.8m，机器狗被死死压速）
#:   ✅ 上述两件事已在 2026-09-16 完成：
#:     1) 地面模型已改为 cam_height / tan(总俯角)（bev_height_mapper.forward_obstacle_distance，
#:        含光轴俯角支持）；
#:     2) 真实仿真图复测：3 张平地平地图（nostone/left20/right20）误报 0 px
#:        （旧 sin 模型 6000+ 误报像素），有石图正确报出净空 2.19m。
#:   默认仍 False：端到端增益由矩阵实验 R6/R7/R8 确认后再定。它与 BEV 锥扫
#:   取 min，任何残余误报都会把 vx 砍到最低速；BEV 锥扫单路径已验证可用
#:   （stones_only 场景把成功率从 0.50 提到 0.80）。
#: 像素判据命中距离下限 [m]：忽略比它更近的命中（机身自遮挡带：俯角机位下
#:   前腿/胸口离相机 0.35~0.5m，会被当成直立物把 vx 永久压死，实测端到端 0/10）。
#:   0.6m @1.0m/s 仅 0.6s，本就在反应下限之内，忽略无损失。
SPEED_MOD_PIXEL_MIN_FWD = 0.6
SPEED_MOD_USE_PIXEL_JUDGE = False
SPEED_MOD_MAX_RANGE = 3.0       #: 前向扫描最大距离 [m]（<= BEV_X 才有意义）
SPEED_MOD_MAX_ACC = 2.0         #: vx 变化率限幅 [m/s^2]，防止速度阶跃踢翻底层
# ---------- 速度-转向耦合（修复“急转画大圆弧顶死”）----------
#: 总开关。FMM 假设全向点机器人，而盲走底层转弯半径 = vx/wz：wz=±1.0 + vx=1.0
#:   时轨迹是半径 1m 的圆弧、弯向障碍一侧并顶死（2026-09-16 wz 日志实测）。
#:   急转时压低 vx 目标可把转弯半径缩到同比例，机器狗才转得动。
SPEED_TURN_COUPLING_ENABLE = True
#: |wz| 低于该值不耦合（近似直行）[rad/s]
SPEED_TURN_WZ_LO = 0.15
#: 【急转档】独立开关：|wz| >= SPEED_TURN_WZ_HARD 时不看 clearance 直接耦合。
#:   ⚠ 默认 False。2026-09-17 三组实测（各 10 trial）证明这一档是【净负收益】：
#:       · 单级耦合（仅 clearance 档）: no_obstacles 0.90 / stones_only 0.80
#:       · 加上急转档:                 no_obstacles 0.00~0.10 / stones_only 0.00
#:     失效机理：航点转角处 |wz| 常达 0.7~1.0，急转档把 vx 压到 0.3 后，
#:     盲走底层在粗糙坡上【失去前进动量】，指令 0.8~1.0 而实际 0.000，判"卡住"终止。
#:     stones_only 场景白名单一直生效（n=16>1），两组唯一变量就是本档，因果确定。
#:   True = 启用急转档（仅建议在极窄廊 + 平缓地面下试）。
SPEED_TURN_HARD_ENABLE = False
#: |wz| 达到该值【无条件】耦合（不看 clearance）[rad/s]。
#:   航点转角急转时 1m 转弯圆弧在粗糙坡打滑侧翻；白名单后 clearance≡3.0，
#:   仅靠 clearance 条件触发不到，故急转单独一档（2026-09-17 实测定标）。
#:   ⚠ 本值仅在 SPEED_TURN_HARD_ENABLE=True 时才起作用，见上方实测结论。
SPEED_TURN_WZ_HARD = 0.7
#: |wz| 达到该值耦合到最低速 [rad/s]
SPEED_TURN_WZ_HI = 0.5
#: 急转时的前进速度目标下限 [m/s]（转弯半径 = 本值 / wz）
SPEED_TURN_MIN_VX = 0.3
#: 耦合启用的净空上限 [m]：clearance 大于它（开阔地航点转向）不耦合，
#:   避免粗糙地面上降速失去动量顶死（实测 stones_only 0.60 vs 不耦合 1.00）；
#:   clearance 小于它（真在绕障碍）才耦合，防止 1m 转弯圆弧顶住石头。
#:   0 = 无条件耦合。
SPEED_TURN_CLEARANCE_M = 2.0
#: 到达 waypoint 的判定半径 [m]
NAV_GOAL_REACH_THRESH_M = 0.15
#: 最终目标附近的显式减速区；进入 1m 后线性降速，保证能进入 0.15m 到达圆。
NAV_GOAL_SLOW_DIST_M = 1.0
NAV_GOAL_APPROACH_MIN_VX = 0.20
#: 越过中间航点平面即判为通过（GPS 导航标准行为）。最终目标永不使用该规则，
#:   必须进入 NAV_GOAL_REACH_THRESH_M 定义的欧氏距离圆。
NAV_ADVANCE_ON_CROSS = True
#: 航点在身后多少米算“已越过” [m]。太小会被步态摆动误触发，太大等于跳过航点。
NAV_CROSS_MARGIN_M = 0.2
#: 中间航点到达阈值 [m]。⚠ 必须 >= 最小转弯半径 v/wz_max（1.0m/s ÷ 1.0rad/s = 1.0m），
#:   否则 goal 落进转弯圆内会触发纯追踪极限环：wz 饱和 ±1.0、全速跑圈、永不到达
#:   （2026-09-16 wz 日志实测：连续 90 步 wz=-1.0 且 clearance=3.0 无障）。
#:   最终目标点不用它，仍用 NAV_GOAL_REACH_THRESH_M。
NAV_MID_GOAL_REACH_THRESH_M = 1.0
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
        LIVE_FOLLOW_VIEWER_CAMERA=False, LIVE_HOLD_FINAL_ROUTE=False,
        LIVE_VIEWER_SYNC=True, LIVE_DRAW_INTERVAL=1,
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
        LIVE_FOLLOW_VIEWER_CAMERA=False, LIVE_HOLD_FINAL_ROUTE=False,
        LIVE_VIEWER_SYNC=True, LIVE_DRAW_INTERVAL=1,
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
        RENDER=True,
    ),
}

# apply_preset() 会把下面两个名字覆盖成解析后的 True/False，
# 所以先把“用户声明值”（None=自动 / True / False）单独存一份，
# 供 resolve_*() 反复调用时判断是否仍处于自动模式。
_DECLARED_CAM_READ = CAM_READ_ENABLE
_DECLARED_CAM_READ_RGB = CAM_READ_RGB


def cam_read_consumers():
    """返回“需要每步深度图”的开关名列表（用于自动联动与一致性校验）。"""
    need = []
    if BEV_ENABLE_UPDATE:
        need.append("BEV_ENABLE_UPDATE")
    if SAVE_BEV_FMM_IMAGES:
        need.append("SAVE_BEV_FMM_IMAGES")
    if SAVE_CAMERA_IMAGES:
        need.append("SAVE_CAMERA_IMAGES")
    if SAVE_RGB_IMAGE:
        need.append("SAVE_RGB_IMAGE")
    if SAVE_DEPTH_IMAGE:
        need.append("SAVE_DEPTH_IMAGE")
    if LIVE_SHOW_CAMERA_IMAGES:
        need.append("LIVE_SHOW_CAMERA_IMAGES")
    return need


def cam_rgb_consumers():
    """返回“需要每步 RGB 图”的开关名列表（避障本身不需要 RGB，只要 depth）。"""
    need = []
    if SAVE_RGB_IMAGE:
        need.append("SAVE_RGB_IMAGE")
    if LIVE_SHOW_CAMERA_IMAGES:
        need.append("LIVE_SHOW_CAMERA_IMAGES")
    if SAVE_CAMERA_IMAGES and SAVE_RGB_IMAGE:
        pass  # 已由 SAVE_RGB_IMAGE 覆盖，避免重复列出
    return need


def resolve_cam_read_rgb(value=None):
    """CAM_READ_RGB=None(自动) 时按消费方解析；显式值原样返回。

    只有真的要存/显示 RGB 才拷这张图，否则每步白拷一张，泄漏量翻倍。
    """
    if value is None:
        value = _DECLARED_CAM_READ_RGB
    if value is not None:
        return bool(value)
    return len(cam_rgb_consumers()) > 0


def resolve_cam_read(value=None):
    """CAM_READ_ENABLE=None(自动) 时按消费方解析成 True/False；显式值原样返回。

    这是必要的：BEV_ENABLE_UPDATE=True 却 CAM_READ_ENABLE=False 会拿到空深度图，
    避障静默失效。让预设只管“要不要存图/建图”，读不读相机由此处自动推导。
    """
    if value is None:
        value = _DECLARED_CAM_READ
    if value is not None:
        return bool(value)
    return len(cam_read_consumers()) > 0


def apply_preset(name):
    """把预设写进本模块的全局命名空间。

    因为 pathplanner.py / play.py 是在 import 时通过 `import *` 取走这些名字，
    所以必须在模块加载阶段就完成覆盖（见文件末尾）。
    """
    if name not in PRESETS:
        raise KeyError("未知预设: %r，可选: %s" % (name, sorted(PRESETS.keys())))
    for key, value in PRESETS[name].items():
        globals()[key] = value
    # 预设本身不必声明 CAM_READ_ENABLE；这里按消费方自动联动，
    # 避免“开了建图却忘了开相机读取”这类静默失效。
    globals()["CAM_READ_ENABLE"] = resolve_cam_read(
        PRESETS[name].get("CAM_READ_ENABLE", None))
    globals()["CAM_READ_RGB"] = resolve_cam_read_rgb(
        PRESETS[name].get("CAM_READ_RGB", None))
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
    # 只列用户开关：全大写、非下划线开头（排除 _VIZ_MODULE_READY 这类内部标志）
    on = [name for name, value in sorted(globals().items())
          if name.isupper() and not name.startswith("_") and value is True]
    check_consistency(verbose=True)
    cam_state = "depth=%s rgb=%s" % (
        "开" if CAM_READ_ENABLE else "关",
        "开" if CAM_READ_RGB else "关")
    percept = ("感知降频=每%d步" % PERCEPT_UPDATE_INTERVAL
               if CAM_READ_ENABLE and PERCEPT_UPDATE_INTERVAL else "感知降频=关")
    fmm_rate = ("FMM每步(约50Hz)" if int(FMM_REPLAN_INTERVAL_STEPS) == 1
                else "FMM每%d步" % int(FMM_REPLAN_INTERVAL_STEPS))
    if PLANNER_MODE == "mppi":
        planner_rate = ("模式=MPPI | %s,MPPI每%d步,世界记忆=%s" % (
            fmm_rate, int(MPPI_REPLAN_INTERVAL_STEPS),
            "开(%d步)" % int(TEMPORAL_MEMORY_MAX_AGE_STEPS)
            if TEMPORAL_MEMORY_ENABLE else "关"))
    else:
        planner_rate = "模式=%s | %s" % (PLANNER_MODE, fmm_rate)
    # 感知/规划关键参数：这几项直接决定“能不能看见障碍、会不会撞”，
    # 每次启动都打出来，避免出现“配置改了但没生效”的静默错误。
    step_judge = ("开(阈值%s,跨%d格)" % (BEV_STEP_THRESH or BEV_HEIGHT_RANGE_THRESH,
                                       BEV_STEP_SPAN)
                  if BEV_ENABLE_STEP_JUDGE else "关")
    hyst = ("障碍区开(阈值%.2f,保持%d步),空地死区%.2frad" % (
                FMM_TURN_HYSTERESIS_THRESH, FMM_TURN_HOLD_STEPS,
                FMM_OPEN_SPACE_YAW_DEADBAND_RAD)
            if FMM_TURN_HYSTERESIS_ENABLE else "关")
    fmm_cost = (
        "双层(硬半径%.2fm,软距离%.2fm,权重%.1f)" % (
            FMM_HARD_RADIUS_M, FMM_SOFT_CLEARANCE_M, FMM_SOFT_COST_WEIGHT)
        if FMM_USE_SOFT_COST else
        "固定膨胀(半径%.2fm)" % FMM_INFLATE_RADIUS_M)
    nav_state = (
        "BEV建图=%s ROI起始=%.2f 台阶判据=%s 记忆融合=%s | "
        "FMM代价=%s | 速度调制=%s(最低%.2fm/s,%.1f→%.1fm) | %s | 转向锁存=%s" % (
            "开" if BEV_ENABLE_UPDATE else "关",
            BEV_V_START_RATIO,
            step_judge,
            "开" if BEV_ENABLE_MEMORY_FUSION else "关",
            fmm_cost,
            "开" if SPEED_MOD_ENABLE else "关", SPEED_MOD_MIN_VX,
            SPEED_MOD_SLOW_DIST, SPEED_MOD_STOP_DIST,
            "横向符号%+.0f 俯角%.0f° 近距掩蔽%.2fm" % (
                BEV_LATERAL_SIGN, BEV_CAM_PITCH_DEG, BEV_MIN_RANGE_M),
            hyst,
        )
    )
    # 本轮新增的三组关键开关也打出来，避免“改了配置但没生效”的静默错误：
    #   · 障碍白名单（只绕 actor 障碍，不绕地形/边界墙）
    #   · 停滞逃逸（原地踏步死锁时倒退+转向脱困）
    #   · 指令重采样周期（评测期禁止底层随机覆盖上层规划器指令）
    escape_state = (
        "开(窗口%d步<%.2fm触发,倒退%d步@%.1fm/s+转向%d步@%.1frad/s%s,上限%d次)" % (
            STALL_WINDOW_STEPS, STALL_MIN_TRAVEL_M,
            STALL_REVERSE_STEPS, STALL_REVERSE_VX,
            STALL_TURN_STEPS, STALL_TURN_WZ,
            ",交替方向" if STALL_TURN_ALTERNATE else ",固定方向",
            STALL_MAX_ESCAPES)
        if STALL_ESCAPE_ENABLE else "关")
    new_state = (
        "障碍白名单=%s(掩膜半径%.2fm,刹车用%s) | 停滞逃逸=%s | "
        "急转降速档=%s | 指令重采样=%gs(1e6=已禁止底层随机覆盖) | 相机跟随=%s" % (
            "开" if BEV_OBSTACLE_ACTORS_ONLY else "关",
            BEV_ACTOR_MASK_RADIUS_M,
            "原始几何图" if BEV_CLEARANCE_USE_RAW_MAP else "白名单图",
            escape_state,
            "开" if SPEED_TURN_HARD_ENABLE else "关",
            EVAL_COMMAND_RESAMPLING_TIME,
            "开" if LIVE_FOLLOW_VIEWER_CAMERA else "关",
        )
    )
    return "preset=%s | 已开启: %s\n%s\n相机读取: %s | %s\n%s\n%s" % (
        ACTIVE_PRESET, ", ".join(on) if on else "无", scene_summary(),
        cam_state, percept + " | " + planner_rate, nav_state, new_state)

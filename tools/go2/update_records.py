#!/usr/bin/env python3
"""Generate the two requested desktop Markdown records; never connects to robot."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME = ROOT / "go2_runtime"
DOCS = ROOT / "docs/go2"
TZ = ZoneInfo("Asia/Shanghai")


def read_json(path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return default


def link(path):
    # Historical operation text remains; links point to the current layout.
    legacy = ROOT / "deploy/ours"
    if path.is_relative_to(legacy):
        path = RUNTIME / path.relative_to(legacy)
    aliases = {
        'MOBILE_ACCESS.md': 'BOARD_WEB_GUIDE.md',
        'WIRED_WEB_GUIDE.md': 'BOARD_WEB_GUIDE.md',
        'WEB_CONSOLE_FEASIBILITY.md': '../../go2_web/README.md',
        'WIFI_PLAN.md': 'AP_GUIDE.md',
        'WIFI_CASE_STUDIES.md': 'AP_GUIDE.md',
        'NATIVE_AP_CHECK_20261102.md': 'AP_GUIDE.md',
    }
    if not path.exists():
        target = DOCS / aliases.get(path.name, path.name)
        if target.exists(): path = target.resolve()
    return "[{}]({})".format(path.name, str(path).replace(" ", "%20"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", default="records")
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--source", default="tools/go2/update_records.py")
    parser.add_argument("--motion-capable", action="store_true", help="Record a separately confirmed real motion action")
    args = parser.parse_args()
    now = datetime.now(TZ).isoformat(timespec="seconds")
    action_label = {"check":"本机依赖检查", "test":"离线测试", "prepare":"部署准备", "shadow":"只读影子规划",
                    "trial":"现场低速试验", "records":"更新操作记录", "web":"离线网页服务",
                    "wifi_survey":"无线接入只读检查", "web_console_analysis":"网页方案分析",
                    "web_console_implemented":"完成离线网页实验台", "web_console_layout":"调整浅色首屏布局", "web-live":"只读实时相机网页", "web_camera_verified":"实测网页只读实时相机", "web_console_fuller":"放大相机窗口并填满状态区", "web_console_singlepage":"完成单屏前端与无线切换准备", "web_board_deployed_ap_pending":"板载网页登录/自启动完成，无线AP通路待解决", "web-deploy":"板载网页部署", "web-lan":"电脑局域网页面", "web-lan-live":"电脑局域网相机页面", "web_identity_updated":"网页更名GO2并按用户要求重设登录密码", "wifi_case_studies":"调研Go2无线网卡与路由器案例"}.get(args.action, args.action)
    event = dict(time=now, action=args.action, exit_code=args.exit_code,
                 source=args.source, motion_capable=args.motion_capable or args.action == "trial")
    with (HERE/"operations.jsonl").open("a") as log:
        log.write(json.dumps(event, ensure_ascii=False)+"\n")
    try:
        desktop = Path(subprocess.check_output(["xdg-user-dir", "DESKTOP"], text=True).strip())
    except (OSError, subprocess.SubprocessError):
        desktop = Path.home()/"桌面"
    desktop.mkdir(parents=True, exist_ok=True)
    simple_path = desktop/"HIMLoco实机操作简要记录_20261030.md"
    detail_path = desktop/"HIMLoco实机操作详细记录_20261030.md"
    events = [json.loads(s) for s in (HERE/"operations.jsonl").read_text().splitlines() if s.strip()]
    board_deployed = any(e["action"] == "web_board_deployed_ap_pending" for e in events)
    wifi_checked = any(e["action"] == "wifi_survey" for e in events)
    web_analyzed = any(e["action"] == "web_console_analysis" for e in events)
    web_built = any(e["action"] == "web_console_implemented" for e in events)
    camera_verified = any(e["action"] == "web_camera_verified" for e in events)
    light_layout = any(e["action"] == "web_console_layout" for e in events)
    runs = []
    for folder in sorted((RUNTIME/"runs").glob("*")):
        if not folder.is_dir():
            continue
        summary = read_json(folder/"summary.json")
        if summary is not None:
            runs.append((folder, summary, read_json(folder/"metadata.json", {})))
    cal = read_json(RUNTIME/"calibration.local.json", {})
    motion = read_json(RUNTIME/"runs/ours_first_walk_20261030/motion_analysis.json", {})
    motion_runs = [(folder, summary, meta) for folder, summary, meta in runs
                   if summary.get("sent_velocity_commands")]
    fixed_trials = [(folder, summary, meta) for folder, summary, meta in motion_runs
                    if meta.get("min_vx", 0) >= 0.10 and meta.get("goal_relative") == [0.5, 0.0]]
    fixed_status = ("固定每秒零点一米、最长五秒的新测试尚未执行。" if not fixed_trials else
                    "日志中已出现后续固定速度试验，具体结果需查看详细记录和现场反馈，不能只按脚本退出状态判断成功。")
    simple = f"""HIMLoco 实机操作简要记录

记录更新于北京时间 {now}。这里只记录操作经过，具体文件和命令见另一份详细记录。早期操作依据源码、试验日志和现场反馈整理。

重新连接网线后，确认本机能读取机器狗的关节、姿态和运动状态。随后登录板载算力盒子，确认深度相机连接正常，并成功读取深度图、彩色图和真实内参。

只接入第三章的双层代价 FMM 和标准足迹 MPPI 高层，由机器狗自带控制系统接收速度。最初在本机增加了感知、规划、通信和日志程序；最初算力盒子只在临时目录增加相机采集程序，没有刷机或安装新的板载驱动。

多轮只计算、不发速度的测试中，发现了竖直柜门漏检，已补充实机障碍识别和近障停车。还加入数据过期及通信超时拦截。相机高度和倾角由地面深度初步估计，前向位置按用户测量范围取三十四厘米，标定尚未完整验证。

用户同意后，最初执行了一次十秒速度试验，最高速度每秒零点一米。实际建议速度约每秒零点零五到零点零七米。机器狗仅身体前倾后收回，没有迈步，最终停稳；不能记作导航成功，也不能把里程计约七厘米变化写成实际行走距离。

当前怀疑小速度不足以启动有效步态，但本机启动范围还需实测。{fixed_status}当前日志共有 {len(motion_runs)} 轮写过速度通信管道，后续结果以详细记录和现场反馈为准。

项目根目录已新建一键部署文件夹。默认只部署和影子规划，实际运动使用单独入口并由现场确认。每次通过统一入口执行后，自动刷新桌面这两份记录。

最近一次记录操作：{action_label}，退出码 {args.exit_code}。软件正常退出不等于导航成功。
"""
    if wifi_checked:
        simple += "\n另已核查无线接入方式：机器狗支持热点和加入路由器，但相机算力盒子和开发接口能否无线直达尚未验证。本次只扫描和制定方案，没有改网络，也没有新增运动测试。\n"
    if web_analyzed and not web_built:
        simple += "\n另已离线查看 M20 Pro 参考项目的网页、控制仲裁、自启动和录包源码，形成 Go2 网页实验台可行性方案。这一步未连接机器狗，未搭建网页，也未安装开机服务。\n"
    if web_built:
        simple += "\n参考 M20 项目结构，已完成本机离线网页实验台，可看历史图像、地图、遥测，模拟测试的开始和停止，保存参数、实验记录并导出。真实历史与生成数据分别标记。通过八项后台测试及浏览器检查，没有连接狗或发送速度，也没有安装狗端自启动服务。\n"
    if light_layout:
        simple += "\n按用户要求，网页已调整为浅色系，把彩色和深度画面移到局部感知与规划上方，将运行摘要缩小移至右侧；已检查常见桌面尺寸，三块画面都能在首屏看到。\n"
    if camera_verified:
        simple += "\n已将 RGB 与深度窗口改为正方形铺满，原始画面居中裁切。机器狗连线后，重新建立 SSH、上传并编译临时相机程序，接通网页实时 RGB/深度，接收约 15 fps，网页约 5 次/秒刷新；手动断开和重连通过。仅相机实时，地图与遥测仍为历史，本次没有发送运动指令、没有录包或安装自启动服务。\n"
    if any(e["action"] == "web_console_fuller" for e in events):
        simple += "\n进一步放大网页相机窗口：桌面宽度 1280 时每幅约 256×256，较此前约 192×192 放大三分之一；四张状态卡片在右侧纵向排列，与相机区域上下对齐，消除下方空白。地图高度随可用首屏空间调整；未更改取流或运动逻辑。\n"
    if any(e["action"] == "web_console_singlepage" for e in events):
        simple += "\n网页首页已删除最近实验列表，记录统一保留第二页；压紧顶部并改为单屏不可滚动。窄屏用感知、配置、遥测标签切换。已验证记录搜索/返回、参数保存与实时画面持续更新。拔线前只读检查发现盒子仅显示有线网络，未检测到可用 Wi-Fi 接口；热点能否访问盒子还需无线连接后验证，没有修改网络或新增运动。\n"
    if board_deployed:
        simple += "\n2026-11-01 已把网页登录、实验历史副本和只读相机部署到算力盒子的持久目录，设置网页开机启动。相机在盒子本地读取并压缩预览，账号登录/退出及约15fps取流通过；未启动物理导航、速度桥或实机录包。20项后台测试通过。电脑连接go2热点得到192.168.12.32，关闭电脑网线并显式添加到盒子的无线测试路由后，网页仍超时；热点8765拒绝连接。原Wi-Fi与网线均已恢复，不能记作手机无线验收完成。需要盒子USB无线网卡或无线桥；已确认匹配内核头文件存在，但尚无网卡型号，未安装驱动。\n"
    if any(e["action"] == "web_identity_updated" for e in events):
        simple += "\n按用户要求，将网页大名称、标签页标题和登录页改为GO2，网页登录密码已按用户指定重设，账号仍为himloco；电脑与盒子校验文件同步，重启本网页服务后浏览器新密码登录通过。只修改网页登录，没有修改盒子SSH密码。只读检查SSH端口：算力盒子192.168.123.18开放，运动控制端192.168.123.161拒绝连接；SSH仍依赖实际网络通路，不能替代无线硬件。\n"
    if any(e["action"] == "wifi_case_studies" for e in events):
        simple += "\n已调研其他Go2项目的USB无线网卡和小路由器方案，确认两者均有案例，整理了盒子自建热点、无线SSH/网页和路由器排障的原始链接。导航算法仍在电脑，并非全部代码已部署到盒子；本轮只调研，没有更改网络、安装驱动或发运动指令。\n"
    details = [f"# HIMLoco 实机操作详细记录\n\n更新时间：{now}（Asia/Shanghai）。\n\n"
               "本记录由 `real_robot_deploy/update_records.py` 生成。早期记录从会话中保留的文件和试验日志重建，"
               "不是操作系统的完整审计日志；一键入口后续执行会自动追加动作、时间、退出码。密码不写入本记录。\n",
               "## 1. 当前结果与未完成项\n\n"
               "真实 D435i、Go2 状态、双层代价 FMM 和标准足迹 MPPI 的影子链路已经接通。"
               f"当前日志记录 {len(motion_runs)} 轮写过速度管道。首轮为经用户授权的 10 秒真实速度试验；用户确认没有迈步，导航未成功。"
               +fixed_status+
               "需要继续标定厂商速度接口有效范围、验证步态、停止响应和遥控接管，然后再做单障碍绕行。\n",
               "## 2. 设备和连接\n\n"
               "| 项目 | 实际检查结果 |\n| --- | --- |\n"
               "| 本机项目 | `/home/lenovo/HIMLoco` |\n"
               "| 桌面 | `/home/lenovo/桌面` |\n"
               "| 有线接口 | `enx00e04c584b68`，`192.168.123.200/24` |\n"
               "| Go2 控制器地址 | `192.168.123.161`，已验证网络可达 |\n"
               "| 算力盒子 | `unitree@192.168.123.18`，主机名 ubuntu |\n"
               "| 板载型号 | 设备树报告 NVIDIA Orin Nano Developer Kit，aarch64 |\n"
               "| 板载系统 | Ubuntu 20.04.5，内核 5.10.104-tegra，已有 ROS Foxy / Noetic |\n"
               "| 相机 | D435i，USB 3 链路 5000 Mbps，固件 5.15.1.55 |\n"
               "| 相机程序 | 复用已有 librealsense C++ 库，未安装 ROS 相机驱动 |\n"
               "| 图像配置 | 深度 Z16 + RGB8，各 640×480、请求 15 Hz |\n"
               "| 深度内参示例 | fx=fy≈386.664，cx≈317.288，cy≈239.499；运行时逐帧读取 |\n"
               "| 位姿来源 | Go2 SportModeState 本体里程计，不是 VIO |\n"
               "| 读取话题 | DDS `rt/sportmodestate`；前期另验证 `rt/lowstate` |\n"
               "| 速度接口 | 宇树 SDK2 SportClient.Move(vx,0,wz)，退出请求 StopMove |\n",
               "## 3. 本机新增、修改文件\n\n"
               "统一入口没有复制 FMM/MPPI 算法，复用现有实机运行目录。\n\n"
               "| 文件 | 作用与变更 |\n| --- | --- |\n"]
    descriptions = {
        "docs/go2/VIDEO_QUALITY.md": "本机高质量预览、独立15Hz循环、离线验证和明日实机验收项。",
        "docs/go2/GO2_EDU_HARDWARE.md": "EDU主机与设备调研，区分官方系列配置和本机实测开发入口。",
        "go2_runtime/manage.sh": "网线实机网页一键入口；本机免登录，启动不发送速度。",
        "docs/go2/WIRED_WEB_GUIDE.md": "当前网线开发流程、目标坐标、计算路径、标定与记录限制。",
        "go2_web/wired.py": "真实状态/相机共享、地面复核、影子规划、受限执行、心跳停止、独立真实记录。",
        "go2_web/test_wired.py": "实机模式输入时效、执行约束、标定确认和心跳停止的离线测试。",
        "go2_runtime/manage.sh": "统一入口；默认只读 shadow；prepare 部署；trial 明确运动；test 离线检查；退出刷新记录。",
        "tools/go2/update_records.py": "生成桌面两份文档；追加一键操作历史；汇总运行结果和源码 SHA256。",
        "docs/go2/README.md": "一键入口命令、运动范围、环境变量和记录位置。",
        "docs/go2/WIFI_PLAN.md": "无线接入评估、当前网络检查结果、图像带宽及录包方案；尚未切换网络。",
        "docs/go2/WEB_CONSOLE_FEASIBILITY.md": "离线分析 M20 网页系统的可借鉴架构，以及 Go2 板载网页、自启动、无线和录包的实现前提。",
        "go2_web/server.py": "默认离线HTTP后台；--wired启用本机实机接口、标定/任务控制与真实记录导出。",
        "go2_web/static/index.html": "实验工作台、实验记录、参数与环境三个页面及离线标记。",
        "go2_web/static/style.css": "浅色工作台样式；相机画面前置、运行摘要右置，桌面及窄屏布局。",
        "go2_web/static/app.js": "历史时间轴、快照/代价图显示、模拟任务、停止、记录开关、参数和下载；隐藏画布不重绘。",
        "go2_web/test_console.py": "8 项后台离线测试，使用临时文件和本地临时端口，不访问机器狗。",
        "go2_web/README.md": "离线网页启动、数据含义、API 与测试说明。",
        "tools/go2/.gitignore": "忽略本地操作历史等运行文件。",
        "go2_runtime/run_ours.py": "真实深度反投影、外参变换、既有 FMM/MPPI 调用、时钟偏移估计、速度拦截、目标判定与记录。",
        "go2_runtime/robot_bridge.cpp": "DDS 状态订阅；默认只读；显式 execute 才创建速度客户端；约 50Hz 刷新、限幅和独立超时停车。",
        "go2_runtime/bridge_safety.hpp": "与 DDS 分离的非法指令、状态过期、指令过期、倾斜及高度拦截逻辑。",
        "go2_runtime/camera_stream.cpp": "板载 D435i 采集，深度/RGB/内参/帧号/采集端时间通过 SSH stdout 传输；剔除重复深度帧。",
        "go2_runtime/estimate_floor.py": "用 RANSAC/SVD 从平地深度估高度和倾斜，输出初步外参；不假称能解出 X/Y 和安装航向。",
        "go2_runtime/preview.py": "将快照渲染为 RGB、深度、BEV、路径/MPPI 候选和软代价图。",
        "deploy/build_runtime.sh": "用本机现有宇树 C++ SDK 编译状态/速度桥。",
        "deploy/prepare_camera.sh": "通过 SSH 将相机源码写入盒子 /tmp 并编译；不创建自启服务。",
        "go2_runtime/run.sh": "指定 Python 解释器和本地 .deps，启动 run_ours.py；自身不默认启用运动。",
        "go2_runtime/calibration.example.json": "未测占位参数，只可调试，不能直接用于运动。",
        "go2_runtime/requirements.txt": "规划运行依赖列表，scikit-fmm 固定为 2025.6.23。",
        "go2_runtime/test_runtime.py": "15 项新增离线测试，涵盖几何、地面拟合、竖直障碍、近障、失效数据和最小速度。",
        "go2_runtime/test_bridge_safety.cpp": "脱离机器人检查通信桥看门狗的输入边界和锁存行为。",
        "go2_runtime/README.md": "早期接入过程、技术边界、影子测试和首次速度试验结果。",
        "go2_runtime/.gitignore": "忽略编译产物、现场图像日志、依赖和本机标定文件。",
        "legged_gym/legged_gym/envs/go2/bev_height_mapper.py": "仅将 Go2RoughCfg 导入移入两个仿真绘图函数，使实机可加载算法而不初始化 Isaac Gym。原仿真建图计算未改。",
    }
    for rel, description in descriptions.items():
        details.append(f"| `{rel}` | {description} |\n")
    details += ["\n直接复用的原算法：`bev_height_mapper.py` 中 FMMGradientController，以及 `mppi_local_planner.py` 中 FootprintMPPIController。"
                "没有改这两个优化器来替换成其他方法；当前使用 standard MPPI，时序记忆关闭，未启用第四章通行性模型。\n",
                "## 4. 算力盒子端实际操作\n\n"
                "通过 SSH 只读检查 hostname、系统版本、设备树型号、USB 拓扑、视频设备、RealSense 枚举、ROS 包和话题。"
                "早期以已有 RealSense 开发库编译短时读图验证程序，随后部署正式相机流程序。\n\n"
                "| 盒子路径 | 作用 |\n| --- | --- |\n"
                "| `/tmp/himloco_camera_readonly.cpp` | 初次深度/RGB 读取验证源码 |\n"
                "| `/tmp/himloco_camera_readonly` | 初次读取验证二进制 |\n"
                "| `/tmp/himloco_ours_camera_stream.cpp` | 正式 SSH 相机流源码；prepare 时更新 |\n"
                "| `/tmp/himloco_ours_camera_stream` | 正式采集程序；运行结束退出，重启后 /tmp 文件可能消失 |\n\n"
                "没有刷机，没有安装板载软件包，没有写入关节控制器，没有配置 systemd/开机自启，"
                "没有执行 StandUp、StandDown、SwitchJoystick、SwitchGait 或修改网卡地址。"
                "唯一真实控制阶段使用 Move 发送有界速度，并在退出时请求 StopMove。\n",
                "## 5. 本机运行产物和依赖\n\n"
                "本机已有宇树 SDK `/home/lenovo/unitree_sdk2`。已有 ROS Humble，但运行入口不依赖相机 ROS 话题。"
                "运行解释器为 `/home/lenovo/miniconda3/envs/isclab/bin/python`；训练环境 himloco 没有安装新包。"
                "缺失的 scikit-fmm 先安装至 `/tmp/himloco_real_deps`，再复制到 `deploy/ours/.deps`。"
                "后者依赖目录不加入 Git；其他依赖复用 isclab。\n\n"
                "`deploy/ours/build/robot_bridge` 为本机通信桥。早期还在 `/tmp` 编译过只读状态订阅和看门狗测试程序。"
                "SSH ControlMaster 套接字为 `/tmp/himloco-ours-ssh`，空闲连接会超时关闭。没有将登录密码写入仓库或这两份文档。\n",
                "## 6. 感知适配和当前参数\n\n"
                "真实柜门暴露了同格高度差规则的竖直平面漏检。实机前端保持原高度聚合，额外将离地 0.15–1.20m 且达到同格点数阈值的点补成障碍。"
                "这一实机几何前端适配必须在论文中披露，不能混同于未修改的仿真感知配置。\n\n"
                "FMM 栅格 0.05m，范围前方 6m、左右总宽 6m；硬半径 0.20m，软影响距离 0.65m，软权重 4，指数 2。"
                "标准 MPPI 默认 384 样本、20 步、步长 0.10s；足迹基础长 0.72m、宽 0.42m，基础裕量 0.05m，实机另加 0.04m。"
                "初步安装 X 取用户估计的 0.34m；Y=0 和前视安装航向为假设。地面拟合只约束高度和倾斜。\n\n"
                "本机标定文件的当前内容如下，`calibrated=false` 表示不是完整精确标定。"
                "只有显式 preliminary-trial 且距离/时长/速度满足初次试验范围时，才允许这一初步外参用于试验。\n\n"
                "```json\n"+json.dumps(cal, ensure_ascii=False, indent=2)+"\n```\n",
                "## 7. 拦截、停止和验证边界\n\n"
                "默认不创建速度客户端。前方中央有效深度不足、近障深度低于 0.45m、FMM 不可达、指令足迹碰撞或输入过期时输出零速度。"
                "独立桥检查指令年龄 300ms、状态年龄 250ms、倾角约 0.45rad、站立高度 0.20–0.60m；触发后停车退出。"
                "速度硬上限 0.20m/s、转速 0.35rad/s，入口默认上限 0.10m/s。首次 10s 试验没有出现阈值触发，用户确认最终停稳；"
                "尚未专门进行断网急停、遥控接管优先级及停止距离的实机验收。\n\n"
                "桌面记录中的脚本成功退出不等于导航成功。telemetry 的 sent=true 表示写入速度通信管道，不是机器人执行确认；"
                "StopMove 请求发出也不等于独立测量停止响应。图像和姿态尚未按曝光时刻做严格插值，当前适用范围为初步低速验证。\n",
                "## 8. 已保存的每轮运行\n\n"
                "每轮目录下包含 metadata.json、telemetry.jsonl、summary.json，部分保存 latest.npz 和 preview.png。"
                "初期失败目录也保留，不删除或伪装成成功。\n\n"
                "| 目录 | 模式 | 帧数 | 软件状态 | 平均/P95 ms | 写过速度管道 |\n| --- | --- | ---: | --- | --- | --- |\n"]
    def number(value):
        return "—" if value is None else f"{value:.2f}"
    for folder, summary, meta in runs:
        status = summary.get("status", "早期格式，见日志")
        details.append(f"| {link(folder)} | {meta.get('mode', '未知')} | {summary.get('frames', 0)} | {status} | "
                       f"{number(summary.get('plan_ms_mean'))}/{number(summary.get('plan_ms_p95'))} | {summary.get('sent_velocity_commands', False)} |\n")
    details += ["\n### 首次真实试验的结论\n\n"
                "前方目标 0.5m，速度上限 0.10m/s，最长 10s。147 帧，平均规划 64.7ms、P95 68.4ms；"
                "建议速度均值约 0.0579m/s，最大约 0.0687m/s。用户观察为身体前倾后收回，脚没有迈出，最终停稳。"
                "里程计约 0.068m 的净变化不得解释为真实走了这段距离。当前怀疑低速启动范围不匹配，尚未证明唯一原因。\n\n"
                "```json\n"+json.dumps(motion, ensure_ascii=False, indent=2)+"\n```\n",
                "## 9. 一键入口、具体命令和下一次操作\n\n"
                "根目录新建 `real_robot_deploy`，复用 `deploy/ours`；没有删除或搬走旧代码。\n\n"
                "```bash\ncd /home/lenovo/HIMLoco\n"
                "bash real_robot_deploy/one_click.sh check\n"
                "bash real_robot_deploy/one_click.sh prepare\n"
                "bash real_robot_deploy/one_click.sh shadow --duration 10\n"
                "bash real_robot_deploy/one_click.sh test\n"
                "bash real_robot_deploy/one_click.sh records\n```\n\n"
                "无参数时默认 prepare + 20s shadow，不发速度。下列命令是下一次运动测试入口，"
                "只有在现场输入 START 后才运动；是否已有后续运行请核对第 8 节目录和第 10 节历史：\n\n"
                "```bash\nbash real_robot_deploy/one_click.sh trial\n```\n\n"
                "它调用 run_ours.py 的 execute + preliminary-trial，目标前方 0.5m，"
                "min-vx=max-vx=0.10m/s，最多 5s。零速度拦截仍有效，通信桥仍渐变限幅。"
                "外参文件的初步标定状态不会被该命令修改成完整标定。\n",
                "## 10. 通过统一入口执行的操作历史\n\n"
                "motion_capable=true 表示该动作具有运动能力，不代表实际迈步，需核对现场确认和每轮 telemetry。\n\n"
                "| 北京时间 | 操作 | 退出码 | 操作可发速度 |\n| --- | --- | ---: | --- |\n"]
    for e in events:
        details.append(f"| {e['time']} | {e['action']} | {e['exit_code']} | {e['motion_capable']} |\n")
    details += ["\n## 11. 源文件清单与 SHA256\n\n用于确认记录对应哪些文件内容，不包含二进制、依赖、现场图像或密码。\n\n"
                "| 相对路径 | SHA256 |\n| --- | --- |\n"]
    for rel in descriptions:
        current = rel.replace("deploy/ours/", "go2_runtime/")
        current = current.replace("go2_runtime/build.sh", "deploy/build_runtime.sh").replace("go2_runtime/prepare_camera.sh", "deploy/prepare_camera.sh")
        path = ROOT/current
        if path.exists():
            details.append(f"| `{current}` | `{hashlib.sha256(path.read_bytes()).hexdigest()}` |\n")
    if wifi_checked:
        details.append("\n## 无线接入只读检查补充\n\n本次只检查本机网卡与扫描附近 Wi-Fi，并查阅官方文档和公开实现。"
                       "有线网卡 DOWN，当前旧 SSH 连接不存在，不能进一步检查盒子无线网卡。没有切换网络、修改路由或下发速度。"
                       "具体观察、接入步骤、当前约 184Mbit/s 原始图像负载、录包局限与后续方案见 "
                       +link(HERE/"WIFI_PLAN.md")+"。\n")
    if web_analyzed:
        details.append("\n## 网页实验台离线分析补充\n\n只读检查 `/home/lenovo/m20pro-ros2-real-main` 的静态网页、Python/ROS2 HTTP 后端、"
                       "command_mux、systemd、相机网关与录包脚本。未启动参考项目服务或连接 Go2。"
                       "当时仅形成板载感知/规划/速度桥/录包、浏览器监视和显式任务启动的方案，后续实施状态见下一节。详见 "
                       +link(HERE/"WEB_CONSOLE_FEASIBILITY.md")+"。\n")
    if web_built:
        details.append("\n## 已完成的本机离线网页\n\n新增 `real_robot_deploy/web_console/`：Python HTTP 后台、HTML/CSS/JS 页面、SQLite 本地记录、8 项离线测试和说明。"
                       "通过 `bash real_robot_deploy/one_click.sh web` 启动，访问 `http://localhost:8765`，仅绑定 127.0.0.1。"
                       "只读加载现有 11 组有遥测的真实历史实验，另行保存明确标记的流程模拟记录。"
                       "支持 RGB/深度/BEV 定格快照、FMM 路径/留存 MPPI 候选、软代价切换、遥测时间轴、实验搜索、ZIP 导出、参数保存、模拟启动/停止和采样记录开关。"
                       "模拟器没有运行 FMM/MPPI，不产生虚假的规划耗时或实机成功率。页面没有机器人 API，未加载 SSH、SDK2 或 ROS。"
                       "`data/console.sqlite3` 为新建本地数据文件，源实机日志未修改。浏览器检查中发现隐藏画布负尺寸循环，已补尺寸检查并重新验证页面切换。"
                       "8 项后台测试通过；已验证历史回放、模拟超时、模拟停止、记录持久化、页面切换及窄屏显示。"
                       "截图保存在 `real_robot_deploy/web_console/data/preview-desktop.png`。"
                       "浏览器导出验证生成 `/home/lenovo/下载/himloco_ours_ready_shadow_20261030.zip`，其中携带 source=historical 标记。"
                       "本轮没有连接机器狗、改变无线配置、部署狗端服务或发送速度。实机网页控制与自启动仍为后续阶段。\n")
    if light_layout:
        details.append("\n## 浅色首屏布局调整\n\n按用户要求修改 index.html、style.css 和画布颜色：RGB/深度移至局部感知与规划上方，"
                       "四块状态摘要缩小移入右侧配置下方，全页改为浅灰背景、白色面板和蓝色强调。"
                       "相机画面完整比例显示。浏览器检查 1280×720 时三块区域底部分别约 346、346、699 像素，均在首屏内；"
                       "1366×768 同样完整可见。未修改后台或机器人接口。截图位于 `real_robot_deploy/web_console/data/preview-light-firstscreen.png`。\n")
    if camera_verified:
        details.append("\n## 网页实时相机与正方形窗口\n\n修改 `web_console/static/index.html`、`style.css`、`app.js`：正方形窗口，object-fit: cover 居中裁切；桌面运行摘要放相机右侧。新增 `web_console/camera.py`，修改 `server.py`，通过固定 SSH 命令运行 `/tmp/himloco_ours_camera_stream`，解析 640×480 RGB8/Z16，生成 480×360 JPEG。新增 web-live 一键入口，只建立 SSH 与部署相机，不启动速度桥。主机和命令不能由网页请求修改。\n\n机器狗启动后确认有线网卡 192.168.123.200/24、盒子 192.168.123.18 可达，使用已知账户重新认证 SSH；密码不保存。运行 `deploy/ours/prepare_camera.sh` 上传源码并在盒子 `/tmp` 编译，相机流由网页按钮启动。当前页面实测约 15 fps 接收、约 5 Hz 预览更新，断开后恢复历史快照，重连恢复实时；超过 1.5 秒无新帧标记断流。最近接收时间不代表端到端延迟。13 项后台测试通过，包括协议解析、无效帧、断流、禁用入口与网页参数边界。1280×720 检查两帧窗口均约 192×192，局部规划区域底部约 712 px，首屏可见。\n\n本次仍无运动接口、无实机录制、无实时地图/MPPI、无自启动。现有原始 SSH 传输约 23 MB/s，无线性能尚未验证。本次检查期间前台服务收到退出信号，已确认旧相机进程退出，并以本机后台进程重新启动；PID 保存在 `web_console/data/server.pid`，日志在 `server.log`，没有安装系统服务。网页服务失联时图像标记定格。新增验证证据 `web_console/data/live-camera-verification.json` 与 `preview-square-live.png`。\n")
    if any(e["action"] == "web_console_fuller" for e in events):
        details.append("\n## 相机放大与状态区填充\n\n修改 `web_console/static/style.css`：相机行采用两个等宽画面加窄状态列，取消 210px 上限；右侧四张卡片纵向均分画面高度，统一相机标题栏高度；窄屏画面自动占满各自列宽。根据视觉列宽与视口高度调整地图高度，保留桌面首屏的局部规划区域。仅修改页面样式和说明，实时采集继续运行。截图：`web_console/data/preview-fuller-live.png`。\n")
    if any(e["action"] == "web_console_singlepage" for e in events):
        details.append("\n## 单屏前端与无线切换准备\n\n修改 index.html 删除最近实验表及入口，app.js 删除相关 DOM 访问并保留第二页列表与搜索；新增窄屏区域标签；style.css 使用固定视口、受约束网格、压缩顶栏与控制表单。首页不滚动，其他两页使用 main 独立滚动。浏览器验证 1280×720 首页 main 客户高度与内容高度均 682px，回放面板底部 712px；窄屏 425×612 main 两高度均574px，参数保存和三标签切换可用；第二页搜索未迈步试验并查看返回成功。实时相机持续接收约15fps，未新增运动或生成导航成功记录。截图 `web_console/data/preview-singlepage-live.png`。\n\n通过 SSH 只读执行 ip address / route、nmcli device status、iw dev：盒子 eth0=192.168.123.18/24，默认网关192.168.123.1，未发现可用无线接口。未改网络；Wi-Fi AP 到盒子路径与吞吐待验证。更新网页 README 与 WIFI_PLAN 的切换步骤与边界。\n")
    if board_deployed:
        details.append("\n## 2026-11-01 板载网页登录与自启动\n\n新增 web_console/access.py：PBKDF2加盐密码、8小时服务端会话、退出失效、登录尝试限制；登录文件权限600且位于忽略目录。新增login.html/css/js及退出按钮；server.py新增bind/auth/Host白名单/同源验证/本机接口IP识别，所有实验/图像/操作接口受登录保护。新增本地USB相机模式，板载相机不经SSH传原始图像；前端明确显示服务位置为机器狗算力盒子。新增test_access.py及本地相机分支测试，20项测试通过。\n\n新增 deploy_board.py、himloco-web.service、MOBILE_ACCESS.md、check_ap.py 与 web-lan/web-lan-live/web-deploy 一键入口。通过已知账户SSH上传到 /home/unitree/himloco-web，使用盒子已有numpy1.17.4/Pillow7.0.0/Python3.8.10/g++/librealsense，编译只读相机。sudo安装 /etc/systemd/system/himloco-web.service，enabled/active及服务重启通过；没有重启狗验证上电。服务以unitree身份运行，开启NoNewPrivileges/ProtectSystem和仅data可写等约束。网页登录初始信息保存在电脑 web_console/data/login-password.txt，不写进本记录或git；板载只复制密码校验文件。\n\n有线浏览器验证192.168.123.18:8765的登录、退出、窄屏UI、历史和相机，实测约15fps，物理指令计数0。AP测试使用用户指定go2热点，在电脑关闭有线网卡后明确验证192.168.123.18经192.168.12.1走wlp0s20f3，HTTP访问超时；192.168.12.1:8765拒绝连接。原电脑网络恢复成功，未修改狗端路由/防火墙/固件。测试中出现的同名电脑连接配置已整理，仅保留一个关闭自动连接的配置。已检查网卡驱动基础：内核5.10.104-tegra，nvidia-l4t-core35.3.1，匹配headers安装且build目录存在；没有USB无线网卡，未安装驱动。无线手机独立访问仍受网络路径/硬件限制，不能宣称完成。\n\n证据与使用/卸载步骤见 MOBILE_ACCESS.md，截图preview-board-mobile.png / preview-board-login.png，数据证据board-access-verification.json / ap-verification.json。源实机日志不变，板载复制11组历史，流程模拟不代表真实导航。\n")
    if any(e["action"] == "web_identity_updated" for e in events):
        details.append("\n## GO2页面名称与网页登录设置\n\n修改web_console/static/index.html和login.html的品牌、图标字母、标签页标题与面包屑。按用户指定值重新生成PBKDF2盐和校验摘要，原子写入电脑data/access.json并同步到盒子data/access.json，权限600；同步更新电脑私有login-password.txt，账号不变。不把密码写入本记录或源码。仅重启himloco-web.service使新校验生效，浏览器登录通过；未重启机器狗或修改SSH密码。截图web_console/data/preview-go2-brand.png。只读TCP检查192.168.123.18:22返回OpenSSH标识，192.168.123.161:22拒绝连接，192.168.123.1:22超时；未尝试登录运动控制器或修改其服务。\n")
    if any(e["action"] == "wifi_case_studies" for e in events):
        details.append("\n## 其他Go2无线方案调研\n\n新增WIFI_CASE_STUDIES.md，核查DroneBlocks AC1L盒子热点+网页方案、Qiita T2UB Nano作者安装记录、Robo Renta UGREEN AX900记录、TU Delft论文2.1节TL-WR802N路由器硬件，以及Beryl AX的SSH成功但DDS未通的排障案例。区分他人文档与本机验收、不把SSH通当作DDS/导航已通；未安装任何参考脚本。推荐先识别用户已有USB网卡芯片和AP能力再决定，不把同名型号直接视作本机兼容。\n")
    details += ["\n## 12. 参考来源\n\n"
                "宇树官方开发文档：https://support.unitree.com/home/zh/developer/\n\n"
                "宇树 SDK2：https://github.com/unitreerobotics/unitree_sdk2\n\n"
                "SportModeState 字段：https://github.com/unitreerobotics/unitree_ros2/blob/master/README.md\n\n"
                "RealSense 安装坐标模型：https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/urdf/_d435.urdf.xacro\n\n"
                "相似低速响应现象的实现者记录：https://github.com/Yaocheng-yan/Unitree-go2-Navi/blob/main/技术文档.md\n\n"
                "外部项目参数只作参考；真实相机尺寸、步态启动速度和定位精度以本机实测为准。\n"]
    if any(e['action'] == 'web_wired_real_connected' for e in events):
        simple += '\n最新状态（2026年11月1日）：按用户要求改为网线开发，电脑本机网页免登录，旧盒子网页服务已停止并取消自启动。网页现已接入真实相机、本体状态、双层代价规划、标定复核、受限执行入口、停止与实验导出。真实影子规划十五秒得到二百帧遥测和二十八份图像深度采样，没有向速度桥写入运动命令。另已验证自然结束和网页手动停止。相机地面高度和倾角已重新复核，前向三十四厘米及居中仍是估计，安装偏航待核对，完整标定未通过。本轮没有实际行走。\n'
        details.append('\n## 当前采用的网线实机网页（取代此前板载登录方案）\n\n电脑访问 http://localhost:8765/，仅监听127.0.0.1，无账号密码。盒子himloco-web.service已验证inactive/disabled，保留文件；不改无线配置。相机盒子采集后通过SSH传至电脑，本体状态经DDS到电脑，BEV/FMM/标准足迹MPPI均在电脑计算，显式执行才向本体下发速度。\n\n新增web_console/wired.py管理状态桥、共享相机、地面复核、固定里程计目标、规划与2Hz快照；camera.py加入时钟同步和数值深度共享；server.py加入实机接口与真实记录导出；前端加入模式、标定面板、真实影子任务、实际试验确认、状态/地图/遥测和目标点选。one_click.sh新增web-wired，start_wired_web.sh为一键入口。详见 '+link(HERE/'WIRED_WEB_GUIDE.md')+'。\n\n网页标定文件为web_console/data/calibration.web.json，不覆盖CLI原标定。最新地面高度约0.42349m、残差约0.002756m、内点约90.72%。X=0.34m/Y=0及偏航仍未实测确认，calibrated=false；不能把地面拟合当成完整外参标定。地面复核本次有效10分钟，服务重启需重新复核。\n\n真实记录deploy/ours/runs/web_shadow_20261101_163354_3d85ff：200帧，28份原始RGB-D采样，平均规划74.858ms/P95 95.432ms，速度桥写入0；原摘要按当时逻辑保存timeout，未改写旧记录。新版后续影子任务已验证shadow_complete与手动stopped。导出ZIP确有28份real/frames采样。27项后端测试通过，15项算法/感知测试此前通过；浏览器未见错误，1280×720首页main高度与scrollHeight均682px。\n\n执行入口限制前方0.3–0.5m、侧向0、上限0.10m/s、最长5秒；现场手动确认和初步标定确认不可省略。独立速度桥检查状态/指令过期，浏览器心跳过期和页面离开会终止自身任务。真实执行及遥控接管尚未现场验收；本轮未开启--execute。记录为2Hz快照而非完整视频/rosbag，所有规划帧遥测单独保留。速度桥写入不等于实际迈步，里程计到达需现场核验。\n')
    if any(e['action'] == 'web_capture_desktop' for e in events):
        simple += '\n新增网页“截图到桌面”按钮，单击保存同一实时帧的彩色图、彩色深度图和毫米制深度图到桌面GO2相机截图目录。已通过网页点击验证，文件实际保存成功，不发送运动指令。\n'
        details.append('\n## 单次实时截图到电脑桌面\n\n修改web_console/static/index.html增加截图到桌面按钮，app.js增加点击保存、处理中禁用和结果路径提示；server.py新增固定路径POST /api/camera/capture，读取同一最新帧，拒绝超过1秒的旧接收帧，不接受浏览器指定路径。每次使用时间与帧号独立建目录，保存640×480的rgb.png、depth_color.png、16位depth_mm.png（毫米，0无效）和带内参/帧号的metadata.json。先在临时目录写完再改名，避免出现半组截图。\n\n浏览器点击验证目录：/home/lenovo/桌面/GO2相机截图/20261101_164839_654096_frame_125，三张PNG均可读取且尺寸640×480，metadata存在。重启电脑网页服务后只读重连相机，未开启执行桥；本次重启会重置地面复核会话。证据web_console/data/preview-capture-button.png。\n')
    if any(e['action'] == 'web_video_quality_offline' for e in events):
        simple += '\n在机器狗充电期间，仅修改电脑：预览不再缩图，提高编码质量，画面刷新独立于状态轮询，目标每秒十五帧。用带离线标识的本机回放验证显示约十五帧；没有连接狗或部署新相机程序，实机效果待下次连接验证。另整理了EDU主机与设备资料，区分能SSH开发的盒子和通过DDS访问的本体。\n'
        details.append('\n## 本机高质量预览与EDU硬件调研\n\n修改camera.py：640×480原尺寸JPEG quality95、subsampling0，每帧编码；增加preview_pair锁内同帧读取。server.py增加/api/camera/preview，同一响应返回RGB/深度并以ETag去重。app.js独立无重叠15Hz画面循环，异步解码后一起提交，状态和地图仍按原周期；分别显示采集与浏览器提交fps。原桌面PNG截图路径不变。test_camera.py更新尺寸断言和原子图像对验证，test_console.py验证配对接口和304去重。28项测试通过。\n\n使用此前保存图像生成带OFFLINE REPLAY / NOT LIVE ROBOT标识的本地15Hz回放；浏览器显示15.0fps，两图640×480，无错误。编码50次平均7.10ms/P95 7.72ms，双JPEG约185KB。临时8766服务已停止；正式8765服务仅本地重启，不自动连接狗，未部署或改动盒子采集。明日实际帧率/延迟/规划并行负载待测，不能把离线结果写成实机验收。详见 '+link(HERE/'VIDEO_QUALITY.md')+'。\n\n调研官方Go2产品规格、SDK2、ROS2和相机接口，结合本机此前已验证的.18盒子SSH、.161接口可读但22拒绝、.1默认网关身份待确认，未将所有设备都当可登录主机。报告见 '+link(HERE/'GO2_EDU_HARDWARE.md')+'。\n')
    if any(e['action'] == 'native_ap_checked_20261102' for e in events):
        simple += '\n十一月二日重新开启机器狗后，已实际核查原生热点：电脑通过热点可以到本体控制端，但访问算力盒子仍超时。固定无线接口测试并同步检查盒子网口，没有看到连接请求到达；抓包正向校验通过。另外发现盒子默认网关无响应。只读取配置，没有修改狗端网络、启动相机规划或发送运动，电脑已恢复原Wi-Fi。\n'
        details.append('\n## 2026-11-02 原生热点实测\n\n通过有线SSH读取盒子接口、路由、邻居、策略路由、iptables与监听；临时加入已保存go2热点配置，使用SO_BINDTODEVICE和ping -I强制无线出口，保留网线用于观测，finally恢复<手机热点SSID>。热点IP192.168.12.32；网关.12.1与本体.123.161可ping，盒子.123.18的ping/22/80均失败。盒子eth0同步只读AF_PACKET报头捕获未见测试SYN；另用盒子到.161:22的SYN/RST验证捕获有效。盒子INPUT=ACCEPT、SSH监听0.0.0.0:22，不能归因于网页8765未启用。盒子默认网关.123.1邻居FAILED且ping无响应，回程亦待核对。未改狗端网络或服务、未启动运动。详细边界见 '+link(HERE/'NATIVE_AP_CHECK_20261102.md')+'；数据web_console/data/native-ap-check-20261102.json。\n')
    if any(e['action'] == 'preview_recovered_20261102' for e in events):
        simple += '\n十一月二日修复实时预览消失：发现SSH登录会话失效，重新认证后又发现狗重启清除了临时相机程序。已重新编译并增加持久保存位置，延长SSH空闲保留时间，恢复网页真实RGB与深度，采集和显示均约每秒十五帧。此次没有启动规划或运动。\n'
        details.append('\n## 2026-11-02 实时预览恢复与持久化\n\n诊断/api/state显示SSH认证失败，控制socket不存在；重新登录后显示/tmp/himloco_ours_camera_stream不存在。重新执行deploy/ours/prepare_camera.sh，并修改该脚本将编译产物原子复制到盒子/home/unitree/.local/lib/himloco/camera_stream，保留原/tmp副本供CLI使用。web_console/camera.py固定SSH采集命令改为持久路径，test_camera.py同步验证固定路径。one_click.sh的prepare_camera连接空闲保留由600秒改8小时，并设置15秒心跳、3次失败断开；未保存密码或添加授权密钥。\n\n重启电脑网页服务并通过页面只读连接。7项相机测试通过，shell语法检查通过；实机采集约14.99fps，浏览器采集15.0/显示15.1fps，两图640×480，速度桥写入0，任务为空、桥无--execute。截图web_console/data/preview-restored-20261102.png。更新WIRED_WEB_GUIDE.md说明狗重启/SSH会话失效仍需重新认证。没有重启狗或启动规划/运动。\n')
    if any(e['action'] == 'wifi_adapter_checked_20261102' for e in events):
        simple += '\n已检查新插入盒子的USB无线网卡：识别为RTL8188EUS，系统自带驱动已加载，能扫描二点四GHz网络，但当前驱动未向NetworkManager提供热点能力，也不支持五GHz。狗现有热点恰为五GHz，因此这张卡当前不能直接发热点或加入该热点。没有安装新驱动或修改网络。\n'
        details.append('\n## 2026-11-02 USB无线网卡只读检查\n\n盒子lsusb显示0bda:8179 Realtek RTL8188EUS，USB2、r8188eu内核驱动已绑定，wlan0被NetworkManager识别。WIFI-PROPERTIES.AP=no、2GHZ=yes、5GHZ=no，iw dev/list无wiphy，iwconfig显示Managed。内核CONFIG_88EU_AP_MODE=y不能单独证明NetworkManager可启动AP，应以当前驱动接口能力和实测为准。能扫描周边2.4GHz网络，尚未进行关联/传输测试。电脑扫描确认go2为5745MHz/149信道，<手机热点SSID>为5180MHz/36信道，这张卡均不支持。未创建连接、未安装/卸载驱动、未改路由/服务、未启动运动。数据web_console/data/wifi-adapter-check-20261102.json。可后续用用户提供的2.4GHz热点测试客户端链路，或另评估支持AP的驱动；不能将扫描成功写成无线网页验收通过。\n')
    if any(e['action'] == 'phone_wifi_connected_20261102' for e in events):
        simple += '\n手机热点切换到二点四GHz后，已按用户提供的信息将盒子连接到<手机热点SSID>，无线地址为10.118.86.232。电脑强制通过Wi-Fi访问盒子，四次ping全部成功，SSH正常响应。此网卡作为无线客户端已验证；不是发热点验收，也没有把网页和算法迁到盒子，网线仍保留。\n'
        details.append('\n## 2026-11-02 手机2.4GHz热点连接成功\n\n电脑与盒子均扫描确认<手机热点SSID>为2437MHz/6信道。用户明确提供连接凭据后，通过盒子NetworkManager创建himloco-phone-wifi配置，WPA2、2.4GHz、DHCP、自动重连，ipv4.never-default=yes，IPv6关闭；不更改eth0地址和原默认路由。密码仅由NetworkManager保存，不写入本记录。nmcli等待25秒先超时，但随后DHCP成功，不能据前者误报最终失败。最终wlan0为10.118.86.232/24、状态100，eth0仍192.168.123.18。\n\n电脑10.118.86.29使用SO_BINDTODEVICE固定wlp0s20f3连接10.118.86.232:22，获得OpenSSH8.2响应；ping -I wlp0s20f3四次全通，平均10.433ms。仅证实无线客户端IP/SSH通路，未验收持续图传、无线DDS或无线实走。未更换驱动、未迁移网页和算法、未启动运动。无线IP由手机DHCP分配，以后可能变化。数据web_console/data/phone-wifi-connected-20261102.json。\n')
    if any(e['action'] == 'wifi_ssh_verified_20261102' for e in events):
        simple += '\n用户无线SSH停在首次主机身份确认处后，已核对无线地址的ED25519公钥与本机原来信任的有线地址公钥一致，将无线地址以哈希形式加入known_hosts，并实际完成无线密码登录，返回用户名unitree、主机名ubuntu。没有关闭主机身份校验或保存登录密码。\n'
        details.append('\n## 2026-11-02 无线SSH身份核验与登录\n\n无线10.118.86.232的ED25519指纹与known_hosts内既有192.168.123.18身份一致：SHA256:qScQ/8sSNf0zkmODUt2eShGfBReCA8+jnLeadbSbaUA。比较完整公钥一致后，保留既有记录，仅追加无线IP的哈希主机记录到电脑~/.ssh/known_hosts。随后以ControlPath=none、StrictHostKeyChecking=yes连接无线IP，完成认证并执行hostname/id -un/ip -4 -br address show wlan0，返回ubuntu/unitree/10.118.86.232，退出0。未关闭身份检查、未删除旧key、未保存密码、未发送运动。原有有线复用会话当时已断开，本次成功登录未借用该会话。\n')
    if any(e['action'] == 'board_runtime_deployed' for e in events):
        simple += '\n最新状态（2026年11月2日）：已把完整网页后台、相机、本体状态和双层代价规划部署到算力盒子，并启用本项目网页服务自启动。电脑和盒子连接手机热点<手机热点SSID>后，访问http://10.118.86.232:8765/，无需登录，不再使用电脑localhost旧入口。实际断开电脑网线后访问网页、实时图像和状态通过，随后恢复网线。板载影子规划五秒四十五帧，未发送运动指令。截图按钮改为保存盒子并打包下载到当前设备。部署说明见项目BOARD_WEB_GUIDE.md；整狗重启、手机显示和实际行走尚未验收。\n'
        details.append('\n## 2026-11-02 完整算法与网页迁入盒子\n\n用户授权部署后，将应用、SDK2 ARM64库和隔离Python依赖安装到/home/unitree/himloco-runtime。新增deploy_board_web.sh、web_console/deploy_runtime.py、requirements.board.txt、go2-ours-web.service和BOARD_WEB_GUIDE.md；build.sh支持aarch64。server.py新增board/autoconnect入口，wired.py沿用本地状态与受限任务管理；camera.py本地USB采用同机单调时钟。app.js/index.html显示板载运行位置，截图保存盒子并经/api/camera/captures下载ZIP；电脑模式仍支持桌面保存。板载免登录为用户既有要求。\n\n依赖仅装入deploy/ours/.deps，未替换系统Python或厂商运控。主要版本numpy1.24.4、scipy1.10.1、scikit-image0.21.0、matplotlib3.7.5、Pillow10.4.0；scikit-fmm2023.4.2在盒子编译。首次影子因缺importlib-resources失败，保留web_shadow_20250929_175816_103dbc；补齐importlib-resources5.12.0/zipp3.20.2，并以python3 -S隔离系统matplotlib命名空间后算法导入成功。原生相机与SDK桥在盒子编译，内部DDS走eth0。\n\n安装/etc/systemd/system/go2-ours-web.service，unitree用户，enabled/active，绑定0.0.0.0:8765；自启动仅相机/只读状态，不创建运动任务。旧himloco-web.service停用且旧文件保留；电脑旧wired服务断开相机/状态并停止。服务重启连接通过，未重启整狗。PC和盒子各29项测试通过；截图ZIP含rgb.png、depth_color.png、depth_mm.png、metadata.json。成功影子web_shadow_20250929_180332_3a32eb共45帧/5.079秒，平均111.90ms、P95 128.23ms，快照丢弃0、速度桥写入0；狗趴下，unreachable不代表站立导航验收。无现场确认的执行请求被400拒绝，未生成运动任务。\n\n实际关闭电脑有线接口，固定wlp0s20f3经10.118.86.29访问盒子10.118.86.232：HTML、状态、RGB/深度预览通过，帧2873至2883持续更新，约15fps；已恢复电脑有线接口。证据web_console/data/board-wireless-verification-20261102.json和board-runtime-status-20261102.json。无线IP由DHCP分配可能变化。盒子当前公网默认路由未打通，依赖由电脑下载后离线传入，不影响局域网网页。热点密码、SSH密码不写入记录。\n\n盒子系统日期仍为2025-09-29，日志文件名受影响；实际操作日期2026-11-02，未更改系统时间，正式实验前应校时。本轮浏览器自动化超时，未完成页面点击/视觉验收、手机显示、整狗重启、无线实际行走。相机前向34cm、居中与偏航仍待实测，完整标定标志仍为false，站立后需重新地面复核。详见'+link(HERE/'BOARD_WEB_GUIDE.md')+'。\n')
    if any(e['action'] == 'outdoor_deploy_prepared_20261104' for e in events):
        simple += '\n2026年11月4日：按用户要求准备下一轮部署与室外测试流程。用户随后明确本轮只规划，部署留到下个任务。本轮未连接盒子、未部署或发送运动。新增只读网页检查脚本和室外测试流程文档，修正本机实机模式状态说明；网页29项、规划与拒绝逻辑15项测试通过。待无线网卡接好后，依次检查网络、更新时间与服务、站立标定复核、影子规划、现场确认后的短距测试。\n'
        details.append('\n## 2026-11-04 下一轮部署准备（尚未执行部署）\n\n用户明确“先规划，部署下个任务再执行”。新增real_robot_deploy/OUTDOOR_TEST_PLAN.md，明确网络身份/日期检查、更新板载服务、只读验收、静态地面复核、5秒影子规划、0.3至0.5米/最高0.10m每秒/最长5秒的首次响应测试、后续纸箱场景验证顺序。当前短测限制不等于已支持长距离连续避障。新增check_board_web.py，仅GET根页面与两次/api/state，检查板载模式、连续帧、输入新鲜、任务空闲及HTTP日期偏差，输出可选JSON；不运行标定、任务或运动。脚本通过本机HTTP替身验证，并未用于本次盒子验收。修改web_console/server.py的notice，根据实机/离线模式正确描述功能，避免已有运动接口仍称没有接口。\n\n本机web_console测试29项通过，deploy/ours/test_runtime.py在PYTHONPATH=deploy/ours/.deps下15项通过；首次直接运行后者因未带隔离依赖skfmm而导入失败，按现有run.sh同样环境重跑通过，未安装或更改系统依赖。部署脚本bash语法检查通过。未连接盒子、未更改网络或服务、未修改标定、未进行真实影子或行走，当前IP/新网卡型号能力仍待连接后确认。之前AI补洞图仅展示副本，不用于规划和定量评估。计划见'+link(HERE/'OUTDOOR_TEST_PLAN.md')+'。\n')
    if any(e['action'] == 'box_ap_configured_20261106' for e in events):
        simple += '\n2026年11月6日：新USB无线网卡已配置为盒子独立热点GO2-OURS，固定网页http://192.168.50.1:8765/。电脑或手机连接此热点即可访问，不再依赖手机分享网络。完成驱动编译安装、热点自动连接与开机加载配置；实际电脑断开网线后网页、RGB/深度约十五帧、本体状态均通过，未发送运动指令。电脑测试后恢复原网络，盒子热点继续运行。整狗重启、手机兼容和实走尚未验收。\n'
        details.append('\n## 2026-11-06 新网卡独立热点实施\n\n用户插入新网卡并要求继续无线工作；通过已信任有线SSH检查lsusb=0bda:c811，最初无无线接口。采用morrownr/8821cu-20210916源码，固定commit bda65aac150d2cde0df9603206eec23a6f3b77c4，设备表标该ID为8811CU。电脑下载源码传到/home/unitree/himloco-network/8821cu-bda65aac，在盒子5.10.104-tegra/aarch64及既有headers下make -j2，未运行上游root安装脚本。手工安装模块到/lib/modules/5.10.104-tegra/updates/himloco/8821cu.ko，SHA256=306bd8380cdf8007bfea3c0ba8672618173b3db9f3c8996fc98aa51b340ea827，depmod/modprobe成功，NetworkManager报告AP=yes及双频。新建/etc/modprobe.d/go2-8821cu.conf关闭驱动省电和USB自动挂起，/etc/modules-load.d/go2-8821cu.conf配置开机加载。未升级内核、未用DKMS，后续换内核需重编译。\n\n本机新增network/setup_box_ap.sh与network/AP_GUIDE.md。通过root脚本创建NetworkManager go2-ours-ap，UUID=1b8da5ca-1145-4ff2-8c4f-2948d7faf807，wlan0，SSID=GO2-OURS，2.4GHz频道6，WPA2/CCMP，自动连接yes/优先级100，IPv4 shared/192.168.50.1/24/never-default，IPv6关闭；dnsmasq分配192.168.50.10至254。密码仅输入并由NetworkManager保存，不写本记录。保留eth0及原厂go2热点、原手机连接配置；网页服务仍为既有go2-ours-web.service，未部署新算法。\n\n首次网页状态报JSON解析错误；独立只读桥采集有效，原桥仍存活而读取线程已退出，单独connect无效。无任务状态执行网页disconnect再connect恢复，只重建只读相机/状态，无execute。电脑实连AP获得192.168.50.123，断开电脑有线接口后仅无线接口路由可达，SO_BINDTODEVICE验证HTML200、配对预览、帧371至379、约14.99fps、本体输入新鲜、bridge_writes=0。证据web_console/data/ap-wireless-verification-20261106.json。随后恢复电脑XAUAT及原有线配置，盒子AP继续运行。电脑新建go2-ours-client保存连接并禁自动连接，防止抢占原网络；临时测试脚本清理。\n\nNetworkManager和网页服务enabled、热点配置autoconnect已核对，未整狗重启验收、未手机显示或远距/多客户端负载验收、未运动。盒子仍错误显示2025-09-29，实际操作2026-11-06；编译有时钟偏差警告，但最终模块加载和传输通过。尚未校时，不得将错误文件日期当真实实验日期。操作与恢复详见'+link(HERE/'network/AP_GUIDE.md')+'。\n')
    if any(e['action'] == 'ap_ssh_verified_20261106' for e in events):
        simple += '\n2026年11月6日SSH排查：有线地址192.168.123.18实际登录成功。电脑当时连校园网XAUAT，访问192.168.50.1走错网络；切到GO2-OURS后，无线SSH也实际登录成功。核对无线主机公钥与有线盒子一致后加入known_hosts，未关闭身份校验、未修改SSH密码。验证后恢复校园网，使用无线地址前仍需切到GO2-OURS；网线开发使用192.168.123.18。\n'
        details.append('\n## 2026-11-06 SSH双通道复核\n\n用户报告两种连接均无法SSH。检查电脑eth=192.168.123.200/24、Wi-Fi=XAUAT/10.59.166.180，访问192.168.50.1经校园网网关10.59.255.254，因此超时；有线192.168.123.18端口22/8765正常、ping两次全通，使用原密码实际完成SSH认证并返回ubuntu/unitree，ssh和网页服务均active。未据用户描述误判SSH服务损坏或重装。\n\n短暂切换电脑到已有go2-ours-client，获得192.168.50.123，路由直连wlp0s20f3。先从已信任有线SSH读取/etc/ssh/ssh_host_ed25519_key.pub，对比无线ssh-keyscan完整算法/公钥一致，再将无线地址的哈希记录加入电脑~/.ssh/known_hosts，不删除旧key、不关闭StrictHostKeyChecking、不保存SSH密码。新建独立SSH进程、ControlPath=none实际登录192.168.50.1，返回ubuntu/unitree且退出0；证据web_console/data/ap-ssh-verification-20261106.json。随后恢复校园网，有线仍保留。狗端配置未变更、未重启或发送运动。用户终端原始失败报错未提供，无法确定此前有线失败的具体输入原因。\n')
    if any(e['action'] == 'ros_cloud_viewed_20261106' for e in events):
        simple += '\n2026年11月6日ROS检查：通过有线SSH检查盒子ROS2 Foxy，保存约124条发现的话题清单。电脑ROS2 Humble已补齐独立目录的CycloneDDS通信依赖，实际接收狗自带雷达/utlidar/cloud约十五Hz点云，并启动RViz2配置。新增可直接执行的命令与说明。D435i仍是网页直接采集，没有新建ROS相机节点。未发送运动。\n'
        details.append('\n## 2026-11-06 盒子ROS与电脑点云查看\n\nSSH实际认证192.168.123.18；盒子安装Foxy/Noetic，显式source Foxy及cyclonedds_ws/install，使用已有eth0 CycloneDDS XML/domain0发现约124条话题，保存web_console/data/ros-box-topics-20261106.txt。/utlidar/cloud类型sensor_msgs/PointCloud2，有bare DDS发布者。盒子Foxy echo不支持--once/--field，最初尝试返回参数错误，未运行任何控制。电脑已装Humble/RViz2但缺少CycloneDDS，且sudo需密码；下载5个官方ROS apt运行包解到web_console/data/ros-pc-overlay/root，不写系统目录或.bashrc。新增network/ros2_pc_env.sh设置当前终端有线网卡、RMW、domain、环境；电脑话题发现通过并保存ros-pc-topics-20261106.txt。\n\n本机rclpy标准消息订阅十帧，15.04Hz短采样，frame_id=utlidar_lidar，首帧1122点、末帧1185点，字段x/y/z/intensity/ring/time；证据ros-cloud-sample-20261106.json。新增network/go2_lidar.rviz、view_lidar.sh，设置FixedFrame与点云话题及BestEffort/Volatile，实际启动本机RViz2，OpenGL4.6初始化正常；未自动截图视觉验收。首个后台启动未保持，改为PTY会话启动；未将仅启动命令误当持续运行。\n\n只验证有线DDS；无线网页SSH通不等于DDS跨网段，尚未配置无线点云中继。非标准unitree_go接口未在本机编译，列话题可见但echo对应消息仍需包。D435i既有USB采集不发布ROS话题，本轮未抢占设备或启动第二驱动、未修改盒子ROS环境/厂商进程、未发布运动。完整命令与来源见'+link(HERE/'network/ROS_TOPICS_GUIDE.md')+'。\n')
    if any(e['action'] == 'unified_web_url_20261109' for e in events):
        simple += '\n2026年11月9日：已给本电脑有线配置加入持久主机路由，有线和连接GO2-OURS的无线均可访问http://192.168.50.1:8765/。未修改狗端接口地址。其他电脑用网线仍需旧地址或同样路由，手机连热点可直接用固定地址。已分别测试网页与连续相机帧，并恢复电脑原校园网和有线连接。\n'
        details.append('\n## 2026-11-09 前端地址统一\n\n初始电脑校园网10.59.166.180、有线192.168.123.200，192.168.123.18:8765可达，192.168.50.1经校园网超时。给本机NetworkManager有线配置UUID=2e40e687-e023-3929-b89d-241c09da8653添加ipv4.routes主机路由192.168.50.1/32 via 192.168.123.18，并nmcli device reapply，无需修改盒子、IP、厂商服务或网页代码。经网线访问统一地址HTTP200及实时相机通过。随后电脑切到go2-ours-client并关闭有线接口，验证同地址经无线可达、画面帧增长；最终恢复校园网和原有线配置。证据web_console/data/unified-url-verification-20261109.json。未启动标定、任务或运动。本次未将本体状态异常视为导航已就绪，只验收统一地址访问。\n\n路由存入本电脑有线配置，插网线时安装、断开时撤销，非将两个网卡配到相同子网。其他有线电脑需另配路由或仍用192.168.123.18；手机连AP直接192.168.50.1。若关闭盒子AP撤销192.168.50.1，旧有线地址仍为备用。使用/撤销命令已写network/AP_GUIDE.md。\n')
    if any(e['action'] == 'web_cleanup_navigation_plan_20261109' for e in events):
        simple += '\n2026年11月9日：整理网页与部署工作，删除六份过时/重复指南和三个旧只读部署文件，重写现行使用与开发说明。完整部署统一到deploy_board_web.sh，默认地址改为固定热点地址；移除旧预览/登录局域网入口，默认一键动作改为仅本机检查，电脑备用相机入口增加板载服务占用检查。保留全部标定、实验原始数据和操作历史。本轮仅本机整理，未部署、连接或控制机器狗。后续按“状态/校时/标定→影子→首次迈步→停止接管→连续模式→纸箱导航”逐级验收。\n'
        details.append('\n## 2026-11-09 网页收尾与导航测试计划\n\n统一real_robot_deploy/README.md为现行入口，重写BOARD_WEB_GUIDE.md及web_console/README.md，更新OUTDOOR_TEST_PLAN.md。删除MOBILE_ACCESS.md、NATIVE_AP_CHECK_20261102.md、WEB_CONSOLE_FEASIBILITY.md、WIFI_PLAN.md、WIFI_CASE_STUDIES.md、WIRED_WEB_GUIDE.md，历史过程仍保存在操作记录；历史链接重定向至现行指南。删除web_console/deploy_board.py、himloco-web.service、check_ap.py旧只读部署/AP探查代码。one_click.sh删除web-live/web-lan/web-lan-live入口，web-deploy改为当前完整部署器别名；无参数默认check，不自动连接。备用相机准备前检查当前go2-ours-web.service及旧服务占用。deploy_board_web.sh与deploy_runtime.py默认主机改192.168.50.1，支持GO2_HOST覆盖。access.py仍被server和测试引用，保留；网页静态布局、算法、标定、data、runs、原始图像均未删改。\n\n计划明确现有执行仍硬限0.3至0.5m/0侧移/0.10m每秒/5秒；尚未验收迈步、停止和接管，不能直接做长程绕障。列P0状态线程恢复、时间、实测外参、未知区域策略与连续目标模式；设计空旷响应、停止/失联、连续闭环、纸箱场景重复与论文指标。三次响应基线和每场景至少五次正式重复是待执行建议，不是已做实验。本轮不连接盒子、不更新服务、不触发任务。验证结果以本次最终报告为准。\n')
    if any(e['action'] == 'static_preflight_ground_20261109' for e in events):
        simple += '\n2026年11月9日第一步静态检查：盒子时间原为2025年，已在无任务状态按电脑校正为2026年11月9日，网络授时尚未同步。网页本体状态有旧解析错误，经只读断开/重连恢复；机身高度约30.8cm，状态与相机持续更新。备份原标定后完成三帧地面拟合，相机离地约42.2cm、地面占比约80.9%、残差约2.7mm。前向34cm、居中和偏航仍是估计，未批准完整标定。保存原始截图和检查证据，无影子任务、无运动。画面左前桶及地面翘边需在行走前清理。\n'
        details.append('\n## 2026-11-09 第一步静态基础与地面复核\n\n用户提供官方安装外观图并说明位置相似但尺寸不准确；仅作为安装参考，不从照片认定外参真值。盒子初次HTTP状态connected=false，旧JSON解析错误、任务null、桥写入0。通过已信任有线SSH认证；date显示2025-09-29，NTP开启但NTPSynchronized=no，RTC为1970年。确认任务空闲后使用sudo date按电脑当前Unix时间校正为2026-11-09，保留NTP配置，未写RTC或设置持久授时；HTTP时间与电脑误差约1.77秒，重启后仍需核对。\n\n执行/api/real/disconnect和connect，仅重建只读相机/状态，随后15次约0.3秒间隔采样并保存stationary-states-20261109.json。本次未修复读线程遇异常退出的源码缺陷，不宣称已完成长期稳定性验收。原标定备份为盒子data/calibration.before-ground-20261109.json；POST /api/real/calibration/fit，按现有站立/静止/新鲜度门限检查后采三帧，完成高度与倾角估计。standing_base_height_m=0.307993，相机离地0.42190494m、保留X=0.34/Y=0估计后相机相对机身Z=0.12328586m，最低地面内点率0.80888889，最大RMS=0.00269572m。calibrated=false、xy_measured=false、xy_physical_confirmed=false、mounting_yaw_confirmed=false，未调用approve；保留4cm额外足迹裕量。地面复核会话10分钟有效。\n\n板载截图20261109_130819_369637_frame_402，下载四文件到本机data/preflight-capture-20261109，实际RGB检查可见左前桶与地面接缝翘边，本次不认定行走区域合格。同步板载标定、旧备份及ground-verification.npz到本机带20261109后缀的证据副本，未覆盖电脑既有默认标定。check_board_web.py七项通过：HTTP/板载模式/连续相机帧/本体新鲜/实机空闲/模拟空闲/时间误差，约14.99fps。证据ground-result-20261109.json、preflight-20261109.json、step1-summary-20261109.json；状态采样与地面原始数据保留。未启动规划或实际执行，bridge_writes=0，任务null。本步骤通过静态地面与连接复核，水平外参/偏航仍待测量，不能宣称完整标定或可行走。\n')
    if any(e['action'] == 'standing_shadow_20261109' for e in events):
        simple += '\n2026年11月9日静态影子规划：站立状态运行前方0.5米、侧向0、5秒ours，保存RGB-D。42帧完成，平均119.7ms、P95 167.1ms，桥写入0，无运动。38帧不可达、2帧输入过期、2帧正常；局部图左前桶附近障碍膨胀覆盖目标，离线复算确认，不能据任务完成就开始走。导出已保存，下一轮须清理桶及地面后同参数复测，不改小安全裕量。\n'
        details.append('\n## 2026-11-09 站立静态影子试验\n\n用户同意下一步静态影子规划，未授权实际行走。确认相机/本体输入新鲜、z约0.308m、本次地面已复核、任务空闲后POST /api/real/start，kind=shadow，forward=0.5/lateral=0/max_vx=0.10/duration=5，record_rgbd=true。任务web_shadow_20261109_131042_ec0b47在盒子运行，5.079秒/42帧，shadow_complete，bridge_writes=0，sent_velocity_commands=false，snapshot_drops=0，保存9份原始RGB-D采样。规划平均119.666ms/P95 167.137ms；原因unreachable38、ok2、stale_input2。完成的是采集规划流程，不是目标到达或导航验收。\n\n末帧BEV障碍14格，前向约0.6至0.8m/左向约0.1至0.3m；与既有画面左前桶位置相符，仍不能将估计外参投影距离称测量真值。以原始末帧occ及相同FMM配置离线复算：goal_rc=[10,60]，原始目标单元占用0、0.20m硬膨胀后占用1、reachable=false，故当前路径仅起点。没有缩小硬膨胀、修改足迹或绕过停车保护。短时2帧输入超时触发零输出，运动中时效表现尚未验收。建议现场移开桶并整理地面后，同一目标重新做影子，条件满足后再单独确认实际行走。\n\n首次直接GET实验导出因历史索引未刷新404，随后GET /api/experiments刷新再导出成功，不重启服务、不重跑或改写日志。完整ZIP约5.90MB保存web_console/data/web_shadow_20261109_131042_ec0b47.zip；状态shadow-status-20261109.json、summary/metadata/末帧NPZ和shadow-obstacle-diagnosis-20261109.json保留。任务均在板载deploy/ours/runs中保留。未部署代码、未校准成完整外参、未发送任何实际运动。\n')
    if any(e['action'] == 'shadow_bin_removed_retest_20261109' for e in events):
        simple += '\n2026年11月9日移桶复测：用户移开垃圾桶后，先因地面复核超过10分钟重新拟合，再用相同0.5米/侧向0/5秒参数运行真实影子。29帧完成，路径可达、未再出现不可达；19帧正常、10帧输入过期归零。平均规划175.4ms/P95 190.2ms，运动指令0，保存10份RGB-D。场景阻挡已解除但输入时效尚未通过，下一步应定位耗时，不直接实走或放宽超时。\n'
        details.append('\n## 2026-11-09 垃圾桶移除后的同参数影子复测\n\n用户明确移桶并要求复测，不要求再次处理地面翘边，未授权实际运动。开始前读取状态新鲜、站立z约0.308m、无活动任务；地面会话有效期已过，初始本机检查拦截，未提交任务。随后按原规则POST calibration/fit，通过三帧静态复核，相机离地0.42140205m、地面内点率0.98696130、RMS0.00232041m，相对机身Z0.12309214m；保留X0.34/Y0/偏航估计和calibrated=false。旧值保存ground-before-retest-20261109.json，新值ground-retest-result-20261109.json。\n\nPOST真实shadow，参数forward0.5/lateral0/max_vx0.10/duration5，record_rgbd=true，任务web_shadow_20261109_131844_fe88ca。5.114秒29帧，桥写入0、无速度指令、采样丢弃0，保存10份RGB-D。末帧障碍0格、FMM路径11点；reason=ok19/stale_input10，无unreachable，确认旧桶阻挡已解除。建议速度0至0.061625m/s（影子采样min_vx=0，并非实际命令）。计算平均175.432ms/P95 190.207ms；采集年龄上界174.275至256.672ms，输入过期由相机或本体旧样本在规划后超过约250ms产生，尚未分解全部时延来源。空旷可达后完整MPPI计算路径比之前不可达时更重，不能只据首次影子较低耗时判通过。\n\n结论：路径层通过本轮静态空旷检查，时效10/29帧归零未达到稳定行走准备要求。未改速度/超时/足迹/安全裕量，未运行execute。导出约5.998MB到web_console/data/web_shadow_20261109_131844_fe88ca.zip；状态、末帧NPZ、summary、metadata和shadow-retest-analysis-20261109.json保留。建议先定位规划/取帧/状态年龄耗时，保持安全门限再影子复验，而非直接实走。\n')
    if any(e['action'] == 'planning_latency_optimized_20261109' for e in events):
        simple += '\n2026年11月9日时延优化：定位到BEV重复栅格累积和本体状态约10Hz输出造成额外等待，新增等价原生CPU栅格统计，状态输出改50Hz，并在计算前等待更新的数据。保留250ms过期保护、全部深度点、MPPI采样数和足迹参数。已备份并部署到盒子；最终两轮5秒静态影子共108帧全部正常、0过期、0运动指令，平均91.2/92.9ms，P95 108.3/109.6ms。48项本机/相应盒子测试及原始19帧栅格一致性检查通过。尚未行走，后续需现场确认短距响应和停止。\n'
        details.append('\n## 2026-11-09 定位与优化规划/数据时延\n\n盒子在原保存帧cProfile：BEV每轮约83ms，32次ufunc.at/8轮累计465ms（每轮约58ms）；MPPI约36ms。新增deploy/ours/bev_reduce.cpp及run_ours.py ctypes接口，单循环计算全部有效投影点的min_z/max_z/count/离地障碍点数，无库保留NumPy参考；build.sh生成libbev_reduce.so与状态桥并原子替换，不改数据分辨率或采样。19份前后场景RGB-D原始快照回放四组栅格逐项一致，另测试20万重复单元及高度阈值边界/空输入。相同盒子20次离线回放NumPy133.91ms至native76.93ms，BEV84.58至27.38ms，MPPI保持约35ms。\n\nrobot_bridge.cpp由每5个20ms循环输出一次改为每循环输出，实测51条/约49.67Hz，保留真实DDS state_age，过滤所有输出浮点非有限值以避免启动非法JSON。SDK低层与原硬件看门狗阈值未改。wired.py connect检测bridge_error后允许手动重建只读桥。select_command可选diagnostics统计BEV/FMM/MPPI/命令足迹检查耗时、原帧及原状态开始/结束年龄、过期来源；row写planning_timing，summary写分阶段均值和原因。未用刷新时间戳伪装新数据。\n\n先在独立/home/unitree/himloco-latency-20261109编译/离线测试，运行时更新前确认任务空闲。原版备份/home/unitree/himloco-network/runtime-before-latency-20261109_132819.tgz，包含旧源码/二进制与标定；更新/home/unitree/himloco-runtime中的run_ours、wired、桥、原生库及相应构建/测试文件，重启本项目服务，仅恢复只读输入。首次优化影子web_shadow_20261109_132906_61539b共53帧，平均93.22/P95 110.66ms，52正常、首帧1次相机过期（开始年龄149.84ms，结束265.94ms），此失败记录保留。\n\n随后增加planning_inputs_ready：仅当相机和状态开始年龄均不超过100ms才开始规划，旧输入等待下一份（input_waits为等待检查次数，非丢帧数）；开始前250ms输入检查、计算后原数据250ms检查、执行前检查/心跳均保留。未修改MPPI384采样/20步、硬膨胀0.20m、足迹与估计标定裕量。服务再次重启后按静止门限重新地面复核，完整标定仍false。最终影子web_shadow_20261109_133111_c62bd4和web_shadow_20261109_133125_a160fd，各54帧，全ok、无过期、桥写入0、各保存9份RGB-D无队列丢弃；平均91.176/92.877ms，P95 108.314/109.589ms。最大相机年龄176.36/181.68ms，最大状态年龄131.00/131.57ms；input_waits6/0。本轮全部任务均shadow，未调用execute或发送运动。\n\n本机17运行+31网页测试、盒子17运行+31网页测试通过，独立C++看门狗通过；新增慢计算仍归零、旧输入等待、坏读线程手动重连测试。最终源码与板载逐字节核对，manifest见web_console/data/latency-deployed-manifest-20261109.json，结果latency-final-results-20261109.json、原始每轮ZIP与剖析文件保留。末次相机/本体新鲜、服务enabled/active，进程仅robot_bridge eth0无execute。此为当前静态短时验收，未验收运动中与复杂障碍长期性能。完整报告及回滚说明见'+link(HERE/'LATENCY_OPTIMIZATION.md')+'；测试计划已更新，下一步仍须现场确认短距响应。\n')
    if any(e['action'] == 'labels_and_short_walk_20261109' for e in events):
        simple += '\n2026年11月9日：按用户要求删除页面全部“第三章”字样，盒子页面已更新为GO2/工作台及ours配置，无需重启服务。用户现场确认站稳、前方空旷、持遥控器后执行一次前向0.5米目标、最高0.10m每秒、最长5秒实际测试；结束后已停止。46帧全部规划正常、桥写入46次，但用户确认仍只前倾晃动、没有迈步、已停稳，不能记为导航成功。随后仅查询运控服务，未加速、切模式或重试。\n'
        details.append('\n## 2026-11-09 页面去章节文字与一次短距实际测试\n\n修改static/index.html三处：meta描述、面包屑删除第三章实验、配置标题改ours配置。原子更新盒子同路径并备份index.before-label-cleanup-20261109.html，未重启服务或清除地面会话；HTTP获取HTML确认无第三章，实验列表亦无该词。用户明确现场正常站立、前方约3米空旷、持遥控器可随时接管后，只执行一次/api/real/start：kind=execute，forward0.5/lateral0/max_vx0.10/duration5，operator_confirmed=true/preliminary_ack=true/record_rgbd=true。相机X/Y/偏航仍估计，不标完整标定。\n\n任务web_execute_20261109_133758_2d51c3，客户端保持约0.1秒间隔心跳并监视状态，5.040秒到时结束，随后显式POST stop并确认恢复只读桥。46帧、46次Python到桥写入（不是46次SDK调用或迈步次数），全部reason=ok，无输入等待/过期拦截，采样无丢弃。速度建议各帧约0.10m每秒、wz=0，保留桥渐变限幅；平均规划108.387/P95 123.889ms，最大相机年龄196.683ms。bridge.stderr未见Move failed，未将无错误日志当作足部行走证据。mode/gait_type全为0，仅记录字段，不按名称推断唯一故障原因。狗回报vx峰值0.09008m每秒、里程计首末前向约0.0845m，但用户现场明确仍只前倾/晃动、未迈步、停稳；因此physical_result=no_steps、success=false，里程计不作行走真值。\n\n新增只读query_motion_state.cpp，使用CheckMode/ServiceList，不调用SelectMode/ReleaseMode/ServiceSwitch或运动。盒子编译到/home/unitree/himloco-network/query_motion_state；查询返回CheckMode result0/form0/name=mcf，ServiceList返回0，sport_mode/advanced_sport/ai_sport各status1。未修改任何服务/运控模式，尚未确认mcf下具体起步条件，也未据官方示例0.3m每秒认定最低速度。没有自动再试、加速或切步态。\n\n导出web_console/data/web_execute_20261109_133758_2d51c3.zip；walk-before/status/monitor/summary-20261109.json保存输入、结果、客户端监测。现场结果保存在walk-motion-analysis-20261109.json及盒子该轮motion_analysis.json，原始telemetry/summary未改写；只读模式查询另存walk-readonly-mode-query-20261109.txt。此次操作历史通过--motion-capable明确记为具备运动能力。下一步需要结合遥控正常行走经验、SDK实际输出/返回与起步条件诊断，不能据本轮结论解锁连续导航。\n')
    if any(e['action'] == 'speedcal_030_20261109' for e in events):
        simple += '\n2026年11月9日：用户明确要求提高到0.30m每秒后，新增独立的三秒直行速度标定入口，保留普通导航限速及全部停止保护；更新盒子、重新地面复核后仅执行一次。用户确认已真正迈步前进、已停稳且未接管。本轮31帧无过期拦截，实际桥输出达到0.30m每秒，里程计前向变化约0.35米；仅认定起步响应通过，不当作完整导航成功或最低速度测定。原始数据、现场结果和回滚备份已保存，没有自动重试。\n'
        details.append('\n## 2026-11-09 0.30m/s单次起步响应标定\n\n用户明确“直接提高到0.3m/s，然后测试”，沿用此前现场站稳、前方空旷、持遥控器可接管的确认，不再次索要相同授权。新增独立speed_calibration布尔任务字段；server.parameters默认仍最高0.20，只有实机标定分支允许0.30。wired.validate_start要求execute/现场确认/本次地面/初步标定ACK，限制前向0.3至0.5m、侧向0、速度不超过0.30、最长3秒；普通实际任务仍0.10m/s/5秒，网页普通表单未放宽。标定MPPI保留ours标准足迹及384采样/20步，非停车速度0.30、角速度0，近障/目标/过期停车仍可归零。\n\nrobot_bridge新增--execute-calibration，普通--execute仍0.20桥限速；专用模式独立限0.30直行、首个输入起3秒停止，保留原渐变0.005每20ms、状态/姿态/命令超时看门狗。增加状态字段bridge_vx/bridge_wz/sdk_move_calls/sdk_move_result，区分Python写桥与真实SDK输出。BridgeSafety测试验证普通模式拒绝0.30、标定接受0.30且拒绝更大速度/非零转向、过期仍锁存；新增Python测试确认模式/时长/目标/速度/确认不可绕过。32项本机与盒子网页测试、17项本机运行测试和独立C++测试通过。\n\n先在/home/unitree/himloco-speedcal-20261109编译，确认无任务后备份/home/unitree/himloco-network/runtime-before-speedcal-20261109_135034.tgz，更新bridge源码/头/二进制及server/wired/相关测试，重启本项目服务，未改厂家模式或步态。重新完成静态地面复核（speedcal-ground-20261109.json），按一次性API启动web_execute_20261109_135125_e27c9f，目标0.5m/侧向0/最高0.30/3秒，speed_calibration/operator_confirmed/preliminary_ack/record_rgbd均true。客户端维持心跳并监控，结束后显式stop，未重试。\n\n任务3.102秒、31帧、桥写入31（不是SDK次数），23帧ok/8帧goal_confirmation、无过期或采样丢弃。实际渐变桥速度最大0.30，已记录SDK返回码均0，状态样本内SDK调用累计最大140（不是完整总数），本体回报前向峰值0.342723m/s；指令上限不等于实际反馈绝无瞬时超调。平均规划98.801ms/P95 119.647ms。用户明确“已真正迈步前进，已停稳，未接管”。里程计首末前向0.34947m/侧向0.02994m仅为估计，不是外部测量；没有完成连续10帧到达确认，软件结果仍timeout，不改写成目标到达。物理结果stepped_and_stopped，起步响应通过，完整导航success仍false。mode/gait_type仍0，不能用这两字段未变化直接判断是否迈步。\n\n导出web_console/data/web_execute_20261109_135125_e27c9f.zip，保存speedcal-before/ground/status/monitor/summary/motion-analysis-20261109.json；用户观察另存盒子该轮motion_analysis.json，原始遥测/summary不篡改。测试后只读桥恢复、用户确认停稳，没有自动再试/切模式/提高普通导航限速。测试计划与板载使用说明已更新：当前0.30能起步，最低起步速度仍未知；下一步验证停止距离、遥控接管和障碍停车后再决定正常导航速度。\n')
    if any(e['action'] == 'gait_note_web_stop_20261109' for e in events):
        observed=read_json(ROOT/'go2_web/data/web-stop-analysis-20261109.json',{}).get('operator_observation','待用户现场确认')
        simple += '\n2026年11月9日：按用户要求记录不同模式/步态训练速度范围差异及0.10m每秒下不迈步的现场解释，本轮不深入分析或切换步态。此前0.30m每秒真正迈步已获用户确认，首次起步响应通过。接着执行一次网页主动停止测试，在约1.59秒请求停止，任务记录手动停止，后续本体平移速度近零；停止监测存在查询超时，无法据此计算精确停止延迟/距离。现场结果：'+observed+'。未继续接管或近障测试。\n'
        details.append('\n## 2026-11-09 步态速度问题记录与下一步网页停止测试\n\n用户明确：该步态训练时0.10m/s不走，切换导航模式后步态不同，不同步态训练随机速度采样范围不同；按用户要求仅记录，不深查训练或更改厂家模式。独立资料web_console/data/gait-speed-observation-20261109.json标注来源为用户现场说明、具体训练范围未独立核验。首次0.30m/s物理迈步/停稳/未接管已确认，因此起步响应通过，不等于完整导航或重复性验收通过。用户明确通过后进行下一步，沿用现场空旷与遥控接管条件；本轮仅做一次网页停止，不自动开展遥控接管/失联/近障测试。\n\n检查实时RGB与输入，见stop-test-before-20261109.jpg，狗当前站立静止；地面复核已过期，重新完成静态拟合并保存stop-test-ground-20261109.json。未改变步态/训练/限速/安全门限。启动web_execute_20261109_140349_dd09e3，速度标定模式、最高0.30m/s、目标0.5m、3秒硬限，客户端保持心跳。约1.59254秒主动POST stop，约1.74831秒收到成功响应，任务结果stopped/手动停止，末轮遥测elapsed1.3197s/11帧/11次桥写入，全ok、无过期/采样队列丢弃，平均118.705ms/P95 131.876ms。停止前记录桥输出峰值0.265m/s、本体前向速度峰值0.266596m/s，仍在渐变，不能称满0.30m/s停车已验收。\n\n停止后监测取到两次状态后发生TimeoutError，finally再次stop/查询也有超时；保留完整错误，不把HTTP约156ms响应当物理停止延迟。随后8次只读采样任务仍stopped、平移速度接近零，未再启动运动。现场停稳/接管结果：'+observed+'。web-stop-monitor/status/followup/summary/analysis-20261109.json和完整ZIP保留，分析写入盒子该轮motion_analysis.json；原始telemetry/summary不改写。精确停止延迟/额外距离因观测缺口未计算，需另验。操作历史带--motion-capable，本轮没有自动重试或开始接管测试。\n')
    if any(e['action'] == 'web_remote_takeover_deployed_20261109' for e in events):
        simple += '\n2026年11月9日：按用户要求将0.30m每秒手柄接管测试接入网页，默认前向2米、最长8秒，可选4至8秒。手柄虽通过本体遥控链路工作，盒子可读取本体DDS转发的数据；本机独立手柄话题无实际输出，已按官方布局接入LowState内40字节遥控数据。新增摇杆/按键触发锁存退出、数据通道超时及桥端时限/距离保护，普通导航限制保留。网页启动须实际操作核对后归中、地面复核和现场确认；当前只观察到归中，按钮仍拦截。本轮仅部署和只读验收，未启动行走；真实接管待现场验证。\n'
        details.append('\n## 2026-11-09 网页0.30m/s长时接管入口与官方手柄读取\n\n用户要求网页接入0.30并延长时长，说明使用宇树官方实体手柄。本轮明确只部署，不自动执行运动；用户确认手柄已开机连接、摇杆归中。新增独立takeover_test模式，默认前2m/侧0/0.30/8秒、网页可选4至8秒，防止旧0.5m目标过早结束。后端限前0.5至2m/侧0/最高0.30/最长8秒，要求operator_confirmed、初步标定ACK（如需）、本次地面、remote_verified、近期真实手柄输入及归中；不能与speed_calibration混用，普通0.10/5秒与速度标定0.30/3秒保留。\n\nstatic新增实际测试/手柄接管按钮与模式弹窗、时限、遥控数据/验证/归中状态和确认项；前方至少4米空旷。本轮源码/返回HTML无第三章字样。未勾确认或数据未验证启动禁用；前端不发送新flag时普通模式限制不变。JS经语法及离线DOM替身互锁测试，HTML全部引用ID存在。style允许确认弹窗内部滚动，首页布局不重做。\n\n官方SDK示例rt/wirelesscontroller读取无消息，进一步只读三源探测3秒：processed0/unprocessed0/lowstate1495。按官方state_machine/gamepad.hpp的40字节wireless_remote布局读取keys偏移2，lx/rx/ry/ly偏移4/8/12/20；本机头0x55/0x51，未知头/非有限或超界轴拒绝。最初探测工具atomic输出编译歧义改用load后成功；未运行控制示例。最终原生桥只读98帧遥控数据valid/ready，轴0、keys0、sdk_move_calls0。手柄作用在本体不妨碍算力盒子订阅DDS，无需SSH运控主机。未检查或改变步态训练配置。\n\n新增remote_takeover.hpp状态机及test_remote_takeover.cpp。新桥flag --execute-takeover独立最高0.30/角速度0/8秒/里程位移2m；启动要求连续归中0.5秒。摇杆任轴幅度>0.20或任意非零按键锁存退出，归中不会解锁；数据包500ms不更新或非法也锁存。记录真实输入活动时间供Python要求2分钟内出现操作；新执行桥重新等待归中，不依赖网页及时响应。桥事件被run_ours.read_state识别为错误通道，wired分类taken_over/stopped，关闭执行桥后仅恢复只读；不调用模式/步态切换。LowState可能重复缓存无线数据，不能把通道新鲜视为物理RF在线或保证手柄断电会500ms停止，文档/弹窗明确要求保持手柄连接与现场操作验证。\n\n本机34网页+18运行测试、盒子34+18通过，C++门限/中位/按钮/摇杆/丢包/非法包/锁存及Raw布局测试通过；盒子运行测试初次因缺calibration.example.json fixture失败，补齐示例后18通过，未改生产标定。未注入真实控制话题、未启动任何execute。备用one_click test增加纯C++接管测试，完整部署器增加新头文件。备份/home/unitree/himloco-network/runtime-before-takeover-20261109_143048.tgz，更新/home/unitree/himloco-runtime源码/桥二进制/网页后重启本项目服务；最终字节核对manifest保留。只读API显示遥控通道新鲜归中，但input_verified=false；远程未核对请求HTTP400，task=null、SDK调用0。\n\n浏览器CUA创建页面超时，未取得截图或视觉验收；没有把HTTP/JS替身测试当真实浏览器点击验收。证据takeover-deploy-verification-20261109.json、takeover-deployed-manifest-20261109.json。本轮未替用户勾确认、未复核成完整标定、未启动长测试，真实遥控动作响应与运动中接管仍待用户现场验证。操作说明见'+link(HERE/'network/REMOTE_TAKEOVER.md')+'，README/BOARD_WEB_GUIDE/测试计划已更新。\n')
    if any(e['action'] == 'takeover_user_verified_20261109' for e in events):
        simple += '\n2026年11月9日：用户在网页自行完成官方手柄接管试验，反馈现场结果与操作步骤一致。核对盒子原始日志，本轮前向目标2米、速度上限0.30米每秒、最长4秒，在约2.64秒被手柄触发退出，21帧规划均正常，无过期拦截；后台恢复只读。本轮接管功能通过，不代表完整导航通过，也未测定精确停止时间和距离。已保存电脑副本，补充日志位置说明，并清除网页左下角残留CHAPTER 03。本次仅核对与整理，没有另外启动运动。\n'
        details.append('\n## 2026-11-09 用户实测手柄接管结果与日志位置\n\n用户提供14:46网页截图并说明完全按步骤操作、结果一致。读取盒子API与原始实验目录/home/unitree/himloco-runtime/deploy/ours/runs/web_execute_20261109_144517_b8dfc2/，核实takeover_test=true、forward=2、lateral=0、max_vx=0.30、duration=4；截图普通表单0.5/0.1/5是草稿，不是本次弹窗实际参数。结果taken_over，标签手柄触发接管程序已退出，末帧2.642366秒、21帧、21桥写入，全部reason=ok，snapshot_drops/input_waits均0，平均规划124.614ms、P95 137.340ms。桥实际输出达到0.30，反馈本体前向峰值0.349494m/s；里程计前向首末约0.378m、侧向约0.034m仅为估计。桥写入数不是SDK调用数。结束检查本体平移速度近零，桥进程无execute标志，仅只读；用户实际迈步/接管退出/回中不恢复的整体反馈与截图和日志一致，记录为接管功能通过，不认定完整导航成功或精确物理停止延迟/距离。没有触发瞬间具体按键/轴值事件档案，不能反推操作是哪一键。\n\nsummary.error的下发前检查失败是接管锁存后桥不可继续写入的通用检查信息，最终分类仍taken_over。共享bridge.stderr.log的CycloneDDS prevent_callbacks断言mtime为14:30:49，早于14:45:17本轮，是旧部署重启记录，本轮未新增，不能归因本轮。先前排查时一度怀疑当前退出，核对时间后已向用户更正；未为旧条目修改SDK退出或重启服务。\n\n原始metadata.json、telemetry.jsonl、summary.json、latest.npz保持不变。record_rgbd=false，没有frames序列或完整视频。新增motion_analysis.json记录现场反馈和核验边界。电脑副本在web_console/data/web_execute_20261109_144517_b8dfc2.zip，分析takeover-user-motion-analysis-20261109.json，以及status/summary/metadata副本；人工分析同时写入盒子实验目录和本机ZIP。网页实验记录提供查看和导出，浏览器下载路径由设置决定；BOARD_WEB_GUIDE新增日志位置、文件解释与SSH命令，桌面MD只是操作记录。\n\n本次只读核验和静态页面修正，没有新运动指令、没有重启控制服务。将static/index.html残留CHAPTER 03页脚改为GO2及视觉导航实验台；盒子原子替换前保存index.before-footer-cleanup-20261109.html。接管说明和测试计划补充本次现场验收，物理停止距离和障碍闭环仍待后续验证。\n')
    if any(e['action'] == 'go2_source_layout_cleaned_20261109' for e in events):
        simple += '\n2026年11月9日：按用户最终要求拆分目录。根目录deploy只放部署脚本与服务配置，网页单独go2_web，感知规划及机器狗接口单独go2_runtime，辅助记录在tools/go2，文档在docs/go2；real_robot_deploy已删除。清理电脑下载的实验ZIP、逐帧状态副本、旧预览与缓存约43MB，保留盒子原始数据、标定和点云依赖；以后不主动下载板载实验日志。用户已关机充电，本次没有连接机器狗或部署，下次开机再部署与验证。\n'
        details.append('\n## 2026-11-09 根目录整理、删除本地副本与后续导航计划\n\n用户先要求删除本地下载日志，并问deploy与real_robot_deploy是否可合并；随后明确deploy只用于部署，网页独立根目录文件夹。最终结构：deploy/{deploy.sh,deploy_runtime.py,build_runtime.sh,prepare_camera.sh,setup_box_ap.sh,go2-ours-web.service,requirements.board.txt}；go2_web保存原web_console实现与有效data；go2_runtime保存原deploy/ours功能代码、依赖、标定与早期电脑运行的原始实验，ros子目录保存点云工具；tools/go2为check_board_web.py/update_records.py/operations.jsonl及离线迁移测试；docs/go2为说明和测试计划。中途临时合并方案已被用户纠正并替换，不保留别名目录或旧入口。\n\n修改Python项目根目录推导、运行路径、CLI脚本、systemd路径、ROS覆盖依赖路径及文档链接。删除start_wired_web.sh重复包装，旧runtime长篇过时教程改为当前入口。部署器不再导入电脑runs，标定仅首次初始化；下次部署将旧板载deploy/ours及real_robot_deploy/web_console迁移到go2_runtime及go2_web，保留runs/.deps/配置，发现新旧目录冲突拒绝覆盖。源文件清单排除日志与本地标定。迁移测试在临时目录验证原始数据字节保持与冲突拒绝，不连接机器狗。离线最终通过18项运行时、34项网页、3项布局迁移共55项Python测试，两项独立C++保护/接管检查及shell语法检查。迁移测试首次因测试入口未传隔离依赖路径找不到skfmm，修正manage.sh的PYTHONPATH后通过，未修改算法。\n\n已删除60项本地实验下载/诊断副本、旧网页预览与失效登录文件，约43MiB，包括本轮14:45试验ZIP和takeover-user系列；未删除盒子原始记录、用户截图、相机标定、有效运行依赖与ROS点云依赖。用户已说明机器狗关机充电，下次开机再部署；本轮无SSH、无服务重启、无运动发送。电脑布局变更尚未部署，盒子当前仍是旧目录，网页地址与热点配置不改。\n\n下一步不能将接管专用0.30直行测试当作完整导航：现普通导航限0.10/5s，接管专用模式固定直行并有8秒/2m限制。先准备独立受限连续导航模式及停止逻辑离线测试，再下次部署后只读输入/时钟/地面检查、影子验证，现场确认后依次空旷2米到点、偏置目标转向、单纸箱绕行。完整到点、转向绕障、停止距离、里程计误差及重复性均未完成。具体顺序与门槛见docs/go2/NEXT_SESSION.md。\n')
    simple_path.write_text(simple)
    detail_path.write_text("\n".join(details))
    print("已更新：", simple_path)
    print("已更新：", detail_path)


if __name__ == "__main__":
    main()

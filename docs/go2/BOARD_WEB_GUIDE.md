# 板载网页使用

## 当前配置

网页 **http://192.168.50.1:8765/**，热点 **GO2-OURS**，免登录。当前电脑已实现有线/无线同网址；网络与恢复见[热点说明](AP_GUIDE.md)。不要使用旧手机DHCP地址或电脑localhost代替板载地址。

盒子安装目录`/home/unitree/himloco-runtime`，服务`go2-ours-web.service`。RGB-D由盒子本地USB采集，BEV/FMM/MPPI在盒子运行，本体状态和高层速度走内部eth0；不依赖浏览器承担规划。自启动仅连接只读输入。

## 操作流程

1. 确认实时RGB/深度及本体状态，若失联先看原因；无任务时可断开后重新连接。
2. 现场遥控站立并静止，测量相机外参，在参数页保存实际测量，再做本次地面复核。
3. 设置目标，开启RGB-D采样，先运行影子规划。影子不发送速度。
4. 现场确认后才能启动短距离实际执行；当前限制前向0.3–0.5m、侧向0、最高0.10m/s、最多5秒。
5. 停止后到实验记录页查看和导出，核对实际迈步与现场结果。

2026-11-09速度响应测试：经用户单独授权的speed_calibration任务支持最高0.30m/s、直行、前方0.3–0.5m、最多3秒，桥端独立限时；普通网页任务仍保持0.10m/s/5秒限制。此入口用于起步标定，不能将普通导航表单自行改大后绕过校验。本次0.30m/s已获现场迈步/停稳确认，但最低速度、停止距离和连续避障仍待验证。

新增网页“实际测试 / 手柄接管…”：默认接管测试为2m目标、0.30m/s、最长8秒（可选4–8秒），需实际手柄输入验证和归中、本次地面复核及现场确认。操作及真实遥控数据的边界见[官方手柄接管](REMOTE_TAKEOVER.md)。本轮仅部署与只读验收，未自动运行该长测试。

地面复核有效10分钟，重启服务或修改测量后须重做。地面拟合使用深度及内参，结合狗本体姿态/高度，只估高度与倾角。前向34cm、居中和安装偏航仍不能代替实际测量；D435i IMU未融合。不要把初步标定勾选成完整标定。

目标为启动时机身坐标：+X向前，+Y向左，单位米。启动时用SportModeState位置/偏航变换为固定里程计目标，每帧再变回机身系；目标不会随狗移动前推。无SLAM全局地图，里程计到达需现场核验。点图选目标只填下次任务，不即时移动。

## 图像和记录

- RGB/深度原生640×480、采集约15fps；页面正方形区域居中裁切，保存图为完整画幅。采集帧率不等于端到端帧率/延迟。
- 深度预览为伪彩色，无效值0显示黑色；原始depth_mm为16位毫米深度。
- “截图并下载”：盒子data/captures保存同帧RGB、彩深度、原始深度及metadata，再下载ZIP至当前设备。
- 每轮go2_runtime/runs保存元数据、逐规划帧遥测、汇总和最近快照；勾选采样后约2Hz保存RGB-D，不是连续全帧视频或rosbag。
- 历史回放/流程模拟与当前实机任务分开标记。AI编辑图仅展示，不进入算法或论文定量评估。

## 日志在哪里

实验主日志由本项目后台写在**盒子**，每轮独立目录。本次源码重排尚未部署，当前路径为`/home/unitree/himloco-runtime/deploy/ours/runs/<实验ID>/`；下次完整部署后迁入`/home/unitree/himloco-runtime/go2_runtime/runs/<实验ID>/`。不是浏览器在电脑桌面自动写文件，也不是ROS bag。

| 文件 | 内容 |
| --- | --- |
| metadata.json | 启动时间、真实任务参数、模式、标定快照、采样开关 |
| telemetry.jsonl | 每规划帧一行JSON：深度帧号、本体状态、建议速度、桥实际速度/SDK结果、手柄状态、原因与分阶段耗时 |
| summary.json | 帧数、结果、退出原因、平均/P95、桥写入、丢弃等汇总 |
| latest.npz | 最近一份RGB-D与局部图快照；不是完整录像 |
| frames/ | 仅开启RGB-D采样后保留约2Hz逐时刻快照 |
| motion_analysis.json | 额外记录的用户现场观察与人工核验结论（如有）；不改原始遥测 |

在盒子上查看最近这轮手柄接管日志：

```bash
ssh unitree@192.168.123.18
cd /home/unitree/himloco-runtime/deploy/ours/runs/web_execute_20261109_144517_b8dfc2
ls -lh
cat summary.json
cat metadata.json
tail -n 2 telemetry.jsonl
```

无线SSH可用`unitree@192.168.50.1`。网页左侧“实验记录”可直接查看，只有用户主动导出才下载到浏览器设置的位置。本地已下载的实验副本已清理，之后不主动下载；电脑`go2_runtime/runs/`仅保留早期电脑运行模式的原始数据。

服务诊断：`journalctl -u go2-ours-web.service -n 50 --no-pager`。桥诊断当前位于盒子`/home/unitree/himloco-runtime/real_robot_deploy/web_console/data/bridge.stderr.log`，下次部署后为`/home/unitree/himloco-runtime/go2_web/data/bridge.stderr.log`。旧条目需核对时间，不能直接把tail内容当本轮故障。桌面MD是操作说明，不是逐帧原始日志。

## 更新与排查

```bash
# 电脑，只读检查
cd /home/lenovo/HIMLoco
python3 tools/go2/check_board_web.py --url http://192.168.50.1:8765
# 电脑，显式更新（停止任务后）
GO2_HOST=unitree@192.168.50.1 bash deploy/deploy.sh
# 盒子
systemctl status go2-ours-web.service
journalctl -u go2-ours-web.service -n 40 --no-pager
date -Is
```

隔离Python依赖位于go2_runtime/.deps，服务用python3 -S加载；不替换系统Python或运控。更新会重启网页。部署器仅在无任务时使用，更新前备份板载标定/data/runs，特别是同名历史文件；部署器不再导入电脑历史实验；原有板载目录会迁移保留。当前时间此前错误显示2025年，正式实验前须核对校时，不按旧文件名推断真实日期。

若选电脑备用模式，先在盒子停止本项目服务，再运行电脑`bash go2_runtime/manage.sh web-wired`，访问localhost:8765。结束后关闭电脑采集、恢复板载服务；这条备用路径不是日常操作入口。

当前未完成的导航验收和现场步骤见[测试计划](OUTDOOR_TEST_PLAN.md)。不能将网页/网络可用等同于连续导航已完成。

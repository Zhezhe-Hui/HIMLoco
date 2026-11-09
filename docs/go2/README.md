# GO2 实机实验

当前主方案：盒子运行网页、D435i采集、BEV、双层代价FMM和标准足迹MPPI，通过SDK2向狗发送高层速度。电脑/手机作为操作终端；开机只连接相机和状态，不自动导航。底层运控保持原样。

## 访问与常用命令

- 网页：**http://192.168.50.1:8765/**，无需登录。
- 无线：连接盒子热点 **GO2-OURS**；这台电脑有线也已通过持久路由使用同一网址。
- 盒子有线备用：192.168.123.18；其他电脑有线默认使用该地址的8765端口。
- SSH：无线`ssh unitree@192.168.50.1`，有线`ssh unitree@192.168.123.18`。

```bash
cd /home/lenovo/HIMLoco
# 只读检查，不启动任务
python3 tools/go2/check_board_web.py --url http://192.168.50.1:8765
# 完整部署；需停止任务，电脑需能下载依赖
GO2_HOST=unitree@192.168.50.1 bash deploy/deploy.sh
# 本机离线网页开发，无需开狗
bash go2_runtime/manage.sh web
# 离线检查与测试；不连接狗
bash go2_runtime/manage.sh check
bash go2_runtime/manage.sh test
# 更新桌面操作记录
bash go2_runtime/manage.sh records
```

## 文档入口

| 文档 | 用途 |
| --- | --- |
| [下次开机步骤](NEXT_SESSION.md) | 部署、连续导航软件准备与分阶段现场验收 |
| [导航测试计划](OUTDOOR_TEST_PLAN.md) | 当前限制、待修项、现场步骤、验收和实验指标 |
| [时延优化结果](LATENCY_OPTIMIZATION.md) | 计算热点、等价性、实时复测及回滚 |
| [板载使用](BOARD_WEB_GUIDE.md) | 数据流、标定、目标、截图、日志、部署与恢复 |
| [热点连接](AP_GUIDE.md) | 驱动、固定地址、有线统一路由和网络恢复 |
| [官方手柄接管](REMOTE_TAKEOVER.md) | 网页0.30m/s、4–8秒入口，手柄触发退出 |
| [ROS点云](ROS_TOPICS_GUIDE.md) | 有线ROS2话题与RViz雷达点云 |
| [网页开发](../../go2_web/README.md) | 源码职责、运行模式与测试 |
| [硬件说明](GO2_EDU_HARDWARE.md) | EDU主机与硬件资料 |
| [图像质量](VIDEO_QUALITY.md) | 分辨率、编码与帧率说明 |

## 源码目录

| 目录 | 用途 |
| --- | --- |
| `deploy/` | 部署、编译、依赖下载及 systemd 配置 |
| `go2_web/` | 网页前端、HTTP 后台、相机预览、任务与标定接口 |
| `go2_runtime/` | RGB-D采集、感知规划、SDK通信、手柄接管、ROS工具 |
| `tools/go2/` | 只读检查、操作记录生成与部署迁移离线测试 |
| `docs/go2/` | 使用说明、硬件资料与测试计划 |

板载安装根目录仍为`/home/unitree/himloco-runtime`。本次仅整理电脑源码，盒子当前仍是旧`deploy/ours`及`real_robot_deploy/web_console`布局；下次完整部署会迁移目录并保留原始数据。网页地址不变。

本地下载的实验ZIP、逐帧状态副本和旧预览缓存已按用户要求清理；之后不主动下载板载实验日志。标定、运行依赖、点云工具依赖及早期电脑运行的原始实验保留。桌面两份操作说明由`tools/go2/update_records.py`生成，操作事件在`tools/go2/operations.jsonl`。

`go2_runtime/manage.sh`默认只检查，`web`为离线网页。`shadow/trial/web-wired`为电脑运算备用入口，使用前须停止板载服务，避免同时占用D435i。完整部署统一使用`deploy/deploy.sh`。

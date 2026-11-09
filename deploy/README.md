# GO2 部署

本目录只包含部署、编译和开机服务配置。网页源码在 [go2_web](../go2_web/README.md)，机器人接口与感知规划在 [go2_runtime](../go2_runtime/README.md)。

从 HIMLoco 根目录执行：

```bash
GO2_HOST=unitree@192.168.50.1 bash deploy/deploy.sh
```

- `deploy.sh`：准备 ARM64 依赖并调用完整部署器。
- `deploy_runtime.py`：上传源码、迁移旧板载目录、保留标定与实验数据、编译并安装服务。
- `build_runtime.sh`：编译本机/盒子的速度桥和 BEV 加速库。
- `prepare_camera.sh`：电脑备用模式的相机程序部署。
- `setup_box_ap.sh`：板载 USB 网卡热点配置。
- `go2-ours-web.service`：板载只读连接自启动配置。
- `requirements.board.txt`：板载 Python 依赖版本。

部署前停止实验任务；部署会重启本项目网页服务，启动后仅连接相机与状态，不自动导航。首次使用新目录时，将盒子旧 `deploy/ours` 与 `real_robot_deploy/web_console` 原位迁入 `go2_runtime` 与 `go2_web`。若新旧目录冲突会拒绝覆盖。标定只在目标不存在时初始化，实验日志不在电脑与盒子之间自动复制。

2026-11-09 本次只整理电脑源码，没有执行板载部署。盒子当前仍运行旧目录；下次完整部署时才迁移。完整说明见 [GO2 文档](../docs/go2/README.md)。

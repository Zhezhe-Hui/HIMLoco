# GO2 感知、规划与机器狗接口

这里实现 D435i 深度接入、相机外参转换、BEV、双层代价 FMM、标准足迹 MPPI、SDK2 高层速度桥、手柄接管及停止保护。FMM/MPPI 算法继续直接加载 `legged_gym/legged_gym/envs/go2/`，不部署关节控制策略。

网页位于 [go2_web](../go2_web/README.md)，部署工具位于 [deploy](../deploy/README.md)，现场流程与历史结果见 [GO2 文档](../docs/go2/README.md)。

## 常用命令

从 HIMLoco 根目录运行：

```bash
bash go2_runtime/manage.sh check   # 本机依赖检查，不连接狗
bash go2_runtime/manage.sh test    # Python/C++ 离线检查
bash go2_runtime/manage.sh web     # 本机离线网页
bash go2_runtime/manage.sh records # 更新桌面操作记录
bash go2_runtime/manage.sh --help
```

当前日常方案是在盒子上采集和规划，电脑仅通过网页操作。`prepare/shadow/trial/web-wired` 是电脑运算备用路径，使用前先停止盒子服务以避免相机冲突。普通 CLI `trial` 仍是 0.10m/s、0.5m、5秒；当前步态在该速度下未有效迈步，不能当作已验收的导航入口。

## 文件与运行数据

- `run_ours.py`、`estimate_floor.py`：几何、规划、输入检查和地面拟合。
- `camera_stream.cpp`：原始 RGB-D、内参和时间戳采集。
- `robot_bridge.cpp`、`bridge_safety.hpp`、`remote_takeover.hpp`：状态与遥控数据、速度发送、独立保护和接管锁存。
- `bev_reduce.cpp`：等价的原生栅格聚合加速。
- `preview.py`、`profile_planner.py`：离线可视化和性能分析。
- `ros/`：电脑 ROS2 环境、RViz 点云入口。
- `calibration.local.json`、`.deps/`、`build/`：本机配置、依赖和编译产物，不入库。
- `runs/`：早期在电脑运行产生的原始实验；保留作历史回放，区别于已删除的板载下载副本。当前板载实验保存在盒子，不自动下载。

相机前向约0.34m和居中仍含估计，地面拟合只能核验高度与倾角。本体位置来自 SportModeState 里程计，不是相机 VIO。当前0.30m/s起步响应、网页停止和一次手柄接管已获现场确认；完整到点、绕障及精确停止距离仍待验收。

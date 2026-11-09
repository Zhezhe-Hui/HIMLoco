# 2026-11-06 有线ROS话题与实时点云

只读检查：盒子ROS2 Foxy/ROS1 Noetic均安装，本次使用Foxy；电脑Humble。电脑192.168.123.200通过enx00e04c584b68有线连接狗。无需启动运动节点，不发布控制命令。

## 盒子上列出话题

```bash
ssh unitree@192.168.123.18
```

盒子.bashrc会询问`foxy(1) noetic(2)`，输入1。也可以明确执行：

```bash
source /opt/ros/foxy/setup.bash
source /home/unitree/cyclonedds_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=/home/unitree/cyclonedds_ws/cyclonedds.xml
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
ros2 topic list --no-daemon -t
ros2 topic info /utlidar/cloud -v
```

盒子Foxy的旧版topic echo没有本机Humble支持的--once/--field选项，不照搬这些选项到盒子。

## 电脑直接查看

保持网线连接；在电脑新终端：

```bash
cd /home/lenovo/HIMLoco
source go2_runtime/ros/ros2_pc_env.sh
ros2 topic list --no-daemon -t
ros2 topic echo /utlidar/cloud sensor_msgs/msg/PointCloud2 --qos-reliability best_effort --once --field header
```

打开实时点云：

```bash
bash /home/lenovo/HIMLoco/go2_runtime/ros/view_lidar.sh
```

RViz已配置PointCloud2=/utlidar/cloud，Fixed Frame=utlidar_lidar，Best Effort/Volatile，点大小3像素，按Z着色，无累积。鼠标旋转/缩放观察。30fps为RViz渲染上限，不是雷达频率。

本机缺少rmw_cyclonedds_cpp且sudo需密码，因此只下载官方ROS apt仓库的5个运行包（rmw_cyclonedds_cpp、cyclonedds、iceoryx-binding-c/hoofs/posh），解包到web_console/data/ros-pc-overlay/root，不做系统apt安装、不改.bashrc。source脚本仅设置当前终端环境。脚本设置domain0、CycloneDDS和有线接口，避免选到校园网或其他网卡；已有ROS daemon可能使用别的RMW，所以列话题用--no-daemon。

若本机ROS升级后出现二进制不兼容，需同步刷新隔离插件包，或由用户在本机正式安装匹配的ros-humble-rmw-cyclonedds-cpp。Unitree非标准消息（unitree_go等）在本机echo前还需编译相应接口包；本次没有为读取标准PointCloud2编译全部Unitree工程。不要在未安装接口时将echo类型失败误判为网络断开。

## 实测话题与范围

本轮保存的完整话题列表位于web_console/data/ros-box-topics-20261106.txt、ros-pc-topics-20261106.txt。约124条发现结果不是124个持续发数据的传感器：还含控制API、地图/SLAM话题及本地参数事件。必须分别订阅确认。

| 话题 | 类型/含义 |
| --- | --- |
| /utlidar/cloud | PointCloud2，本次实际接收的雷达点云 |
| /utlidar/cloud_base、/utlidar/cloud_deskewed | PointCloud2，其他坐标/处理版本；本次仅发现，未验证数据持续输出 |
| /utlidar/imu | sensor_msgs/Imu，雷达相关IMU话题 |
| /utlidar/robot_odom、/lio_sam_ros2/mapping/odometry | nav_msgs/Odometry |
| /sportmodestate、/lf/sportmodestate | unitree_go/SportModeState |
| /lowstate、/lf/lowstate | unitree_go/LowState |
| /uslam/... | 定位、地图等相关话题，发现不代表算法正在运行 |

电脑通过rclpy实际接收/utlidar/cloud十帧，短窗口估计15.04Hz，首帧1122点、末帧1185点，字段x/y/z/intensity/ring/time，frame_id=utlidar_lidar。短采样不能等同长期带宽/丢包验收。证据ros-cloud-sample-20261106.json。启动RViz日志rviz-lidar.log；未通过自动化截图做视觉验收。

当前D435i由既有网页camera_stream进程直接采集，并未发布ROS图像/点云；不能同时盲目启动第二个RealSense驱动占用同一设备。若以后要在RViz看D435i，需要单独设计网页采集与ROS发布的共用数据管线，本次未改。

## 无线边界

本次证明的是网线直连DDS。电脑连GO2-OURS后网页/SSH可用，不代表狗内部eth0 DDS自动跨网段发现；不能只把脚本网卡改成Wi-Fi就保证收点云。无线点云需在盒子明确配置中继/路由与发现、实测吞吐，本次未做。

官方通信配置参考：https://github.com/unitreerobotics/unitree_ros2 。本轮没有启动厂家运动示例、修改本体服务、改网页控制接口或发送运动指令。

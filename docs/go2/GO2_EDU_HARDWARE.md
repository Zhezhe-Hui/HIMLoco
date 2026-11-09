# Go2 EDU 主机、设备和开发入口

核查日期：2026-11-01。区分官方产品系列能力与本机已观察结果，EDU配置、出厂批次和选配不同，不能只凭EDU名称推断全部硬件。

## 哪些主机可以开发

| 单元 | 本机已观察地址 | 开发方式与限制 |
| --- | --- | --- |
| 背部算力扩展坞 | 192.168.123.18 | 本机已SSH登录，识别为Jetson Orin Nano，Ubuntu20.04、ARM64。可装依赖、放代码、接USB相机，运行ROS/感知/规划/网页。账号unitree。 |
| 狗本体网络控制端 | 192.168.123.161 | 本机通过SDK2/DDS读取状态与高层控制；此前22端口拒绝连接。不能按可SSH登录的Ubuntu开发机使用，也没有依据把任意芯片型号或内部所有控制职责都归到该IP。 |
| 内部网络网关地址 | 192.168.123.1 | 本机曾观察为盒子的默认网关，SSH超时；不是已经确认开放的第三台开发机，不能据此认定是某个MCU。 |
| 外部开发电脑 | 192.168.123.200 | 本项目当前网页、BEV/FMM/MPPI运行在这里；连接狗网口后直接通过DDS访问本体，不要求算法必须运行在盒子里。 |

盒子SSH例：`ssh unitree@192.168.123.18`。密码不写入操作记录。以上不是本次关机时重新扫描的结果，而是本机此前联调记录。

逻辑上可把系统理解为“可部署应用的开发计算机 + 厂商本体控制系统”；物理上还包括实时控制板、电机驱动、传感器处理芯片、网络模块等，不应说整只狗恰好只有两块CPU。官方规格提到本体8核高性能CPU与EDU可配置Orin等扩展算力，但未据此确认本机内部CPU具体型号、数量或SSH权限。

## 主要硬件与入口

| 硬件 | 能开发什么 | 本机状态 |
| --- | --- | --- |
| Jetson算力盒子 | 应用、模型推理、ROS节点、网页、数据采集 | 已确认Orin Nano；其他EDU可能选Orin NX等，不代表本机有NX。 |
| 外接D435i | RGB、双目深度、相机IMU数据 | 已接盒子USB3，现项目使用RGB/深度/内参；未接入D435i IMU，也没替代本体IMU。USB设备本身不作为独立SSH主机。 |
| 本体前置广角RGB相机 | 图像应用；官方SDK有VideoClient/GetImageSample | 与D435i是两套相机，当前网页用D435i，不是本体前置相机。 |
| 4D激光雷达 | 点云、建图/定位、其他避障方案 | 官方新页面列L2，较早资料有L1；本机具体型号待铭牌/驱动确认。官方ROS2示例有utlidar/cloud。当前ours不使用雷达。 |
| 本体IMU、关节编码器和12个关节电机 | 状态、姿态、关节级控制接口 | 本项目只使用高层运动状态/速度；关节驱动和实时控制不是通用应用部署目标。 |
| 足端力/接触相关数据 | 足部接触、状态分析 | 当前官方EDU规格列足端力传感器；本机具体装配与字段语义需核对，不能用消息字段存在来证明独立传感器存在。 |
| Wi-Fi、蓝牙、4G/音频等 | 官方APP连接、部分交互功能 | 配置和地区可能不同；本体有Wi-Fi不代表Jetson有可用无线网卡，本机盒子此前只发现eth0。 |
| 电池/BMS、电源、手持遥控器 | 电源状态、现场接管 | 不作为可以随意SSH装软件的主机。 |

盒子/机身上的RJ45、USB、显示和电源接口按实际外壳与扩展坞型号核对；不把其他EDU照片上的接口数量当成本机清单。

## 对当前项目的意义

当前：D435i → 盒子采集 → 有线SSH原始数据 → 电脑BEV/FMM/MPPI → DDS高层速度 → 本体。浏览器 → 电脑localhost网页服务。开发不必全部在盒子完成；以后若要脱离电脑，可把应用迁到盒子，需单独做算力与实时性验收。

## 资料

- [宇树官方Go2规格](https://www.unitree.com/go2/)：本体CPU、12关节、传感器、网络及EDU扩展算力，且明确配置会变化。
- [宇树官方SDK2](https://github.com/unitreerobotics/unitree_sdk2)：ARM64/x86_64开发与示例。
- [宇树官方ROS2](https://github.com/unitreerobotics/unitree_ros2)：DDS通信、ROS2开发及点云示例。
- [官方前置相机VideoClient](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/go2/video/video_client.hpp)。
- [官方用户手册入口](https://www.unitree.com/cn/app/go2/)及[开发文档入口](https://support.unitree.com/home/zh/developer/)。开发文档网页本次无法提取正文，故未把未读章节作为SSH权限或芯片型号的依据。
- [RealSense D400数据手册](https://realsenseai.com/wp-content/uploads/2025/08/Intel-RealSense-D400-Series-Datasheet-August-2025.pdf)：D435i支持更高RGB分辨率；当前采集配置不是硬件上限。

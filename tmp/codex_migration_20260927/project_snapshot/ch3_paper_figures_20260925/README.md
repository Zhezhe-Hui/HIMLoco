# 第三章论文图表

1. `01_success_rate_with_ci`：三档成功率及 Wilson 95% 置信区间。
2. `02_outcome_composition`：成功、碰撞、卡住、超时构成。
3. `03_smoothness_and_clearance`：控制平滑性与路径安全净空。
4. `04_navigation_efficiency`：到达时间、路径长度与规划耗时。
5. `05_hierarchical_navigation_flowchart`：分层视觉导航系统流程。
6. `06_footprint_mppi_flowchart`：双层代价 FMM 引导的足迹 MPPI 算法流程。
7. `07_experiment_protocol_flowchart`：三档实验设计流程。
8. `08~15`：Isaac Gym场景、轨迹、RGB-D/BEV/代价图和过程素材（由
   `generate_ch3_visual_evidence.py`生成）。
9. `16_vio_navigation_success`：四档VIO噪声下的严格到达与阈值敏感性结果。
10. `17_vio_pose_error_and_false_arrival`：位姿RMS与VIO误到达。
11. `18_vio_outcome_composition`：四档VIO的成功和失败构成。

每张图同时提供 300 dpi PNG 和矢量 PDF。表格提供 UTF-8 BOM CSV 和 Markdown。
规划耗时只应在同一难度内横向比较，因为三档实验硬件不完全一致。
VIO图和表由`analyze_vio_robustness.py`生成；严格成功阈值为真值误差0.15 m，
0.20 m结果仅用于阈值敏感性分析。

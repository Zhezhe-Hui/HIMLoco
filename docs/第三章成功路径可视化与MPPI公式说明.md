# 第三章成功路径可视化与MPPI公式说明

## 1. 生成的路径图

图片位于：

`assets/runs/navigation_benchmark/ch3_successful_route_figures_20261002/`

| 文件 | 内容 |
| --- | --- |
| `01_easy_four_successful_routes_overlay.png` | 简单场景中四种方法的完整成功路径叠加图 |
| `02_medium_four_successful_routes_overlay.png` | 中等场景中四种方法的完整成功路径叠加图 |
| `03_hard_four_successful_routes_overlay.png` | 困难场景中四种方法的完整成功路径叠加图 |
| `04_three_difficulties_four_methods_overlay_1x3.png` | 上述三张图的横向1×3组合图 |
| `05_fixed_fmm_vs_soft_cost_footprint_mppi_overlay.png` | 同一简单场景、同一seed下两种方法的路径叠加对比 |
| `successful_route_manifest.json` | 各子图对应的seed、成功标志、路径长度和原始数据路径 |

重新生成全部图片的命令为：

```bash
cd /home/hzz/project/HIMLoco
conda activate himloco
python legged_gym/legged_gym/scripts/generate_successful_route_comparisons.py \
  --input assets/runs/navigation_benchmark/ch3_route_capture_20261002 \
  --output assets/runs/navigation_benchmark/ch3_successful_route_figures_20261002
```

所有图片仅保留300 dpi PNG，不再生成PDF。所有路径都由仿真中的真实执行轨迹生成，绘图脚本在出图前强制检查
`result.success == true`，不使用失败、超时或中途截断的轨迹。彩色圆点表示起点，彩色叉号表示终点，
实线表示机器人实际执行轨迹。

### 1.1 选用的成功试验

| 难度 | 本文方法 | iPlanner | PS-DWA | SAPF |
| --- | ---: | ---: | ---: | ---: |
| 简单 | 20261013 | 20261013 | 20261013 | 20261013 |
| 中等 | 20261012 | 20261013 | 20261013 | 20261016 |
| 困难 | 20261005 | 20261019 | 20261016 | 20261038 |

简单场景四种方法采用相同seed，可以直观比较同一障碍布局下的路线差异。
中等和困难场景的四种方法没有可稳定重现的共同成功seed，因此分别选取各方法的
完整成功试验作定性展示。这两张图不应写成“同一障碍布局的配对对比”；方法性能
的定量结论应以三档30-seed统计、McNemar检验和Wilcoxon检验为准。

### 1.2 论文图注建议

**四方法路径图：** 四种局部规划方法在离散高度地形随机障碍场景中的完整成功
执行路径。彩色圆点和叉号分别表示起点和终点，实线表示机器人实际轨迹。

**FMM与本文方法对比图：** 固定膨胀FMM与双层代价FMM引导的足迹MPPI在相同简单场景
和相同随机种子下的完整成功路线对比。后者同时考虑连续障碍安全代价、矩形机身足迹
与控制变化。

## 2. MPPI的代码来源

本项目没有直接复制某个外部MPPI仓库。当前实现位于
`legged_gym/legged_gym/envs/go2/mppi_local_planner.py`，是基于MPPI采样加权思想编写的
NumPy局部控制器，并针对本项目加入了FMM参考走廊、Go2矩形机身足迹、障碍净空代价
和速度指令平滑。Git记录显示该模块在2026-09-23由本项目提交。

论文中可写为：“本文参考Williams等提出的模型预测路径积分控制框架，根据四足机器人
局部导航需求设计轻量化采样优化器。”建议引用：

Williams G, Aldrich A, Theodorou E A. Model Predictive Path Integral Control: From Theory to
Parallel Computation[J]. Journal of Guidance, Control, and Dynamics, 2017, 40(2): 344-357.
DOI: 10.2514/1.G001921.

## 3. 车体模型与控制量

局部状态定义为

$$
\mathbf{x}_t=[x_t,y_t,\theta_t]^\mathrm{T},
\qquad
\mathbf{u}_t=[v_t,\omega_t]^\mathrm{T},
$$

其中 $v_t$ 和 $\omega_t$ 分别为传递给HIMLoco底层策略的期望前向速度和偏航角速度。
预测模型采用离散单轨模型：

$$
\begin{aligned}
x_{t+1} &= x_t+v_t\cos\theta_t\Delta t,\\
y_{t+1} &= y_t+v_t\sin\theta_t\Delta t,\\
\theta_{t+1} &= \operatorname{wrap}(\theta_t+\omega_t\Delta t).
\end{aligned}
$$

当前预测时域 $T=20$，预测步长 $\Delta t=0.1\ \mathrm{s}$，即每条候选轨迹预测2 s。
每次优化采样 $K=384$ 组速度序列：

$$
\mathbf{U}^{(k)}=\operatorname{clip}\left(\bar{\mathbf{U}}+
\boldsymbol{\epsilon}^{(k)}\right),\qquad k=1,\ldots,K.
$$

为减少逐帧抖动，代码对随机扰动使用时间相关滤波：

$$
\boldsymbol{\epsilon}^{(k)}_t=0.65\boldsymbol{\epsilon}^{(k)}_{t-1}
+0.35\boldsymbol{\eta}^{(k)}_t.
$$

## 4. 轨迹代价函数

本文实现的单条候选轨迹总代价为

$$
J_k=J_{\mathrm{track}}+J_{\mathrm{terminal}}+J_{\mathrm{heading}}
+J_{\mathrm{collision}}+J_{\mathrm{clearance}}+J_{\mathrm{smooth}}+J_{\mathrm{control}}.
$$

各项与当前代码一致：

$$
J_{\mathrm{track}}=5\sum_{t=1}^{T}
\left\|\mathbf{p}^{(k)}_t-\mathbf{r}_t\right\|_2^2,
$$

$$
J_{\mathrm{terminal}}=15
\left\|\mathbf{p}^{(k)}_T-\mathbf{r}_T\right\|_2^2,
\qquad
J_{\mathrm{heading}}=3\,\operatorname{wrap}
(\theta_T^*-\theta_T^{(k)})^2,
$$

$$
J_{\mathrm{collision}}=2500\sum_{t=1}^{T}I
\left(\mathcal{F}(\mathbf{x}^{(k)}_t)\cap\mathcal{O}\ne\varnothing\right),
$$

$$
J_{\mathrm{clearance}}=18\sum_{t=1}^{T}
\left[\frac{\max(0,0.65-d_t^{(k)})}{0.65}\right]^2,
$$

$$
J_{\mathrm{smooth}}=1.5\sum_{t=2}^{T}(v_t-v_{t-1})^2
+4\sum_{t=2}^{T}(\omega_t-\omega_{t-1})^2,
$$

$$
J_{\mathrm{control}}=0.08\sum_{t=1}^{T}(v_{\max}-v_t)^2
+0.35\sum_{t=1}^{T}\omega_t^2.
$$

其中 $\mathbf{r}_t$ 是沿双层代价FMM路径取得的参考点，$\mathcal{F}$ 是包含安全余量的
Go2矩形足迹，$\mathcal{O}$ 是BEV硬障碍集，$d_t$ 是预测位置到最近障碍的距离。
当前足迹基础尺寸为 $0.72\ \mathrm{m}\times0.42\ \mathrm{m}$，四周再增加 $0.05\ \mathrm{m}$ 安全余量。

## 5. MPPI加权更新

对所有候选轨迹计算代价后，先减去最小代价以提高数值稳定性：

$$
\rho=\min_k J_k,
\qquad
w_k=\frac{\exp[-(J_k-\rho)/\lambda]}
{\sum_{j=1}^{K}\exp[-(J_j-\rho)/\lambda]},
$$

其中温度参数 $\lambda=8.0$。名义控制序列更新为

$$
\bar{\mathbf{U}}\leftarrow\sum_{k=1}^{K}w_k\mathbf{U}^{(k)}.
$$

系统仅执行更新后序列的第一个控制量，随后将序列前移并重新规划，构成滚动时域控制。
MPPI以12.5 Hz重新采样优化，速度指令仍以50 Hz发送给HIMLoco底层策略。

## 6. 双层代价FMM

FMM用于从局部目标向机器人位置传播代价并生成参考走廊。障碍代价由硬碰撞层和连续
软安全层组成：

$$
C(d)=
\begin{cases}
+\infty, & d\le r_h,\\
w\left(\dfrac{r_s-d}{r_s-r_h}\right)^p, & r_h<d<r_s,\\
0, & d\ge r_s,
\end{cases}
$$

其中 $d$ 是栅格到最近障碍的距离，当前 $r_h=0.20\ \mathrm{m}$、$r_s=0.65\ \mathrm{m}$、
$w=4.0$、$p=2.0$。硬层保证不碰撞，软层鼓励路径在可通行条件下保持较大净空；
MPPI再对FMM走廊附近的速度序列进行足迹级动力学可执行性优化。

"""Generate paper-ready Chapter 3 plots, tables, and flowcharts."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np


SCENES = (
    ("简单", "ch3_easy_30seed_20260924", "10大石 + 2树 + 100小碎石"),
    ("中等", "ch3_medium_30seed_20260924", "15大石 + 3树 + 200小碎石"),
    ("困难", "ch3_hard_30seed_20260924", "20大石 + 4树 + 400小碎石"),
)
METHODS = ("fmm_fixed", "fmm_soft", "mppi")
METHOD_LABELS = {
    "fmm_fixed": "固定膨胀 FMM",
    "fmm_soft": "双层代价 FMM",
    "mppi": "双层代价 FMM 引导的足迹 MPPI",
}
METHOD_COLORS = {
    "fmm_fixed": "#2878B5",
    "fmm_soft": "#E39D26",
    "mppi": "#2A9D6F",
}
CATEGORY_COLORS = {
    "success": "#2A9D6F",
    "collision": "#D55252",
    "stuck": "#7B61A8",
    "timeout": "#A7ADB4",
    "other": "#5B6770",
}


def configure_style():
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=path).get_name()
            break
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_data(root):
    data = {}
    for label, directory, description in SCENES:
        path = root / directory / "statistical_analysis.json"
        if not path.exists():
            raise FileNotFoundError("missing analysis: %s" % path)
        analysis = json.loads(path.read_text(encoding="utf-8"))
        methods = {row["method"]: row for row in analysis["methods"]}
        if set(methods) != set(METHODS):
            raise ValueError("unexpected methods in %s" % path)
        data[label] = {
            "directory": directory,
            "description": description,
            "analysis": analysis,
            "methods": methods,
        }
    return data


def save_figure(fig, output, stem):
    fig.savefig(str(output / (stem + ".png")), dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(str(output / (stem + ".pdf")),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_success_rates(data, output):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(SCENES), dtype=float)
    width = 0.23
    offsets = (-width, 0.0, width)
    for method, offset in zip(METHODS, offsets):
        rows = [data[label]["methods"][method] for label, _, _ in SCENES]
        rates = np.array([r["success_rate"] * 100.0 for r in rows])
        low = np.array([r["success_ci95_low"] * 100.0 for r in rows])
        high = np.array([r["success_ci95_high"] * 100.0 for r in rows])
        bars = ax.bar(x + offset, rates, width, label=METHOD_LABELS[method],
                      color=METHOD_COLORS[method], edgecolor="white", linewidth=0.8,
                      yerr=np.vstack((rates - low, high - rates)), capsize=3,
                      error_kw={"elinewidth": 1.0, "capthick": 1.0})
        for bar, row in zip(bars, rows):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 3.2,
                    "%d/%d" % (row["successes"], row["trials"]),
                    ha="center", va="bottom", fontsize=8)
    ax.set_title("不同障碍难度下的导航成功率")
    ax.set_ylabel("成功率（%）")
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _ in SCENES])
    ax.set_ylim(0, 108)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#D9DDE1", linewidth=0.7, alpha=0.8)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.text(0.01, 0.98, "误差线：Wilson 95% 置信区间",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color="#4C565F")
    fig.subplots_adjust(bottom=0.23)
    save_figure(fig, output, "01_success_rate_with_ci")


def plot_outcomes(data, output):
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2), sharey=True)
    categories = ("success", "collision", "stuck", "timeout", "other")
    labels = {"success": "成功", "collision": "碰撞", "stuck": "卡住",
              "timeout": "超时", "other": "其他"}
    for ax, (scene, _, _) in zip(axes, SCENES):
        y = np.arange(len(METHODS))
        left = np.zeros(len(METHODS))
        for category in categories:
            values = np.array([
                data[scene]["methods"][method]["failure_categories"].get(
                    category, 0) for method in METHODS
            ])
            bars = ax.barh(y, values, left=left, height=0.58,
                           color=CATEGORY_COLORS[category], label=labels[category])
            for bar, value in zip(bars, values):
                if value >= 3:
                    ax.text(bar.get_x() + bar.get_width() / 2.0,
                            bar.get_y() + bar.get_height() / 2.0,
                            str(int(value)), ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
            left += values
        ax.set_title(scene)
        ax.set_xlim(0, 30)
        ax.set_xticks((0, 10, 20, 30))
        ax.grid(axis="x", color="#D9DDE1", linewidth=0.7, alpha=0.8)
        ax.set_xlabel("试验次数")
    axes[0].set_yticks(np.arange(len(METHODS)))
    axes[0].set_yticklabels([METHOD_LABELS[m] for m in METHODS])
    fig.suptitle("三种方法在不同难度下的试验结果构成", y=1.02, fontsize=13)
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.20, wspace=0.12)
    save_figure(fig, output, "02_outcome_composition")


def grouped_metric(ax, data, metric, ylabel, title, decimals=2, scale=1.0):
    x = np.arange(len(SCENES), dtype=float)
    width = 0.23
    for method, offset in zip(METHODS, (-width, 0.0, width)):
        values = [data[label]["methods"][method][metric] * scale
                  for label, _, _ in SCENES]
        bars = ax.bar(x + offset, values, width, color=METHOD_COLORS[method],
                      edgecolor="white", linewidth=0.7, label=METHOD_LABELS[method])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(), ("%%.%df" % decimals) % value,
                    ha="center", va="bottom", fontsize=7, rotation=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _ in SCENES])
    ax.grid(axis="y", color="#D9DDE1", linewidth=0.7, alpha=0.8)


def plot_quality_and_cost(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    grouped_metric(axes[0], data, "cmd_wz_variation_mean", "角速度指令总变差",
                   "控制平滑性（越低越好）", decimals=3)
    grouped_metric(axes[1], data, "min_path_clearance_m_mean", "最小路径净空（m）",
                   "路径安全净空（越高越好）", decimals=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.20, wspace=0.25)
    save_figure(fig, output, "03_smoothness_and_clearance")


def plot_efficiency(data, output):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    grouped_metric(axes[0], data, "elapsed_s_mean", "到达时间（s）",
                   "成功样本到达时间", decimals=1)
    grouped_metric(axes[1], data, "distance_m_mean", "路径长度（m）",
                   "成功样本路径长度", decimals=1)
    grouped_metric(axes[2], data, "planning_mean_ms_mean", "单步规划耗时（ms）",
                   "规划计算开销", decimals=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, 0.035,
             "注：规划耗时仅用于同一难度内横向比较；三档实验硬件不完全一致。",
             ha="center", fontsize=8, color="#4C565F")
    fig.subplots_adjust(bottom=0.23, wspace=0.28)
    save_figure(fig, output, "04_navigation_efficiency")


def add_box(ax, center, size, text, face, edge="#33424D", fontsize=10,
            linewidth=1.2):
    x, y = center
    w, h = size
    box = FancyBboxPatch((x - w / 2.0, y - h / 2.0), w, h,
                         boxstyle="round,pad=0.012,rounding_size=0.015",
                         facecolor=face, edgecolor=edge, linewidth=linewidth)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            linespacing=1.35)
    return box


def add_arrow(ax, start, end, text=None, color="#42515C", style="->"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": style, "color": color,
                            "linewidth": 1.35, "shrinkA": 2, "shrinkB": 2})
    if text:
        ax.text((start[0] + end[0]) / 2.0,
                (start[1] + end[1]) / 2.0 + 0.025,
                text, ha="center", va="bottom", fontsize=8, color=color)


def flow_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def draw_system_flow(output):
    fig, ax = flow_axes((13.0, 5.8))
    xs = (0.08, 0.24, 0.40, 0.57, 0.74, 0.91)
    colors = ("#DCEAF5", "#E3F0E8", "#F7E8C8", "#E8E0F2", "#DCEAF5", "#E3F0E8")
    texts = (
        "D435i 深度图\n+ 目标相对位置",
        "局部 BEV 建图\n占据栅格 / 净空",
        "FMM 路径引导\n硬膨胀或软代价",
        "双层代价 FMM 引导\n足迹 MPPI 局部优化",
        "速度指令\n$(v_x, \\omega_z)$",
        "HIMLoco 策略\n12维关节动作",
    )
    for x, text, color in zip(xs, texts, colors):
        add_box(ax, (x, 0.60), (0.135, 0.18), text, color, fontsize=9.5)
    for left, right in zip(xs[:-1], xs[1:]):
        add_arrow(ax, (left + 0.070, 0.60), (right - 0.070, 0.60))
    add_box(ax, (0.91, 0.25), (0.135, 0.16),
            "Go2 与复杂地形\n机身位姿 / 深度更新", "#F3DDDB", fontsize=9.5)
    add_arrow(ax, (0.91, 0.51), (0.91, 0.34), "执行")
    ax.annotate("", xy=(0.17, 0.53), xytext=(0.84, 0.25),
                arrowprops={"arrowstyle": "->", "color": "#536773",
                            "linewidth": 1.3,
                            "connectionstyle": "arc3,rad=-0.22"})
    ax.text(0.50, 0.20, "理想 VIO（仿真位姿真值）提供相对位姿反馈",
            ha="center", fontsize=9, color="#536773")
    ax.text(0.40, 0.82, "全局/局部引导层", ha="center", fontsize=10,
            color="#7A5412", fontweight="bold")
    ax.text(0.70, 0.82, "局部轨迹规划层", ha="center", fontsize=10,
            color="#614B7A", fontweight="bold")
    ax.text(0.91, 0.82, "强化学习运动控制层", ha="center", fontsize=10,
            color="#246342", fontweight="bold")
    ax.set_title("四足机器人分层视觉导航系统流程", fontsize=14, pad=10)
    save_figure(fig, output, "05_hierarchical_navigation_flowchart")


def draw_mppi_flow(output):
    fig, ax = flow_axes((9.2, 8.0))
    ys = (0.90, 0.77, 0.64, 0.51, 0.38, 0.25, 0.12)
    texts = (
        "输入：局部占据 BEV、FMM 路径、局部目标",
        "移位上一周期最优控制序列，构造名义序列",
        "采样 384 组相关控制噪声，生成 $(v_x, \\omega_z)$ 序列",
        "独轮车模型前向滚动：20步，$\\Delta t=0.1$ s",
        "矩形机身足迹碰撞检查与轨迹代价计算",
        "按代价指数加权更新控制序列并施加变化率限制",
        "输出首个控制 $(v_x, \\omega_z)$，下一周期滚动优化",
    )
    colors = ("#DCEAF5", "#E3F0E8", "#F7E8C8", "#E8E0F2",
              "#F3DDDB", "#E3F0E8", "#DCEAF5")
    for y, text, color in zip(ys, texts, colors):
        add_box(ax, (0.50, y), (0.68, 0.085), text, color, fontsize=10)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        add_arrow(ax, (0.50, y1 - 0.045), (0.50, y2 + 0.045))
    add_box(ax, (0.88, 0.51), (0.19, 0.25),
            "总代价\n\n路径跟踪\n终点与航向\n足迹碰撞\n障碍净空\n控制平滑\n控制幅值",
            "#FAF4E8", fontsize=9)
    add_arrow(ax, (0.785, 0.51), (0.845, 0.51), "评价")
    ax.annotate("", xy=(0.80, 0.77), xytext=(0.80, 0.12),
                arrowprops={"arrowstyle": "->", "color": "#6A7882",
                            "linewidth": 1.2,
                            "connectionstyle": "arc3,rad=-0.25"})
    ax.text(0.83, 0.18, "滚动时域反馈", rotation=90,
            ha="center", fontsize=8, color="#6A7882")
    ax.set_title("双层代价 FMM 引导的足迹 MPPI 算法流程", fontsize=14, pad=8)
    save_figure(fig, output, "06_footprint_mppi_flowchart")


def draw_experiment_flow(output):
    fig, ax = flow_axes((11.6, 6.0))
    add_box(ax, (0.12, 0.73), (0.18, 0.15),
            "固定控制变量\n离散高度地形\n理想 VIO\nHIMLoco 策略", "#DCEAF5")
    add_box(ax, (0.38, 0.84), (0.18, 0.12), "简单\n10 / 2 / 100", "#E3F0E8")
    add_box(ax, (0.38, 0.64), (0.18, 0.12), "中等\n15 / 3 / 200", "#F7E8C8")
    add_box(ax, (0.38, 0.44), (0.18, 0.12), "困难\n20 / 4 / 400", "#F3DDDB")
    for y in (0.84, 0.64, 0.44):
        add_arrow(ax, (0.215, 0.73), (0.285, y))
    add_box(ax, (0.64, 0.73), (0.21, 0.18),
            "每档 30 个固定 seed\n逐 seed 配对运行\n固定 FMM / 软 FMM / MPPI",
            "#E8E0F2")
    for y in (0.84, 0.64, 0.44):
        add_arrow(ax, (0.475, y), (0.535, 0.73))
    add_box(ax, (0.88, 0.73), (0.18, 0.20),
            "评价指标\n成功率与失败类型\n时间 / 路径 / 平滑性\n规划耗时 / 净空 / 能耗",
            "#DCEAF5")
    add_arrow(ax, (0.75, 0.73), (0.785, 0.73))
    add_box(ax, (0.64, 0.27), (0.28, 0.14),
            "配对统计\nMcNemar：成功率\nWilcoxon：共同成功样本连续指标",
            "#E3F0E8")
    add_arrow(ax, (0.88, 0.62), (0.72, 0.34))
    ax.text(0.38, 0.28, "数字依次为：大石 / 树 / 小碎石",
            ha="center", fontsize=9, color="#536773")
    ax.set_title("第三章三档难度导航实验设计流程", fontsize=14, pad=10)
    save_figure(fig, output, "07_experiment_protocol_flowchart")


def write_tables(data, output):
    main_fields = (
        "难度", "方法", "成功数", "总试验数", "成功率", "95%CI",
        "到达时间_s", "路径长度_m", "角速度变化", "规划耗时_ms",
        "最小净空_m", "碰撞", "卡住", "超时",
    )
    rows = []
    for scene, _, _ in SCENES:
        for method in METHODS:
            r = data[scene]["methods"][method]
            failures = r["failure_categories"]
            rows.append({
                "难度": scene,
                "方法": METHOD_LABELS[method],
                "成功数": r["successes"],
                "总试验数": r["trials"],
                "成功率": "%.1f%%" % (100.0 * r["success_rate"]),
                "95%CI": "[%.1f%%, %.1f%%]" % (
                    100.0 * r["success_ci95_low"], 100.0 * r["success_ci95_high"]),
                "到达时间_s": "%.2f" % r["elapsed_s_mean"],
                "路径长度_m": "%.2f" % r["distance_m_mean"],
                "角速度变化": "%.4f" % r["cmd_wz_variation_mean"],
                "规划耗时_ms": "%.2f" % r["planning_mean_ms_mean"],
                "最小净空_m": "%.3f" % r["min_path_clearance_m_mean"],
                "碰撞": failures.get("collision", 0),
                "卡住": failures.get("stuck", 0),
                "超时": failures.get("timeout", 0),
            })
    with (output / "table_main_results.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=main_fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| " + " | ".join(main_fields) + " |",
             "| " + " | ".join(["---"] * len(main_fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in main_fields) + " |")
    (output / "table_main_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    significance_rows = []
    for scene, _, _ in SCENES:
        for paired in data[scene]["analysis"]["paired"]:
            significance_rows.append({
                "难度": scene,
                "比较": "%s vs %s" % (
                    METHOD_LABELS[paired["reference"]],
                    METHOD_LABELS[paired["other"]]),
                "仅本文方法成功": paired["reference_only_success"],
                "仅对照成功": paired["other_only_success"],
                "共同成功": paired["common_successes"],
                "McNemar_p": "%.6f" % paired["mcnemar_exact_p"],
                "显著性": "显著" if paired["mcnemar_exact_p"] < 0.05 else "不显著",
            })
    sig_fields = ("难度", "比较", "仅本文方法成功", "仅对照成功", "共同成功",
                  "McNemar_p", "显著性")
    with (output / "table_paired_significance.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=sig_fields)
        writer.writeheader()
        writer.writerows(significance_rows)
    lines = ["| " + " | ".join(sig_fields) + " |",
             "| " + " | ".join(["---"] * len(sig_fields)) + " |"]
    for row in significance_rows:
        lines.append("| " + " | ".join(str(row[key]) for key in sig_fields) + " |")
    (output / "table_paired_significance.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def write_readme(output):
    text = """# 第三章论文图表

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
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--input", type=Path,
        default=repo / "assets" / "runs" / "navigation_benchmark")
    parser.add_argument(
        "--output", type=Path,
        default=repo / "assets" / "runs" / "navigation_benchmark" /
        "ch3_paper_figures_20260925")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_style()
    data = load_data(args.input)
    plot_success_rates(data, args.output)
    plot_outcomes(data, args.output)
    plot_quality_and_cost(data, args.output)
    plot_efficiency(data, args.output)
    draw_system_flow(args.output)
    draw_mppi_flow(args.output)
    draw_experiment_flow(args.output)
    write_tables(data, args.output)
    write_readme(args.output)
    print("Figures and tables: %s" % args.output)


if __name__ == "__main__":
    main()

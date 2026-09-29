"""Analyze paired Chapter 3 VIO robustness trials and generate paper figures."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


LEVELS = ("ideal", "mild", "medium", "severe")
LABELS = {"ideal": "理想VIO", "mild": "轻度", "medium": "中度", "severe": "重度"}
COLORS = {"ideal": "#2878B5", "mild": "#55A868", "medium": "#E39D26", "severe": "#C44E52"}


def configure_style():
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def wilson(successes, trials, z=1.959963984540054):
    rate = successes / float(trials)
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / trials +
                           z * z / (4.0 * trials * trials)) / denominator
    return center - radius, center + radius


def exact_mcnemar(a_only, b_only):
    total = a_only + b_only
    if total == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.factorial(total) //
               (math.factorial(i) * math.factorial(total - i))
               for i in range(k + 1)) / float(2 ** total)
    return min(1.0, 2.0 * tail)


def false_arrival_error(row):
    value = row.get("false_arrival_true_error_m")
    return float(value) if isinstance(value, (int, float)) else None


def category(row):
    if row.get("success") is True:
        return "success"
    reason = str(row.get("failure_reason", ""))
    if "false arrival" in reason:
        return "false_arrival"
    if "碰撞" in reason:
        return "collision"
    if "卡住" in reason:
        return "stuck"
    if "timeout" in reason.lower():
        return "timeout"
    return "other"


def load_level(root, level):
    directory = root / ("ch3_vio_%s_30seed_20260926" % level)
    records = []
    for path in sorted(directory.glob("mppi_seed*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["seed"] = int(path.stem.rsplit("seed", 1)[1])
        records.append(row)
    if len(records) != 30:
        raise ValueError("%s contains %d trials, expected 30" % (level, len(records)))
    return records


def analyze(root):
    grouped = {level: load_level(root, level) for level in LEVELS}
    expected_seeds = {row["seed"] for row in grouped["ideal"]}
    base_hash = {row["seed"]: row["obstacle_layout_hash"] for row in grouped["ideal"]}
    for level, rows in grouped.items():
        if {row["seed"] for row in rows} != expected_seeds:
            raise ValueError("unpaired seeds for %s" % level)
        if any(base_hash[row["seed"]] != row["obstacle_layout_hash"] for row in rows):
            raise ValueError("layout hash mismatch for %s" % level)

    levels = []
    lookups = {}
    for level in LEVELS:
        rows = grouped[level]
        lookups[level] = {row["seed"]: row for row in rows}
        successes = sum(row.get("success") is True for row in rows)
        false_errors = [false_arrival_error(row) for row in rows]
        false_errors = [value for value in false_errors if value is not None]
        within_020 = successes + sum(value <= 0.20 for value in false_errors)
        low, high = wilson(successes, len(rows))
        counts = {name: 0 for name in
                  ("success", "false_arrival", "collision", "stuck", "timeout", "other")}
        for row in rows:
            counts[category(row)] += 1
        success_rows = [row for row in rows if row.get("success") is True]
        levels.append({
            "level": level,
            "trials": len(rows),
            "strict_successes": successes,
            "strict_success_rate": successes / float(len(rows)),
            "strict_ci95_low": low,
            "strict_ci95_high": high,
            "within_020_count": within_020,
            "within_020_rate": within_020 / float(len(rows)),
            "false_arrivals": len(false_errors),
            "false_arrival_error_mean_m": statistics.mean(false_errors) if false_errors else None,
            "false_arrival_error_max_m": max(false_errors) if false_errors else None,
            "vio_position_rms_mean_m": statistics.mean(float(row["vio_position_rms_m"]) for row in rows),
            "vio_yaw_rms_mean_deg": statistics.mean(float(row["vio_yaw_rms_deg"]) for row in rows),
            "cmd_wz_variation_success_mean": statistics.mean(
                float(row["cmd_wz_variation"]) for row in success_rows) if success_rows else None,
            "planning_success_mean_ms": statistics.mean(
                float(row["planning_mean_ms"]) for row in success_rows) if success_rows else None,
            "outcomes": counts,
        })

    paired = []
    for level in LEVELS[1:]:
        ideal_only = level_only = 0
        ideal_020_only = level_020_only = 0
        for seed in sorted(expected_seeds):
            a = lookups["ideal"][seed]
            b = lookups[level][seed]
            a_ok = a.get("success") is True
            b_ok = b.get("success") is True
            ideal_only += int(a_ok and not b_ok)
            level_only += int(b_ok and not a_ok)
            a_020 = a_ok or ((false_arrival_error(a) or 999.0) <= 0.20)
            b_020 = b_ok or ((false_arrival_error(b) or 999.0) <= 0.20)
            ideal_020_only += int(a_020 and not b_020)
            level_020_only += int(b_020 and not a_020)
        paired.append({
            "reference": "ideal", "other": level,
            "ideal_only_strict": ideal_only, "other_only_strict": level_only,
            "strict_mcnemar_p": exact_mcnemar(ideal_only, level_only),
            "ideal_only_within_020": ideal_020_only,
            "other_only_within_020": level_020_only,
            "within_020_mcnemar_p": exact_mcnemar(ideal_020_only, level_020_only),
        })
    return {"paired_layout_hashes_verified": True, "levels": levels, "paired": paired}, grouped


def save(fig, output, stem):
    fig.savefig(str(output / (stem + ".png")), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(output / (stem + ".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_success(analysis, output):
    rows = analysis["levels"]
    x = np.arange(4)
    strict = np.array([row["strict_success_rate"] * 100 for row in rows])
    relaxed = np.array([row["within_020_rate"] * 100 for row in rows])
    low = np.array([row["strict_ci95_low"] * 100 for row in rows])
    high = np.array([row["strict_ci95_high"] * 100 for row in rows])
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars1 = ax.bar(x - 0.18, strict, 0.36, color=[COLORS[v] for v in LEVELS],
                   yerr=np.vstack((strict - low, high - strict)), capsize=3,
                   label="严格成功（真值误差≤0.15 m）")
    bars2 = ax.bar(x + 0.18, relaxed, 0.36, facecolor="white",
                   edgecolor=[COLORS[v] for v in LEVELS], linewidth=1.8,
                   label="终止时真值误差≤0.20 m")
    for bars, key in ((bars1, "strict_successes"), (bars2, "within_020_count")):
        for bar, row in zip(bars, rows):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    "%d/30" % row[key], ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[v] for v in LEVELS])
    ax.set_ylabel("比例（%）")
    ax.set_ylim(0, 100)
    ax.set_title("不同VIO噪声下的导航到达性能")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=False)
    save(fig, output, "16_vio_navigation_success")


def plot_pose_error(analysis, output):
    rows = analysis["levels"]
    x = np.arange(4)
    pos = [row["vio_position_rms_mean_m"] * 100 for row in rows]
    yaw = [row["vio_yaw_rms_mean_deg"] for row in rows]
    false_count = [row["false_arrivals"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    for ax, values, title, ylabel, color in (
            (axes[0], pos, "位置估计误差", "位置RMS（cm）", "#2878B5"),
            (axes[1], yaw, "航向估计误差", "航向RMS（deg）", "#E39D26"),
            (axes[2], false_count, "VIO误到达", "次数（/30）", "#C44E52")):
        bars = ax.bar(x, values, color=color, width=0.58)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[v] for v in LEVELS], rotation=15)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    "%.2f" % value if isinstance(value, float) else str(value),
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("VIO噪声注入的实际误差与终止风险", fontsize=13)
    fig.tight_layout()
    save(fig, output, "17_vio_pose_error_and_false_arrival")


def plot_outcomes(analysis, output):
    order = ("success", "false_arrival", "collision", "stuck", "timeout", "other")
    labels = {"success": "成功", "false_arrival": "误到达", "collision": "碰撞",
              "stuck": "卡住", "timeout": "超时", "other": "其他"}
    colors = {"success": "#2A9D6F", "false_arrival": "#D98E04",
              "collision": "#D55252", "stuck": "#7B61A8",
              "timeout": "#A7ADB4", "other": "#5B6770"}
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    y = np.arange(4)
    left = np.zeros(4)
    for name in order:
        values = np.array([row["outcomes"][name] for row in analysis["levels"]])
        bars = ax.barh(y, values, left=left, color=colors[name], height=0.58,
                       label=labels[name])
        for bar, value in zip(bars, values):
            if value >= 2:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2, str(int(value)),
                        ha="center", va="center", color="white", fontsize=8)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[v] for v in LEVELS])
    ax.set_xlabel("试验次数")
    ax.set_xlim(0, 30)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    ax.set_title("不同VIO噪声下的导航结果构成")
    fig.subplots_adjust(bottom=0.23)
    save(fig, output, "18_vio_outcome_composition")


def write_tables(analysis, output):
    fields = ["level", "trials", "strict_successes", "strict_success_rate",
              "strict_ci95_low", "strict_ci95_high", "within_020_count",
              "within_020_rate", "false_arrivals", "false_arrival_error_mean_m",
              "false_arrival_error_max_m", "vio_position_rms_mean_m",
              "vio_yaw_rms_mean_deg", "cmd_wz_variation_success_mean",
              "planning_success_mean_ms"]
    with (output / "vio_robustness_table.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in analysis["levels"])
    lines = ["| VIO档位 | 严格成功率 | 0.20m内 | 误到达 | 位置RMS | 航向RMS | 误到达最大误差 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in analysis["levels"]:
        lines.append("| %s | %d/30 (%.1f%%) | %d/30 | %d | %.3f m | %.2f° | %s |" % (
            LABELS[row["level"]], row["strict_successes"], 100 * row["strict_success_rate"],
            row["within_020_count"], row["false_arrivals"], row["vio_position_rms_mean_m"],
            row["vio_yaw_rms_mean_deg"],
            "-" if row["false_arrival_error_max_m"] is None else
            "%.3f m" % row["false_arrival_error_max_m"]))
    (output / "vio_robustness_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    parser.add_argument("--input", type=Path,
                        default=repo / "assets" / "runs" / "navigation_benchmark")
    parser.add_argument("--output", type=Path,
                        default=repo / "assets" / "runs" / "navigation_benchmark" /
                        "ch3_vio_robustness_20260926")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_style()
    analysis, _ = analyze(args.input)
    (args.output / "vio_statistical_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    write_tables(analysis, args.output)
    plot_success(analysis, args.output)
    plot_pose_error(analysis, args.output)
    plot_outcomes(analysis, args.output)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

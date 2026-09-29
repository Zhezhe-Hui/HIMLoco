"""Generate Chapter 3 external-method comparison tables and figures."""

import csv
import json
import math
from pathlib import Path
import shutil
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "assets" / "runs" / "navigation_benchmark"
OUTPUT = RUNS / "ch3_external_comparison_20260926"
PAPER_OUTPUT = RUNS / "ch3_paper_figures_20260925"
PAPER_STEMS = {
    "01_external_success_rate": "19_external_success_rate",
    "02_external_overall_success_and_planning": "20_external_overall_success_and_planning",
    "03_external_smoothness_and_time": "21_external_smoothness_and_time",
    "04_external_outcome_composition": "22_external_outcome_composition",
    "05_external_paired_mcnemar": "23_external_paired_mcnemar",
}
SCENES = (
    ("简单", "easy"),
    ("中等", "medium"),
    ("困难", "hard"),
)
METHODS = ("mppi", "iplanner", "ps_dwa", "sapf")
LABELS = {
    "mppi": "本文足迹MPPI",
    "iplanner": "iPlanner",
    "ps_dwa": "PS-DWA",
    "sapf": "SAPF",
}
COLORS = {
    "mppi": "#16856B",
    "iplanner": "#3977C3",
    "ps_dwa": "#D98B20",
    "sapf": "#C94C4C",
}
CATEGORY_COLORS = {
    "success": "#16856B",
    "collision": "#C94C4C",
    "stuck": "#7664A7",
    "timeout": "#9AA1A8",
    "other": "#4F5963",
}


def configure_style():
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
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
        "legend.frameon": False,
        "pdf.fonttype": 42,
    })


def load_directory(path, expected_method):
    records = []
    for result_path in sorted(path.glob("*_seed*.json")):
        method, seed_text = result_path.stem.rsplit("_seed", 1)
        if method != expected_method:
            continue
        record = json.loads(result_path.read_text(encoding="utf-8"))
        record.update(method=method, seed=int(seed_text))
        records.append(record)
    if len(records) != 30:
        raise ValueError("%s: expected 30 %s trials, got %d" % (
            path, expected_method, len(records)))
    return records


def load_data():
    data = {}
    for scene_label, scene_id in SCENES:
        external = RUNS / ("ch3_external_%s_30seed_20260926" % scene_id)
        mppi = RUNS / ("ch3_external_mppi_%s_30seed_20260926" % scene_id)
        data[scene_label] = {"mppi": load_directory(mppi, "mppi")}
        for method in METHODS[1:]:
            data[scene_label][method] = load_directory(external, method)

        expected_seeds = {r["seed"] for r in data[scene_label]["mppi"]}
        for method in METHODS:
            records = data[scene_label][method]
            if {r["seed"] for r in records} != expected_seeds:
                raise ValueError("unpaired seeds: %s %s" % (scene_label, method))
        for seed in expected_seeds:
            hashes = {
                r.get("obstacle_layout_hash")
                for method in METHODS for r in data[scene_label][method]
                if r["seed"] == seed
            }
            if len(hashes) != 1 or None in hashes:
                raise ValueError("layout mismatch: %s seed=%d" % (scene_label, seed))
    return data


def wilson(successes, trials, z=1.959963984540054):
    rate = successes / float(trials)
    denom = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denom
    radius = z * math.sqrt(
        rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)) / denom
    return center - radius, center + radius


def mcnemar_exact(reference_only, other_only):
    discordant = reference_only + other_only
    if not discordant:
        return 1.0
    tail = sum(
        math.factorial(discordant) //
        (math.factorial(k) * math.factorial(discordant - k))
        for k in range(min(reference_only, other_only) + 1)
    ) / float(2 ** discordant)
    return min(1.0, 2.0 * tail)


def failure_category(record):
    if record.get("success") is True:
        return "success"
    reason = str(record.get("failure_reason", ""))
    if "碰撞" in reason:
        return "collision"
    if "卡住" in reason:
        return "stuck"
    if "timeout" in reason.lower():
        return "timeout"
    return "other"


def summarize(records):
    successes = [r for r in records if r.get("success") is True]
    low, high = wilson(len(successes), len(records))
    row = {
        "trials": len(records),
        "successes": len(successes),
        "success_rate": len(successes) / float(len(records)),
        "success_ci95_low": low,
        "success_ci95_high": high,
        "planning_all_mean_ms": statistics.mean(
            float(r["planning_mean_ms"]) for r in records),
    }
    metrics = (
        "elapsed_s", "distance_m", "path_smoothness_mean",
        "cmd_wz_variation", "cmd_wz_sign_flips", "energy_torque_sq_mean",
    )
    for metric in metrics:
        values = [float(r[metric]) for r in successes
                  if isinstance(r.get(metric), (int, float))]
        row[metric + "_mean"] = statistics.mean(values) if values else None
        row[metric + "_std"] = statistics.pstdev(values) if len(values) > 1 else (
            0.0 if values else None)
    row["outcomes"] = {
        category: sum(failure_category(r) == category for r in records)
        for category in ("success", "collision", "stuck", "timeout", "other")
    }
    return row


def build_analysis(data):
    scene_stats = {
        scene: {method: summarize(data[scene][method]) for method in METHODS}
        for scene, _ in SCENES
    }
    overall_records = {
        method: [r for scene, _ in SCENES for r in data[scene][method]]
        for method in METHODS
    }
    overall = {method: summarize(records)
               for method, records in overall_records.items()}

    paired = []
    reference_lookup = {
        (scene, r["seed"]): r for scene, _ in SCENES for r in data[scene]["mppi"]
    }
    for method in METHODS[1:]:
        other_lookup = {
            (scene, r["seed"]): r for scene, _ in SCENES for r in data[scene][method]
        }
        reference_only = other_only = 0
        common = []
        for key, reference in reference_lookup.items():
            other = other_lookup[key]
            ref_success = reference.get("success") is True
            other_success = other.get("success") is True
            reference_only += int(ref_success and not other_success)
            other_only += int(other_success and not ref_success)
            if ref_success and other_success:
                common.append((reference, other))
        paired_metrics = {}
        for metric in ("elapsed_s", "distance_m", "path_smoothness_mean",
                       "cmd_wz_variation", "energy_torque_sq_mean"):
            pairs = [(float(a[metric]), float(b[metric])) for a, b in common
                     if isinstance(a.get(metric), (int, float))
                     and isinstance(b.get(metric), (int, float))]
            if not pairs:
                continue
            try:
                p_value = float(wilcoxon(
                    [p[0] for p in pairs], [p[1] for p in pairs]).pvalue)
            except ValueError:
                p_value = None
            paired_metrics[metric] = {
                "n": len(pairs),
                "mppi_mean": statistics.mean(p[0] for p in pairs),
                "other_mean": statistics.mean(p[1] for p in pairs),
                "wilcoxon_p": p_value,
            }
        paired.append({
            "reference": "mppi",
            "other": method,
            "mppi_only_success": reference_only,
            "other_only_success": other_only,
            "both_success": len(common),
            "mcnemar_exact_p": mcnemar_exact(reference_only, other_only),
            "metrics": paired_metrics,
        })
    return scene_stats, overall, paired


def save_figure(fig, stem):
    fig.savefig(OUTPUT / (stem + ".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / (stem + ".pdf"), bbox_inches="tight", facecolor="white")
    paper_stem = PAPER_STEMS[stem]
    fig.savefig(PAPER_OUTPUT / (paper_stem + ".png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(PAPER_OUTPUT / (paper_stem + ".pdf"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_success(scene_stats):
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    x = np.arange(len(SCENES), dtype=float)
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHODS))
    for method, offset in zip(METHODS, offsets):
        rows = [scene_stats[scene][method] for scene, _ in SCENES]
        rates = np.asarray([r["success_rate"] * 100.0 for r in rows])
        low = np.asarray([r["success_ci95_low"] * 100.0 for r in rows])
        high = np.asarray([r["success_ci95_high"] * 100.0 for r in rows])
        bars = ax.bar(x + offset, rates, width, color=COLORS[method],
                      label=LABELS[method], yerr=np.vstack((rates-low, high-rates)),
                      capsize=2.5, edgecolor="white", linewidth=0.7)
        for bar, row in zip(bars, rows):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    "%d/30" % row["successes"], ha="center", va="bottom", fontsize=8)
    ax.set_title("外部方法对比：三种难度下的导航成功率")
    ax.set_ylabel("成功率（%）")
    ax.set_xticks(x)
    ax.set_xticklabels([scene for scene, _ in SCENES])
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.text(0.01, 0.98, "误差线：Wilson 95%置信区间；每组30个配对seed",
            transform=ax.transAxes, va="top", fontsize=8, color="#4F5963")
    fig.subplots_adjust(bottom=0.23)
    save_figure(fig, "01_external_success_rate")


def plot_overall(overall):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    x = np.arange(len(METHODS))
    rates = np.asarray([overall[m]["success_rate"] * 100 for m in METHODS])
    lows = np.asarray([overall[m]["success_ci95_low"] * 100 for m in METHODS])
    highs = np.asarray([overall[m]["success_ci95_high"] * 100 for m in METHODS])
    bars = axes[0].bar(x, rates, color=[COLORS[m] for m in METHODS],
                       yerr=np.vstack((rates-lows, highs-rates)), capsize=3)
    for bar, method in zip(bars, METHODS):
        row = overall[method]
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                     "%d/90" % row["successes"], ha="center", fontsize=9)
    axes[0].set_title("三档合计成功率")
    axes[0].set_ylabel("成功率（%）")
    axes[0].set_ylim(0, 82)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([LABELS[m] for m in METHODS], rotation=18)
    axes[0].grid(axis="y", alpha=0.25)

    planning = [overall[m]["planning_all_mean_ms"] for m in METHODS]
    bars = axes[1].bar(x, planning, color=[COLORS[m] for m in METHODS])
    for bar, value in zip(bars, planning):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                     "%.2f" % value, ha="center", va="bottom", fontsize=9)
    axes[1].axhline(80.0, color="#7A7A7A", linestyle="--", linewidth=1,
                    label="12.5 Hz规划周期（80 ms）")
    axes[1].set_title("同机全样本平均规划耗时")
    axes[1].set_ylabel("规划耗时（ms）")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABELS[m] for m in METHODS], rotation=18)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("外部方法总体性能（RTX 4060，同一代码与seed）", y=1.02)
    fig.tight_layout()
    save_figure(fig, "02_external_overall_success_and_planning")


def plot_quality(scene_stats):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    x = np.arange(len(SCENES), dtype=float)
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHODS))
    for method, offset in zip(METHODS, offsets):
        smooth = [scene_stats[s][method]["path_smoothness_mean_mean"]
                  for s, _ in SCENES]
        elapsed = [scene_stats[s][method]["elapsed_s_mean"] for s, _ in SCENES]
        axes[0].bar(x+offset, [np.nan if v is None else v for v in smooth], width,
                    color=COLORS[method], label=LABELS[method])
        axes[1].bar(x+offset, [np.nan if v is None else v for v in elapsed], width,
                    color=COLORS[method], label=LABELS[method])
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([s for s, _ in SCENES])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("成功轨迹平均转角变化")
    axes[0].set_ylabel("路径平滑度（rad，越低越好）")
    axes[1].set_title("成功样本平均到达时间")
    axes[1].set_ylabel("时间（s，越低越好）")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("外部方法成功样本的轨迹质量与效率", y=1.02)
    fig.subplots_adjust(bottom=0.20, wspace=0.25)
    save_figure(fig, "03_external_smoothness_and_time")


def plot_outcomes(scene_stats):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4), sharey=True)
    categories = ("success", "collision", "stuck", "timeout", "other")
    category_labels = {"success": "成功", "collision": "碰撞", "stuck": "卡住",
                       "timeout": "超时", "other": "其他"}
    for ax, (scene, _) in zip(axes, SCENES):
        y = np.arange(len(METHODS))
        left = np.zeros(len(METHODS))
        for category in categories:
            values = np.asarray([
                scene_stats[scene][m]["outcomes"][category] for m in METHODS])
            bars = ax.barh(y, values, left=left, height=0.58,
                           color=CATEGORY_COLORS[category], label=category_labels[category])
            for bar, value in zip(bars, values):
                if value >= 3:
                    ax.text(bar.get_x()+bar.get_width()/2,
                            bar.get_y()+bar.get_height()/2, str(int(value)),
                            ha="center", va="center", color="white", fontsize=8)
            left += values
        ax.set_title(scene)
        ax.set_xlim(0, 30)
        ax.set_xlabel("试验次数")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(np.arange(len(METHODS)))
    axes[0].set_yticklabels([LABELS[m] for m in METHODS])
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("外部方法对比的成功与失败构成", y=1.02)
    fig.subplots_adjust(bottom=0.18, wspace=0.12)
    save_figure(fig, "04_external_outcome_composition")


def plot_paired(paired):
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    methods = [row["other"] for row in paired]
    values = np.asarray([[row["mppi_only_success"], row["other_only_success"]]
                         for row in paired])
    y = np.arange(len(methods))
    limit = max(1, int(values.max()) + 3)
    ax.barh(y, values[:, 0], color=COLORS["mppi"], label="仅本文方法成功")
    ax.barh(y, -values[:, 1], color="#537AA5",
            label="仅对比方法成功")
    for index, row in enumerate(paired):
        ax.text(values[index, 0] + 0.3, index, str(values[index, 0]), va="center")
        ax.text(-values[index, 1] - 0.3, index, str(values[index, 1]),
                va="center", ha="right")
        ax.text(-limit + 1, index + 0.28,
                "McNemar p=%.4g" % row["mcnemar_exact_p"],
                ha="left", va="bottom", fontsize=8, color="#4F5963")
    ax.set_xlim(-limit, limit)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[m] for m in methods])
    ax.set_xlabel("不一致配对数（右：本文方法占优；左：对比方法占优）")
    ax.set_title("三档90组配对试验的成功性差异")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.subplots_adjust(bottom=0.23)
    save_figure(fig, "05_external_paired_mcnemar")


def write_outputs(scene_stats, overall, paired):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene, _ in SCENES:
        for method in METHODS:
            row = {"difficulty": scene, "method": method,
                   "method_display_name": LABELS[method]}
            row.update({k: v for k, v in scene_stats[scene][method].items()
                        if k != "outcomes"})
            row.update({"outcome_" + k: v
                        for k, v in scene_stats[scene][method]["outcomes"].items()})
            rows.append(row)
    with (OUTPUT / "external_comparison_by_difficulty.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    overall_rows = []
    for method in METHODS:
        row = {"method": method, "method_display_name": LABELS[method]}
        row.update({k: v for k, v in overall[method].items() if k != "outcomes"})
        row.update({"outcome_" + k: v for k, v in overall[method]["outcomes"].items()})
        overall_rows.append(row)
    with (OUTPUT / "external_comparison_overall.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(overall_rows[0]))
        writer.writeheader()
        writer.writerows(overall_rows)

    analysis = {
        "protocol": {
            "paired_seeds_per_difficulty": 30,
            "seed_range": [20261000, 20261029],
            "gpu": "NVIDIA GeForce RTX 4060 8GB",
            "vio": "ideal",
            "same_layout_hash_verified": True,
            "efficiency_metrics_use_successful_trials": True,
            "planning_time_uses_all_trials": True,
        },
        "by_difficulty": scene_stats,
        "overall": overall,
        "paired_against_mppi": paired,
    }
    (OUTPUT / "external_comparison_statistics.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PAPER_OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    scene_stats, overall, paired = build_analysis(data)
    write_outputs(scene_stats, overall, paired)
    plot_success(scene_stats)
    plot_overall(overall)
    plot_quality(scene_stats)
    plot_outcomes(scene_stats)
    plot_paired(paired)
    for name in ("external_comparison_by_difficulty.csv",
                 "external_comparison_overall.csv",
                 "external_comparison_statistics.json"):
        shutil.copy2(OUTPUT / name, PAPER_OUTPUT / name)
    print("External comparison figures: %s" % OUTPUT)


if __name__ == "__main__":
    main()

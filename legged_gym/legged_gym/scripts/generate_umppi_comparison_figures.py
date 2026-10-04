"""Generate Chapter 3 external-method comparison tables and figures."""

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
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "assets" / "runs" / "navigation_benchmark"
OUTPUT = RUNS / "ch3_umppi_external_comparison_20261003"
SCENES = (
    ("Easy", "easy"),
    ("Medium", "medium"),
    ("Hard", "hard"),
)
METHODS = ("umppi", "iplanner", "ps_dwa", "sapf")
LABELS = {
    "umppi": "Ours",
    "iplanner": "iPlanner",
    "ps_dwa": "PS-DWA",
    "sapf": "SAPF",
}
COLORS = {
    "umppi": "#D62728",
    "iplanner": "#3977C3",
    "ps_dwa": "#D98B20",
    "sapf": "#607D8B",
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
    layout_mismatches = []
    for scene_label, scene_id in SCENES:
        external = RUNS / ("ch3_external_%s_30seed_20260926" % scene_id)
        umppi = RUNS / ("ch3_external_umppi_%s_30seed_20261003" % scene_id)
        data[scene_label] = {"umppi": load_directory(umppi, "umppi")}
        for method in METHODS[1:]:
            data[scene_label][method] = load_directory(external, method)

        expected_seeds = {r["seed"] for r in data[scene_label]["umppi"]}
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
                layout_mismatches.append({
                    "difficulty": scene_label,
                    "seed": seed,
                    "hashes": sorted(str(value) for value in hashes),
                })
    return data, layout_mismatches


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


def build_analysis(data, paired_tests_valid):
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
        (scene, r["seed"]): r for scene, _ in SCENES for r in data[scene]["umppi"]
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
            p_value = None
            if paired_tests_valid:
                try:
                    p_value = float(wilcoxon(
                        [p[0] for p in pairs], [p[1] for p in pairs]).pvalue)
                except ValueError:
                    p_value = None
            paired_metrics[metric] = {
                "n": len(pairs),
                "umppi_mean": statistics.mean(p[0] for p in pairs),
                "other_mean": statistics.mean(p[1] for p in pairs),
                "wilcoxon_p": p_value,
            }
        paired.append({
            "reference": "umppi",
            "other": method,
            "umppi_only_success": reference_only,
            "other_only_success": other_only,
            "both_success": len(common),
            "mcnemar_exact_p": (
                mcnemar_exact(reference_only, other_only)
                if paired_tests_valid else None),
            "metrics": paired_metrics,
        })
    return scene_stats, overall, paired


def save_figure(fig, stem):
    fig.savefig(OUTPUT / (stem + ".png"), dpi=300, bbox_inches="tight", facecolor="white")
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
    ax.set_title("Navigation Success Rate Across Terrain Difficulties")
    ax.set_ylabel("Success rate [%]")
    ax.set_xticks(x)
    ax.set_xticklabels([scene for scene, _ in SCENES])
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.text(0.01, 0.98, "Wilson 95% confidence interval; 30 fixed seeds per group",
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
    axes[0].set_title("Overall Success Rate")
    axes[0].set_ylabel("Success rate [%]")
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
                    label="12.5 Hz planning budget: 80 ms")
    axes[1].set_title("Mean Planning Time")
    axes[1].set_ylabel("Planning time [ms]")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABELS[m] for m in METHODS], rotation=18)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Overall Planner Performance", y=1.02)
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
    axes[0].set_title("Path Smoothness")
    axes[0].set_ylabel("Path smoothness [rad]")
    axes[1].set_title("Travel Time")
    axes[1].set_ylabel("Time [s]")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Trajectory Quality and Efficiency on Successful Trials", y=1.02)
    fig.subplots_adjust(bottom=0.20, wspace=0.25)
    save_figure(fig, "03_external_smoothness_and_time")


def plot_outcomes(scene_stats):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4), sharey=True)
    categories = ("success", "collision", "stuck", "timeout", "other")
    category_labels = {"success": "Success", "collision": "Collision",
                       "stuck": "Stuck", "timeout": "Timeout", "other": "Other"}
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
        ax.set_xlabel("Trials")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(np.arange(len(METHODS)))
    axes[0].set_yticklabels([LABELS[m] for m in METHODS])
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Outcome Composition Across Methods", y=1.02)
    fig.subplots_adjust(bottom=0.18, wspace=0.12)
    save_figure(fig, "04_external_outcome_composition")


def plot_paired(paired, paired_tests_valid):
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    methods = [row["other"] for row in paired]
    values = np.asarray([[row["umppi_only_success"], row["other_only_success"]]
                         for row in paired])
    y = np.arange(len(methods))
    limit = max(1, int(values.max()) + 3)
    ax.barh(y, values[:, 0], color=COLORS["umppi"], label="Ours only")
    ax.barh(y, -values[:, 1], color="#537AA5",
            label="Baseline only")
    for index, row in enumerate(paired):
        ax.text(values[index, 0] + 0.3, index, str(values[index, 0]), va="center")
        ax.text(-values[index, 1] - 0.3, index, str(values[index, 1]),
                va="center", ha="right")
        if paired_tests_valid:
            annotation = "McNemar p=%.4g" % row["mcnemar_exact_p"]
        else:
            annotation = "Different scene versions; descriptive only"
        ax.text(-limit + 1, index + 0.28, annotation,
                ha="left", va="bottom", fontsize=8, color="#4F5963")
    ax.set_xlim(-limit, limit)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[m] for m in methods])
    ax.set_xlabel("Discordant outcomes: right favors Ours; left favors baseline")
    ax.set_title("Seed-Aligned Success Differences Across 90 Trials")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.subplots_adjust(bottom=0.23)
    save_figure(fig, "05_external_paired_mcnemar")


def write_outputs(scene_stats, overall, paired, layout_mismatches):
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
            "same_layout_hash_verified": not layout_mismatches,
            "paired_tests_valid": not layout_mismatches,
            "comparison_scope": (
                "strict paired comparison" if not layout_mismatches else
                "descriptive comparison against 2026-09-26 historical baselines"
            ),
            "layout_mismatch_count": len(layout_mismatches),
            "efficiency_metrics_use_successful_trials": True,
            "planning_time_uses_all_trials": True,
        },
        "by_difficulty": scene_stats,
        "overall": overall,
        "paired_against_umppi": paired,
        "layout_mismatches": layout_mismatches,
    }
    (OUTPUT / "external_comparison_statistics.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data, layout_mismatches = load_data()
    paired_tests_valid = not layout_mismatches
    scene_stats, overall, paired = build_analysis(data, paired_tests_valid)
    write_outputs(scene_stats, overall, paired, layout_mismatches)
    plot_success(scene_stats)
    plot_overall(overall)
    plot_quality(scene_stats)
    plot_outcomes(scene_stats)
    plot_paired(paired, paired_tests_valid)
    print("External comparison figures: %s" % OUTPUT)


if __name__ == "__main__":
    main()

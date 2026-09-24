"""Analyze paired navigation trials and emit paper-ready statistics."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from scipy.stats import wilcoxon


EFFICIENCY_METRICS = (
    "elapsed_s",
    "distance_m",
    "cmd_wz_variation",
    "cmd_wz_sign_flips",
    "planning_mean_ms",
    "energy_torque_sq_mean",
    "min_path_clearance_m",
)


def wilson_interval(successes, trials, z=1.959963984540054):
    if trials <= 0:
        return None, None
    rate = successes / float(trials)
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return center - radius, center + radius


def exact_mcnemar_p(reference_only, other_only):
    discordant = reference_only + other_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(
        min(reference_only, other_only) + 1)) / float(2 ** discordant)
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
    if "camera_unavailable" in reason:
        return "camera_unavailable"
    return "other"


def load_trials(root):
    records = []
    for path in sorted(root.glob("*_seed*.json")):
        method, seed_text = path.stem.rsplit("_seed", 1)
        record = json.loads(path.read_text(encoding="utf-8"))
        record.update(method=method, seed=int(seed_text))
        records.append(record)
    return records


def analyze(records, reference="mppi"):
    methods = sorted({record["method"] for record in records})
    grouped = {
        method: sorted(
            (record for record in records if record["method"] == method),
            key=lambda item: item["seed"])
        for method in methods
    }
    if reference not in grouped:
        raise ValueError("reference method not found: %s" % reference)
    reference_seeds = {record["seed"] for record in grouped[reference]}
    for method, method_records in grouped.items():
        seeds = {record["seed"] for record in method_records}
        if seeds != reference_seeds:
            raise ValueError("unpaired seed set for %s" % method)

    for seed in sorted(reference_seeds):
        hashes = {
            record.get("obstacle_layout_hash")
            for method_records in grouped.values()
            for record in method_records
            if record["seed"] == seed and record.get("obstacle_layout_hash")
        }
        if len(hashes) > 1:
            raise ValueError("obstacle layout mismatch for seed %s" % seed)

    method_stats = []
    lookup = {}
    for method, method_records in grouped.items():
        lookup[method] = {record["seed"]: record for record in method_records}
        successes = [record for record in method_records if record.get("success") is True]
        low, high = wilson_interval(len(successes), len(method_records))
        categories = {}
        for record in method_records:
            category = failure_category(record)
            categories[category] = categories.get(category, 0) + 1
        row = {
            "method": method,
            "trials": len(method_records),
            "successes": len(successes),
            "success_rate": len(successes) / float(len(method_records)),
            "success_ci95_low": low,
            "success_ci95_high": high,
            "failure_categories": categories,
        }
        for metric in EFFICIENCY_METRICS:
            values = [float(record[metric]) for record in successes
                      if isinstance(record.get(metric), (int, float))]
            row[metric + "_mean"] = statistics.fmean(values) if values else None
            row[metric + "_std"] = statistics.pstdev(values) if len(values) > 1 else (
                0.0 if values else None)
        method_stats.append(row)

    paired = []
    for other in methods:
        if other == reference:
            continue
        reference_only = 0
        other_only = 0
        common_success = []
        for seed in sorted(reference_seeds):
            ref_record = lookup[reference][seed]
            other_record = lookup[other][seed]
            ref_success = ref_record.get("success") is True
            other_success = other_record.get("success") is True
            reference_only += int(ref_success and not other_success)
            other_only += int(other_success and not ref_success)
            if ref_success and other_success:
                common_success.append((ref_record, other_record))
        comparison = {
            "reference": reference,
            "other": other,
            "reference_only_success": reference_only,
            "other_only_success": other_only,
            "common_successes": len(common_success),
            "mcnemar_exact_p": exact_mcnemar_p(reference_only, other_only),
            "metrics": {},
        }
        for metric in EFFICIENCY_METRICS:
            pairs = [(float(ref[metric]), float(candidate[metric]))
                     for ref, candidate in common_success
                     if isinstance(ref.get(metric), (int, float))
                     and isinstance(candidate.get(metric), (int, float))]
            if not pairs:
                continue
            reference_values = [pair[0] for pair in pairs]
            other_values = [pair[1] for pair in pairs]
            try:
                p_value = float(wilcoxon(reference_values, other_values).pvalue)
            except ValueError:
                p_value = None
            comparison["metrics"][metric] = {
                "n": len(pairs),
                "reference_mean": statistics.fmean(reference_values),
                "other_mean": statistics.fmean(other_values),
                "mean_difference": statistics.fmean(
                    ref - candidate for ref, candidate in pairs),
                "wilcoxon_p": p_value,
            }
        paired.append(comparison)
    return {
        "reference": reference,
        "paired_layout_hashes_verified": all(
            record.get("obstacle_layout_hash") for record in records),
        "methods": method_stats,
        "paired": paired,
    }


def write_outputs(analysis, root):
    (root / "statistical_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    table = []
    for item in analysis["methods"]:
        table.append({
            "method": item["method"],
            "successes": item["successes"],
            "trials": item["trials"],
            "success_rate": item["success_rate"],
            "success_ci95_low": item["success_ci95_low"],
            "success_ci95_high": item["success_ci95_high"],
            "elapsed_s_mean": item["elapsed_s_mean"],
            "distance_m_mean": item["distance_m_mean"],
            "cmd_wz_variation_mean": item["cmd_wz_variation_mean"],
            "planning_mean_ms": item["planning_mean_ms_mean"],
            "failure_categories": json.dumps(
                item["failure_categories"], ensure_ascii=False, sort_keys=True),
        })
    with (root / "paper_table.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference", default="mppi")
    args = parser.parse_args()
    records = load_trials(args.directory)
    if not records:
        raise RuntimeError("no *_seed*.json trials found")
    analysis = analyze(records, reference=args.reference)
    write_outputs(analysis, args.directory)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Run paired, one-process-per-seed navigation benchmarks and aggregate CSV."""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime


METHODS = {
    "fmm_fixed": {
        "HIMLOCO_PLANNER_MODE": "fmm",
        "HIMLOCO_FMM_SOFT_COST": "0",
        "HIMLOCO_TEMPORAL_MEMORY": "0",
    },
    "fmm_soft": {
        "HIMLOCO_PLANNER_MODE": "fmm",
        "HIMLOCO_FMM_SOFT_COST": "1",
        "HIMLOCO_TEMPORAL_MEMORY": "0",
    },
    "mppi": {
        "HIMLOCO_PLANNER_MODE": "mppi",
        "HIMLOCO_FMM_SOFT_COST": "1",
        "HIMLOCO_TEMPORAL_MEMORY": "0",
    },
    "mppi_memory": {
        "HIMLOCO_PLANNER_MODE": "mppi",
        "HIMLOCO_FMM_SOFT_COST": "1",
        "HIMLOCO_TEMPORAL_MEMORY": "1",
    },
    "astar_dwa": {
        "HIMLOCO_PLANNER_MODE": "astar_dwa",
        "HIMLOCO_TEMPORAL_MEMORY": "0",
    },
}

CAMERA_UNAVAILABLE_MARKER = "headless 下无法创建相机"


SUMMARY_METRICS = {
    "elapsed_s": "elapsed_s",
    "distance_m": "distance_m",
    "planning_mean_ms": "planning_time_ms",
    "planning_p95_ms": "planning_p95_ms",
    "path_smoothness_mean": "path_smoothness",
    "cmd_wz_variation": "cmd_wz_variation",
    "cmd_wz_sign_flips": "cmd_wz_sign_flips",
    "energy_torque_sq_mean": "energy_torque_sq",
    "body_contact_steps": "body_contact_steps",
    "side_contact_steps": "side_contact_steps",
    "stall_escape_count": "stall_escape_count",
    "min_path_clearance_m": "min_path_clearance_m",
}


def write_summary(rows, out_dir):
    summaries = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        successful = [row for row in method_rows if row.get("success") is True]
        failures = {}
        for row in method_rows:
            if row.get("success") is not True:
                reason = str(row.get("failure_reason", "unknown"))
                failures[reason] = failures.get(reason, 0) + 1
        summary = {
            "method": method,
            "trials": len(method_rows),
            "successes": len(successful),
            "success_rate": len(successful) / float(len(method_rows)),
            "failure_reasons": json.dumps(
                failures, ensure_ascii=False, sort_keys=True),
        }
        for metric, output_name in SUMMARY_METRICS.items():
            # Efficiency/smoothness only describe completed routes. Safety and
            # recovery counts include every trial so failures are not hidden.
            source = method_rows if metric in {
                "body_contact_steps", "side_contact_steps", "stall_escape_count"
            } else successful
            values = [float(row[metric]) for row in source
                      if isinstance(row.get(metric), (int, float))]
            summary[output_name + "_mean"] = (
                statistics.fmean(values) if values else None)
            summary[output_name + "_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
                if values else None)
        summaries.append(summary)

    fields = sorted({key for row in summaries for key in row})
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    (out_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


def load_resume_records(out_dir):
    """Load only trials that were completely persisted before interruption."""
    trials_path = out_dir / "trials.csv"
    if not trials_path.exists():
        return {}
    with trials_path.open(encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    records = {}
    for row in csv_rows:
        method = row.get("method", "")
        try:
            seed = int(row["seed"])
            returncode = int(row["process_returncode"])
        except (KeyError, TypeError, ValueError):
            continue
        result_path = out_dir / ("%s_seed%d.json" % (method, seed))
        log_path = out_dir / ("%s_seed%d.log" % (method, seed))
        if returncode != 0 or not result_path.exists() or not log_path.exists():
            continue
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if CAMERA_UNAVAILABLE_MARKER in log_text:
            continue
        try:
            record = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        record.update({
            "method": method,
            "seed": seed,
            "process_returncode": returncode,
        })
        records[(method, seed)] = record
    return records


def validate_resume_manifest(existing, expected):
    keys = ("scene", "methods", "seed_start", "num_seeds", "vio")
    mismatches = [key for key in keys if existing.get(key) != expected.get(key)]
    if mismatches:
        raise ValueError(
            "resume configuration mismatch: %s" % ", ".join(mismatches))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="random_corridor")
    parser.add_argument("--methods", nargs="+", default=["fmm_fixed", "mppi"],
                        choices=sorted(METHODS))
    parser.add_argument("--seed-start", type=int, default=20261000)
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--vio", default="ideal",
                        choices=["ideal", "mild", "medium", "severe"])
    parser.add_argument("--output", default="")
    parser.add_argument("--resume", action="store_true",
                        help="skip valid completed trials in the output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[3]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output).expanduser().resolve() if args.output else (
        repo / "assets" / "runs" / "navigation_benchmark" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(repo),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, check=False).stdout.strip()
    git_diff = subprocess.run(
        ["git", "diff", "--binary"], cwd=str(repo),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, check=False).stdout
    manifest = {
        "scene": args.scene,
        "methods": args.methods,
        "seed_start": args.seed_start,
        "num_seeds": args.num_seeds,
        "vio": args.vio,
        "python": sys.executable,
        "git_commit": git_commit,
        "git_dirty": bool(git_diff),
    }
    config_path = out_dir / "config.json"
    if args.resume and config_path.exists():
        existing_manifest = json.loads(config_path.read_text(encoding="utf-8"))
        validate_resume_manifest(existing_manifest, manifest)
        completed_records = load_resume_records(out_dir)
    else:
        completed_records = {}
        config_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "git_diff.patch").write_text(git_diff, encoding="utf-8")

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        for method in args.methods:
            run_name = "%s_seed%d" % (method, seed)
            result_path = out_dir / (run_name + ".json")
            log_path = out_dir / (run_name + ".log")
            if (method, seed) in completed_records:
                rows.append(completed_records[(method, seed)])
                print("[%d/%d] skip completed %s seed=%d" % (
                    len(rows), args.num_seeds * len(args.methods), method, seed),
                    flush=True)
                continue
            env = os.environ.copy()
            env.update({
                "HIMLOCO_PRESET": "nav_only",
                "HIMLOCO_SCENE_PRESET": args.scene,
                "HIMLOCO_SCENE_SEED": str(seed),
                "HIMLOCO_NUM_TRIALS": "1",
                "HIMLOCO_VIO_NOISE": args.vio,
                "HIMLOCO_RESULT_JSON": str(result_path),
            })
            env.update(METHODS[method])
            # Isaac Gym in this project sets graphics_device_id=-1 for --headless,
            # which makes camera creation fail. Visual navigation benchmarks must
            # therefore keep a graphics device even when no human watches them.
            command = [sys.executable, "legged_gym/scripts/play1.py", "--task=go2"]
            print("[%d/%d] %s seed=%d" % (
                len(rows) + 1, args.num_seeds * len(args.methods), method, seed),
                flush=True)
            completed = subprocess.run(
                command, cwd=str(repo / "legged_gym"), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, check=False)
            log_path.write_text(completed.stdout, encoding="utf-8")
            camera_unavailable = CAMERA_UNAVAILABLE_MARKER in completed.stdout
            if result_path.exists():
                record = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                record = {"success": False, "failure_reason": "process_error"}
            if camera_unavailable:
                record.update({
                    "success": False,
                    "failure_reason": "camera_unavailable",
                })
            record.update({
                "method": method,
                "seed": seed,
                "process_returncode": completed.returncode,
            })
            rows.append(record)

            fields = sorted({key for row in rows for key in row})
            with (out_dir / "trials.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            write_summary(rows, out_dir)
            if camera_unavailable:
                raise RuntimeError(
                    "Depth camera is unavailable. Do not run the visual navigation "
                    "benchmark with --headless/graphics_device_id=-1.")

    print("Results: %s" % out_dir)


if __name__ == "__main__":
    main()

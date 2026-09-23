"""Offline ablation for fixed-inflation FMM and the dual-layer cost field."""

import argparse
import csv
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from legged_gym.envs.go2.bev_height_mapper import FMMGradientController


RES = 0.1
SIZE_M = 6.0
CELLS = int(SIZE_M / RES)
GOAL_XY = (5.0, 0.0)


def make_scenarios():
    central_block = np.zeros((CELLS, CELLS), dtype=np.uint8)
    central_block[12:30, 27:34] = 1

    narrow_gate = np.zeros((CELLS, CELLS), dtype=np.uint8)
    narrow_gate[28:31, :27] = 1
    narrow_gate[28:31, 34:] = 1
    return {
        "central_block": central_block,
        "narrow_gate_0p7m": narrow_gate,
    }


METHODS = {
    "fixed_0p20": dict(use_soft_cost=False, inflate_radius_m=0.2, hard_radius_m=0.2),
    "fixed_0p40": dict(use_soft_cost=False, inflate_radius_m=0.4, hard_radius_m=0.4),
    "dual_layer": dict(use_soft_cost=True, inflate_radius_m=0.2, hard_radius_m=0.2),
}


def run_case(occ, method):
    controller = FMMGradientController(
        bev_res=RES,
        bev_x=SIZE_M,
        bev_y=SIZE_M,
        soft_clearance_m=0.8,
        soft_cost_weight=6.0,
        soft_cost_power=2.0,
        **METHODS[method],
    )
    controller.set_goal(GOAL_XY)
    started = time.perf_counter()
    controller.update(occ)
    path = controller.extract_path(max_steps=200)
    planning_ms = (time.perf_counter() - started) * 1000.0

    clearance = distance_transform_edt(occ == 0) * RES
    clearances = np.asarray([clearance[r, c] for r, c in path], dtype=np.float64)
    path_length = 0.0
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        path_length += np.hypot(r1 - r0, c1 - c0) * RES

    return controller, path, {
        "method": method,
        "reached": int(bool(path) and path[-1] == controller.goal_rc),
        "path_points": len(path),
        "path_length_m": path_length,
        "min_clearance_m": float(clearances.min()),
        "mean_clearance_m": float(clearances.mean()),
        "planning_ms": planning_ms,
    }


def save_plot(output_dir, scenarios, runs):
    fig, axes = plt.subplots(len(scenarios), len(METHODS), figsize=(13, 8), squeeze=False)
    for row, (scene_name, occ) in enumerate(scenarios.items()):
        for col, method in enumerate(METHODS):
            controller, path, metrics = runs[(scene_name, method)]
            ax = axes[row, col]
            background = np.ma.masked_where(controller.soft_cost_map <= 0,
                                            controller.soft_cost_map)
            ax.imshow(background, origin="lower", cmap="YlOrRd", vmin=0.0, vmax=1.0)
            ax.imshow(np.ma.masked_where(occ == 0, occ), origin="lower", cmap="gray_r")
            ax.imshow(np.ma.masked_where(controller.hard_obstacle_map == 0,
                                        controller.hard_obstacle_map),
                      origin="lower", cmap="Reds", alpha=0.35)
            if path:
                rc = np.asarray(path)
                ax.plot(rc[:, 1], rc[:, 0], color="#1261a0", linewidth=2.0)
            ax.scatter([controller.robot_c], [controller.robot_r], c="#1a9850", s=35)
            ax.scatter([controller.goal_rc[1]], [controller.goal_rc[0]], c="#7b3294", s=35)
            ax.set_title(
                "%s | reached=%d | min=%.2fm"
                % (method, metrics["reached"], metrics["min_clearance_m"]),
                fontsize=9,
            )
            if col == 0:
                ax.set_ylabel(scene_name)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("FMM fixed inflation vs. hard/soft dual-layer cost")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison.png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(
        "assets", "runs", "fmm_cost_layers", stamp)
    os.makedirs(output_dir, exist_ok=True)

    scenarios = make_scenarios()
    runs = {}
    rows = []
    for scene_name, occ in scenarios.items():
        for method in METHODS:
            controller, path, metrics = run_case(occ, method)
            metrics["scene"] = scene_name
            runs[(scene_name, method)] = (controller, path, metrics)
            rows.append(metrics)

    csv_path = os.path.join(output_dir, "metrics.csv")
    fields = ["scene", "method", "reached", "path_points", "path_length_m",
              "min_clearance_m", "mean_clearance_m", "planning_ms"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    save_plot(output_dir, scenarios, runs)

    print("output_dir=%s" % os.path.abspath(output_dir))
    for row in rows:
        print("{scene:18s} {method:12s} reached={reached} min={min_clearance_m:.2f}m "
              "mean={mean_clearance_m:.2f}m time={planning_ms:.2f}ms".format(**row))


if __name__ == "__main__":
    main()

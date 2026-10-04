"""Generate overlaid paper figures from complete successful trajectories."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


METHODS = ("mppi", "iplanner", "ps_dwa", "sapf")
METHOD_LABELS = {
    "mppi": "本文足迹MPPI",
    "iplanner": "iPlanner",
    "ps_dwa": "PS-DWA",
    "sapf": "SAPF",
    "fmm": "固定膨胀FMM",
}
METHOD_COLORS = {
    "mppi": "#E76F00",
    "iplanner": "#1459C7",
    "ps_dwa": "#00897B",
    "sapf": "#9C27B0",
    "fmm": "#455A64",
}
SCENE_LABELS = {"easy": "简单场景", "medium": "中等场景", "hard": "困难场景"}

CAPTURES = {
    "easy": {
        "mppi": "easy/mppi_seed20261013",
        "iplanner": "easy/iplanner_seed20261013",
        "ps_dwa": "easy/ps_dwa_seed20261013",
        "sapf": "easy/sapf_seed20261013",
    },
    "medium": {
        "mppi": "medium/mppi_seed20261012",
        "iplanner": "medium/iplanner_seed20261013",
        "ps_dwa": "medium/ps_dwa_seed20261013",
        "sapf": "medium/sapf_seed20261016_attempt1",
    },
    "hard": {
        "mppi": "hard/mppi_seed20261005",
        "iplanner": "hard/iplanner_seed20261019",
        "ps_dwa": "hard/ps_dwa_seed20261016_attempt3",
        "sapf": "hard/sapf_seed20261038_screen",
    },
}


def configure_fonts():
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
    font_manager.fontManager.addfont(font_path)
    family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({
        "font.family": family,
        "font.sans-serif": [family],
        "axes.unicode_minus": False,
        "font.size": 10,
    })


def load_capture(root, relative, expected_method):
    directory = root / relative
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("success") is not True:
        raise ValueError("capture is not successful: %s" % result_path)
    if result.get("planner_mode") != expected_method:
        raise ValueError("method mismatch in %s" % result_path)
    trajectories = sorted((directory / "trajectories").glob("*.npy"))
    route_images = sorted((directory / "trial_route").glob("*success_route_camera.png"))
    if len(trajectories) != 1 or len(route_images) != 1:
        raise ValueError("incomplete successful capture: %s" % directory)
    trajectory = np.load(str(trajectories[0]))
    if trajectory.ndim != 2 or trajectory.shape[0] < 2 or trajectory.shape[1] < 2:
        raise ValueError("invalid trajectory: %s" % trajectories[0])
    trajectory = trajectory[:, :2]
    trajectory = trajectory[np.all(np.isfinite(trajectory), axis=1)]
    return {
        "directory": directory,
        "result_path": result_path,
        "image_path": route_images[0],
        "trajectory_path": trajectories[0],
        "trajectory": trajectory,
        "result": result,
    }


def style_axis(ax, title=None):
    ax.axvspan(12.0, 24.0, color="#ECEFF1", alpha=0.9, zorder=0)
    ax.text(18.0, 8.88, "随机障碍分布区", color="#546E7A", fontsize=9,
            ha="center", va="top", zorder=1)
    ax.set_xlim(5.5, 30.4)
    ax.set_ylim(3.0, 9.0)
    ax.set_xlabel("世界坐标 x / m")
    ax.set_ylabel("世界坐标 y / m")
    ax.grid(True, color="#CFD8DC", linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=9)


def plot_trajectory(ax, capture, method, include_seed=True):
    traj = capture["trajectory"]
    seed = int(capture["result"]["scene_seed"])
    label = METHOD_LABELS[method]
    if include_seed:
        label += "（%d）" % seed
    color = METHOD_COLORS[method]
    ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.6,
            alpha=0.96, label=label, zorder=3)
    ax.scatter(traj[0, 0], traj[0, 1], marker="o", s=38, color=color,
               edgecolor="white", linewidth=0.8, zorder=4)
    ax.scatter(traj[-1, 0], traj[-1, 1], marker="X", s=62, color=color,
               edgecolor="white", linewidth=0.8, zorder=4)


def add_endpoint_note(ax):
    ax.text(0.012, 0.018, "圆点：起点    叉号：终点",
            transform=ax.transAxes, fontsize=8.5, color="#37474F",
            ha="left", va="bottom")


def save_png(fig, output, stem):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / (stem + ".png"), dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.06)
    plt.close(fig)


def plot_scene(scene, captures, output):
    fig, ax = plt.subplots(figsize=(12.8, 4.7), constrained_layout=True)
    style_axis(ax, "%s下四种方法的完整成功路径" % SCENE_LABELS[scene])
    for method in METHODS:
        plot_trajectory(ax, captures[scene][method], method, include_seed=(scene != "easy"))
    ax.legend(loc="upper right", ncol=2, framealpha=0.94, fontsize=9)
    add_endpoint_note(ax)
    save_png(fig, output, {"easy": "01", "medium": "02", "hard": "03"}[scene]
             + "_%s_four_successful_routes_overlay" % scene)


def plot_combined(captures, output):
    scenes = ("easy", "medium", "hard")
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.0), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, scene in zip(axes, scenes):
        style_axis(ax, SCENE_LABELS[scene])
        for method in METHODS:
            plot_trajectory(ax, captures[scene][method], method,
                            include_seed=(scene != "easy"))
        ax.legend(loc="upper right", fontsize=7.4, framealpha=0.94)
        add_endpoint_note(ax)
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    fig.suptitle("三种难度下四种方法的完整成功路径",
                 fontsize=16, fontweight="bold")
    save_png(fig, output, "04_three_difficulties_four_methods_overlay_1x3")


def plot_fmm_mppi(root, output):
    pair = {
        "fmm": load_capture(root, "easy/fmm_seed20261013", "fmm"),
        "mppi": load_capture(root, "easy/mppi_seed20261013", "mppi"),
    }
    fig, ax = plt.subplots(figsize=(12.8, 4.7), constrained_layout=True)
    style_axis(ax, "固定膨胀FMM与本文方法的完整成功路径对比（seed=20261013）")
    for method in ("fmm", "mppi"):
        plot_trajectory(ax, pair[method], method, include_seed=False)
    ax.legend(loc="upper right", framealpha=0.94)
    add_endpoint_note(ax)
    save_png(fig, output, "05_fixed_fmm_vs_soft_cost_footprint_mppi_overlay")
    return pair


def write_manifest(captures, pair, output):
    manifest = {
        "selection_rule": "Only captures with result.success == true are plotted.",
        "display_note": "Medium and hard overlays contain successful runs from different seeds.",
        "scenes": {},
        "paired_fmm_mppi": {},
    }
    for scene, methods in captures.items():
        manifest["scenes"][scene] = {}
        for method, capture in methods.items():
            result = capture["result"]
            manifest["scenes"][scene][method] = {
                "success": True,
                "scene_seed": int(result["scene_seed"]),
                "elapsed_s": float(result["elapsed_s"]),
                "distance_m": float(result["distance_m"]),
                "result_json": str(capture["result_path"]),
                "trajectory_npy": str(capture["trajectory_path"]),
                "route_image": str(capture["image_path"]),
            }
    for method, capture in pair.items():
        result = capture["result"]
        manifest["paired_fmm_mppi"][method] = {
            "success": True,
            "scene_seed": int(result["scene_seed"]),
            "elapsed_s": float(result["elapsed_s"]),
            "distance_m": float(result["distance_m"]),
            "result_json": str(capture["result_path"]),
        }
    (output / "successful_route_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_fonts()
    captures = {
        scene: {method: load_capture(args.input, relative, method)
                for method, relative in methods.items()}
        for scene, methods in CAPTURES.items()
    }
    for scene in ("easy", "medium", "hard"):
        plot_scene(scene, captures, args.output)
    plot_combined(captures, args.output)
    pair = plot_fmm_mppi(args.input, args.output)
    write_manifest(captures, pair, args.output)
    print("Generated overlaid successful-route PNGs in %s" % args.output)


if __name__ == "__main__":
    main()

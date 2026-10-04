"""Compose Chapter 3 visual evidence from representative Isaac Gym captures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image


SCENES = (
    ("简单", "easy_seed20261001", "10大石 + 2树 + 100小碎石"),
    ("中等", "medium_seed20261001", "15大石 + 3树 + 200小碎石"),
    ("困难", "hard_seed20261001", "20大石 + 4树 + 400小碎石"),
)


def configure_style():
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=path).get_name()
            break
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def first_match(directory, pattern):
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError("no files matching %s in %s" % (pattern, directory))
    return files[0]


def show_image(ax, path, title):
    with Image.open(str(path)) as image:
        ax.imshow(image.convert("RGB"))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def save(fig, output, stem):
    fig.savefig(str(output / (stem + ".png")), dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(str(output / (stem + ".pdf")),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def scene_overviews(root, output, route=False):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    suffix = "*_route_camera.png" if route else "*_camera_raw.png"
    for ax, (label, directory, description) in zip(axes, SCENES):
        path = first_match(root / directory / "trial_route", suffix)
        show_image(ax, path, "%s：%s" % (label, description))
    fig.suptitle("Isaac Gym 三档离散高度障碍场景" +
                 ("与实际导航轨迹" if route else "原始俯视图"), fontsize=14)
    fig.tight_layout()
    save(fig, output, "09_scene_route_comparison" if route
         else "08_isaacgym_scene_comparison")


def pipeline(root, output, directory="hard_seed20261001", step=1000):
    scene = root / directory
    name = "trial_01_step_%05d.png" % step
    paths = (
        (scene / "camera" / ("rgb_%06d.png" % step), "D435 RGB"),
        (scene / "camera" / ("depth_%06d.png" % step), "D435 Depth Image"),
        (scene / "bev_fmm" / "height" / name, "Local Height BEV"),
        (scene / "bev_fmm" / "obstacles" / name, "Footprint Obstacle Map"),
        (scene / "bev_fmm" / "soft_cost" / name, "Continuous Safety Cost"),
        (scene / "bev_fmm" / "potential" / name, "FMM Potential and Path"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.3))
    for ax, (path, title) in zip(axes.flat, paths):
        show_image(ax, path, title)
    fig.suptitle("RGB-D Perception to Local Planning: Hard Terrain, Step %d" % step,
                 fontsize=14)
    fig.tight_layout()
    save(fig, output, "10_rgbd_bev_cost_planning_pipeline")


def temporal_bev(root, output, directory="hard_seed20261001"):
    steps = (500, 1250, 1950)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.0))
    for col, step in enumerate(steps):
        name = "trial_01_step_%05d.png" % step
        show_image(axes[0, col], root / directory / "bev_fmm" / "obstacles" / name,
                   "Step %d：障碍层" % step)
        show_image(axes[1, col], root / directory / "bev_fmm" / "potential" / name,
                   "Step %d：势场路径" % step)
    fig.suptitle("机器人运动过程中的局部 BEV 与重规划时序变化", fontsize=14)
    fig.tight_layout()
    save(fig, output, "11_temporal_bev_replanning")


def difficulty_layers(root, output, layer, stem, title, step=1000):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    name = "trial_01_step_%05d.png" % step
    for ax, (label, directory, description) in zip(axes, SCENES):
        show_image(ax, root / directory / "bev_fmm" / layer / name,
                   "%s：%s" % (label, description))
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    save(fig, output, stem)


def rgb_depth_comparison(root, output, step=1000):
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.8))
    for col, (label, directory, _) in enumerate(SCENES):
        scene = root / directory / "camera"
        show_image(axes[0, col], scene / ("rgb_%06d.png" % step), "%s RGB" % label)
        show_image(axes[1, col], scene / ("depth_%06d.png" % step), "%s Depth" % label)
    fig.suptitle("三档场景的机载 RGB-D 观测", fontsize=14)
    fig.tight_layout()
    save(fig, output, "13_rgb_depth_difficulty_comparison")


def cost_potential_comparison(root, output, step=1000):
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.0))
    name = "trial_01_step_%05d.png" % step
    for col, (label, directory, _) in enumerate(SCENES):
        base = root / directory / "bev_fmm"
        show_image(axes[0, col], base / "soft_cost" / name, "%s：软代价" % label)
        show_image(axes[1, col], base / "potential" / name, "%s：FMM 势场" % label)
    fig.suptitle("连续安全代价与 FMM 到达时间势场", fontsize=14)
    fig.tight_layout()
    save(fig, output, "14_soft_cost_and_fmm_potential")


def evidence_inventory(root, output):
    labels = []
    counts = {"RGB-D": [], "高度BEV": [], "障碍BEV": [], "软代价": [], "势场": []}
    for label, directory, _ in SCENES:
        scene = root / directory
        labels.append(label)
        counts["RGB-D"].append(len(list((scene / "camera").glob("*.png"))))
        for key, sub in (("高度BEV", "height"), ("障碍BEV", "obstacles"),
                         ("软代价", "soft_cost"), ("势场", "potential")):
            counts[key].append(len(list((scene / "bev_fmm" / sub).glob("*.png"))))
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = range(len(labels))
    bottom = [0] * len(labels)
    colors = ("#2878B5", "#66A6C9", "#E39D26", "#D45B72", "#2A9D6F")
    for (key, values), color in zip(counts.items(), colors):
        ax.bar(x, values, bottom=bottom, label=key, color=color, width=0.58)
        bottom = [a + b for a, b in zip(bottom, values)]
    for i, total in enumerate(bottom):
        ax.text(i, total + 5, "%d 张" % total, ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("保存图像数量")
    ax.set_title("三档代表性试验的多模态过程数据量")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.subplots_adjust(bottom=0.22)
    save(fig, output, "15_visual_evidence_inventory")


def write_readme(root, output):
    rows = []
    for label, directory, description in SCENES:
        result = json.loads((root / directory / "result.json").read_text(encoding="utf-8"))
        rows.append("| %s | %s | %s | %.2f | %.2f |" % (
            label, description, "成功" if result["success"] else result["failure_reason"],
            result["distance_m"], result["final_x_m"]))
    text = """# 第三章视觉证据图

本目录中的 `08` 至 `15` 为从真实 Isaac Gym 单次代表性采集中整理的组合图；
原始图片位于同级 `ch3_visual_evidence_20260925/`。截图试验仅用于展示感知与规划过程，
论文中的成功率和统计检验仍以三档各 30 个固定种子的正式基准为准。

| 难度 | 障碍配置 | 截图试验状态 | 路程/m | 终点x/m |
|---|---|---:|---:|---:|
%s

组合图内容：

1. `08` Isaac Gym 三档原始场景。
2. `09` 三档场景与实际轨迹叠加。
3. `10` RGB-D、BEV、软代价到 FMM 势场的完整链路。
4. `11` 三个时刻的局部 BEV 重规划变化。
5. `12` 三档障碍 BEV 对比。
6. `13` 三档机载 RGB-D 观测。
7. `14` 连续软代价与 FMM 势场对比。
8. `15` 过程图像数据量统计。
""" % "\n".join(rows)
    (output / "VISUAL_EVIDENCE_README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    parser.add_argument("--input", type=Path, default=repo / "assets" / "runs" /
                        "navigation_benchmark" / "ch3_visual_evidence_20260925")
    parser.add_argument("--output", type=Path, default=repo / "assets" / "runs" /
                        "navigation_benchmark" / "ch3_paper_figures_20260925")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_style()
    scene_overviews(args.input, args.output, route=False)
    scene_overviews(args.input, args.output, route=True)
    pipeline(args.input, args.output)
    temporal_bev(args.input, args.output)
    difficulty_layers(args.input, args.output, "obstacles",
                      "12_obstacle_bev_difficulty_comparison",
                      "三档障碍密度下的局部占据图")
    rgb_depth_comparison(args.input, args.output)
    cost_potential_comparison(args.input, args.output)
    evidence_inventory(args.input, args.output)
    write_readme(args.input, args.output)
    print("Visual evidence figures: %s" % args.output)


if __name__ == "__main__":
    main()

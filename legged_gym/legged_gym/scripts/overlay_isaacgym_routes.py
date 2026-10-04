"""Overlay successful planner trajectories on one Isaac Gym camera frame."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np


DEFAULT_METHODS = ("mppi", "iplanner", "ps_dwa", "sapf")
LABELS = {
    "mppi": "Ours",
    "umppi": "Ours",
    "iplanner": "iPlanner",
    "ps_dwa": "PS-DWA",
    "sapf": "SAPF",
    "fmm_fixed": "FMM",
}
COLORS = {
    "mppi": "#D62728",
    "umppi": "#D62728",
    "iplanner": "#1565C0",
    "ps_dwa": "#F57C00",
    "sapf": "#8E24AA",
    "fmm_fixed": "#455A64",
}
EXPECTED_MODES = {"fmm_fixed": "fmm", "umppi": "mppi"}


def _font(size, bold=False):
    name = "NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc"
    return ImageFont.truetype("/usr/share/fonts/opentype/noto/%s" % name, size)


def _load_run(root, method):
    run_dir = root / method
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("success") is not True:
        raise ValueError("Only a successful trajectory may be plotted: %s" % result_path)
    if result.get("planner_mode") != EXPECTED_MODES.get(method, method):
        raise ValueError("Planner mode mismatch: %s" % result_path)
    if method == "umppi" and result.get("mppi_variant") != "unscented":
        raise ValueError("U-MPPI variant metadata mismatch: %s" % result_path)

    route_dir = run_dir / "trial_route"
    pixels = sorted(route_dir.glob("*success_trajectory_pixels.npy"))
    backgrounds = sorted(route_dir.glob("*success_camera_raw.png"))
    if len(pixels) != 1 or len(backgrounds) != 1:
        raise ValueError("Incomplete route capture: %s" % run_dir)
    trajectory = np.load(str(pixels[0])).astype(np.float32)
    if trajectory.ndim != 2 or trajectory.shape[0] < 2 or trajectory.shape[1] != 2:
        raise ValueError("Invalid projected trajectory: %s" % pixels[0])
    return result, trajectory, backgrounds[0]


def _scaled_points(trajectory, scale):
    return [tuple(point) for point in np.rint(trajectory * scale).astype(np.int32)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                        help="comma-separated method directory names")
    parser.add_argument("--crop", default="143,10,352,346",
                        help="crop in raw pixels: left,top,width,height (or left,top,size)")
    parser.add_argument("--legend-right-raw", type=int, default=480,
                        help="legend right edge in raw-camera x pixels")
    parser.add_argument("--legend-top-raw", type=int, default=30,
                        help="legend top edge in raw-camera y pixels")
    parser.add_argument(
        "--allow-layout-version-mismatch", action="store_true",
        help="allow legacy captures with identical seed/camera but different layout hash")
    args = parser.parse_args()

    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    if not methods or any(method not in LABELS for method in methods):
        raise ValueError("Unknown or empty --methods: %s" % (methods,))
    captures = {method: _load_run(args.input, method) for method in methods}
    hashes = {capture[0].get("obstacle_layout_hash") for capture in captures.values()}
    seeds = {capture[0].get("scene_seed") for capture in captures.values()}
    if len(seeds) != 1:
        raise ValueError("All methods must use the same scene seed")
    if len(hashes) != 1 and not args.allow_layout_version_mismatch:
        raise ValueError("All methods must use the same obstacle layout hash")
    if len(hashes) != 1:
        print("WARNING: overlay combines captures from different scene-code versions: %s"
              % sorted(str(value) for value in hashes))

    scale = max(1, args.scale)
    background_method = ("umppi" if "umppi" in captures else
                         "mppi" if "mppi" in captures else methods[0])
    background = Image.open(captures[background_method][2]).convert("RGB")
    crop_values = [int(value) for value in args.crop.split(",")]
    if len(crop_values) == 3:
        crop_left, crop_top, crop_width = crop_values
        crop_height = crop_width
    elif len(crop_values) == 4:
        crop_left, crop_top, crop_width, crop_height = crop_values
    else:
        raise ValueError("--crop must be left,top,size or left,top,width,height")
    background = background.crop(
        (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
    resampling = getattr(Image, "Resampling", Image)
    image = background.resize((crop_width * scale, crop_height * scale),
                              resampling.LANCZOS)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # White casing keeps overlapping paths readable on both rocks and bright terrain.
    proposed = "umppi" if "umppi" in methods else "mppi"
    draw_order = tuple(method for method in methods if method != proposed)
    if proposed in methods:
        draw_order += (proposed,)
    for method in draw_order:
        trajectory = captures[method][1] - np.asarray([crop_left, crop_top])
        points = _scaled_points(trajectory, scale)
        draw.line(points, fill=(255, 255, 255, 215), width=4 * scale, joint="curve")
        color = COLORS[method]
        draw.line(points, fill=color, width=2 * scale, joint="curve")

    starts = np.stack([capture[1][0] for capture in captures.values()])
    goals = np.stack([capture[1][-1] for capture in captures.values()])
    crop_origin = np.asarray([crop_left, crop_top])
    start = tuple(np.rint((starts.mean(axis=0) - crop_origin) * scale).astype(int))
    goal = tuple(np.rint((goals.mean(axis=0) - crop_origin) * scale).astype(int))
    marker_radius = 5 * scale
    for center, fill in ((start, "#D32F2F"), (goal, "#2E7D32")):
        box = (center[0] - marker_radius, center[1] - marker_radius,
               center[0] + marker_radius, center[1] + marker_radius)
        draw.ellipse(box, fill=fill, outline="white", width=2 * scale)

    legend_font = _font(7 * scale)
    legend_w_raw = 78 if len(methods) > 2 else 58
    legend_h_raw = 8 + 12 * len(methods)
    legend_x = (args.legend_right_raw - crop_left - legend_w_raw - 5) * scale
    legend_y = (args.legend_top_raw - crop_top) * scale
    legend_x = max(4 * scale, min(legend_x, image.width - (legend_w_raw + 4) * scale))
    legend_y = max(4 * scale, min(legend_y, image.height - (legend_h_raw + 4) * scale))
    legend_w, legend_h = legend_w_raw * scale, legend_h_raw * scale
    draw.rounded_rectangle((legend_x, legend_y, legend_x + legend_w, legend_y + legend_h),
                           radius=3 * scale, fill=(255, 255, 255, 220),
                           outline=(80, 80, 80, 190), width=scale)
    for index, method in enumerate(methods):
        y = legend_y + (8 + index * 12) * scale
        draw.line((legend_x + 6 * scale, y, legend_x + 18 * scale, y),
                  fill=COLORS[method], width=2 * scale)
        draw.text((legend_x + 23 * scale, y - 5 * scale), LABELS[method],
                  fill="#202124", font=legend_font)

    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format="PNG", dpi=(300, 300), optimize=True)
    print("Saved: %s" % args.output)
    print("Verified successful methods: %s" % ", ".join(methods))
    print("Scene seed=%s, layout hashes=%s" % (
        next(iter(seeds)), ",".join(sorted(str(value) for value in hashes))))


if __name__ == "__main__":
    main()

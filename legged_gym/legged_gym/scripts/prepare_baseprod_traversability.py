"""Prepare traversability labels from the BASEPROD D435i labeled dataset.

BASEPROD labels:
0 void, 1 compact, 2 grass, 3 bedrock, 4 sandy, 5 pebble, 6 rock, 7 vegetation.

Coarse traversability labels:
0 low-risk traversable: compact, grass, sandy
1 medium-risk traversable: bedrock, pebble, vegetation
2 obstacle/high-risk: rock
255 ignore: void
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


BASEPROD_CLASSES = {
    0: "void",
    1: "compact",
    2: "grass",
    3: "bedrock",
    4: "sandy",
    5: "pebble",
    6: "rock",
    7: "vegetation",
}

LABEL_MAP = np.full(256, 255, dtype=np.uint8)
COST_MAP = np.full(256, 255, dtype=np.uint8)
for raw_id in (1, 2, 4):
    LABEL_MAP[raw_id] = 0
    COST_MAP[raw_id] = 0
for raw_id in (3, 5, 7):
    LABEL_MAP[raw_id] = 1
    COST_MAP[raw_id] = 128
LABEL_MAP[6] = 2
COST_MAP[6] = 255


def is_label_png(path):
    try:
        image = Image.open(path)
        if image.mode not in ("L", "P", "I;16", "I"):
            return False
        arr = np.asarray(image)
        if arr.ndim != 2:
            return False
        unique = np.unique(arr)
        return unique.size > 1 and int(unique.max()) <= 7
    except Exception:
        return False


def find_rgb_by_stem(root, stem):
    stems = [stem]
    if stem.endswith("_mask"):
        stems.append(stem[:-5])
    candidates = []
    for candidate_stem in stems:
        for path in root.rglob(candidate_stem + ".png"):
            parts = {part.lower() for part in path.parts}
            if "color_imgs" in parts or "rgb" in parts or "images" in parts:
                try:
                    if Image.open(path).convert("RGB").size[0] > 0:
                        candidates.append(path)
                except Exception:
                    pass
    if not candidates:
        return None
    candidates.sort(key=lambda p: (0 if "color_imgs" in {x.lower() for x in p.parts} else 1, len(p.parts)))
    return candidates[0]


def convert(root_dir, output_dir, limit=None):
    root = Path(root_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    rgb_dir = output / "rgb"
    label_dir = output / "label_trav"
    cost_dir = output / "cost"
    preview_dir = output / "preview"
    for directory in (rgb_dir, label_dir, cost_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    label_paths = [p for p in root.rglob("*.png") if is_label_png(p)]
    label_paths.sort()
    if limit is not None:
        label_paths = label_paths[: int(limit)]
    manifest = []
    for idx, label_path in enumerate(label_paths):
        rgb_path_in = find_rgb_by_stem(root, label_path.stem)
        if rgb_path_in is None:
            continue
        rgb = Image.open(rgb_path_in).convert("RGB")
        raw = np.asarray(Image.open(label_path), dtype=np.uint8)
        trav = LABEL_MAP[raw]
        cost = COST_MAP[raw]

        stem = "%06d" % len(manifest)
        rgb_path = rgb_dir / (stem + ".jpg")
        label_path_out = label_dir / (stem + ".png")
        cost_path = cost_dir / (stem + ".png")
        rgb.save(rgb_path, quality=92)
        Image.fromarray(trav).save(label_path_out)
        Image.fromarray(cost).save(cost_path)

        if len(manifest) < 12:
            overlay = np.asarray(rgb).copy()
            colors = np.zeros((*trav.shape, 3), dtype=np.uint8)
            colors[trav == 0] = (40, 180, 60)
            colors[trav == 1] = (240, 190, 40)
            colors[trav == 2] = (220, 50, 50)
            valid = trav != 255
            overlay[valid] = (0.55 * overlay[valid] + 0.45 * colors[valid]).astype(np.uint8)
            Image.fromarray(overlay).save(preview_dir / (stem + "_overlay.jpg"), quality=92)

        unique, counts = np.unique(trav, return_counts=True)
        manifest.append({
            "rgb": str(rgb_path),
            "label_trav": str(label_path_out),
            "cost": str(cost_path),
            "source_rgb": str(rgb_path_in),
            "source_label": str(label_path),
            "histogram": {int(k): int(v) for k, v in zip(unique, counts)},
        })

    summary = {
        "source": str(root),
        "classes_source": BASEPROD_CLASSES,
        "classes": {
            "0": "low-risk traversable",
            "1": "medium-risk traversable",
            "2": "obstacle/high-risk",
            "255": "ignore",
        },
        "samples": manifest,
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Found %d label PNGs, wrote %d paired samples to %s"
          % (len(label_paths), len(manifest), output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Extracted BASEPROD_labeled dataset root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    convert(args.root, args.output, args.limit)


if __name__ == "__main__":
    main()

"""Prepare a RUGD traversability dataset from Hugging Face parquet shards.

The output labels are coarse traversability classes:

0: low-risk traversable       dirt, sand, grass, asphalt, mulch, bridge, concrete
1: medium-risk traversable    gravel, rock-bed, bush
2: non-traversable / obstacle tree, pole, vehicle, container, building, log,
                              bicycle, person, fence, sign, rock, picnic-table, water
255: ignore                   sky and unknown labels
"""

import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq


LOW_RISK = {
    "dirt", "sand", "grass", "asphalt", "mulch", "bridge", "concrete"
}
MEDIUM_RISK = {
    "gravel", "rock-bed", "bush"
}
OBSTACLE = {
    "tree", "pole", "water", "vehicle", "container/generic-object",
    "building", "log", "bicycle", "person", "fence", "sign", "rock",
    "picnic-table",
}
IGNORE = {"sky"}


def build_mapping(id2label):
    mapping = np.full(256, 255, dtype=np.uint8)
    cost = np.full(256, 255, dtype=np.uint8)
    for raw_id, name in id2label.items():
        idx = int(raw_id)
        if name in LOW_RISK:
            mapping[idx] = 0
            cost[idx] = 0
        elif name in MEDIUM_RISK:
            mapping[idx] = 1
            cost[idx] = 128
        elif name in OBSTACLE:
            mapping[idx] = 2
            cost[idx] = 255
        elif name in IGNORE:
            mapping[idx] = 255
            cost[idx] = 255
        else:
            raise ValueError("unmapped RUGD class: %s=%s" % (raw_id, name))
    return mapping, cost


def build_grayscale_mapping():
    """Mapping for HF mirrors that store RUGD color masks as grayscale PNGs.

    Some RGB classes collide after luminance conversion. The chosen values keep
    the classes most relevant to Go2 outdoor deployment useful: grass remains
    traversable, bush/rock-bed are medium risk, and tree/rock/log/person/water
    remain obstacles.
    """
    label = np.full(256, 255, dtype=np.uint8)
    cost = np.full(256, 255, dtype=np.uint8)

    def set_value(gray, trav_class, risk):
        label[int(gray)] = np.uint8(trav_class)
        cost[int(gray)] = np.uint8(risk)

    for gray in (60, 64, 72, 91, 209, 234):  # grass, asphalt, dirt, concrete, bridge, sand
        set_value(gray, 0, 0)
    for gray in (90, 151, 189):              # mulch/rock-bed, gravel, bush
        set_value(gray, 1, 128)
    for gray in (30, 54, 76, 89, 104, 107, 150, 164, 180, 195, 226):
        set_value(gray, 2, 255)
    set_value(29, 255, 255)                  # sky
    set_value(0, 255, 255)                   # void
    return label, cost


def image_from_cell(cell):
    value = cell.as_py()
    if isinstance(value, dict):
        data = value.get("bytes")
        if data is None and value.get("path"):
            return Image.open(value["path"])
        return Image.open(BytesIO(data))
    raise TypeError("unexpected image cell type: %r" % type(value))


def convert(parquet_paths, id2label_path, output_dir, limit):
    output = Path(output_dir)
    rgb_dir = output / "rgb"
    label_dir = output / "label_trav"
    cost_dir = output / "cost"
    preview_dir = output / "preview"
    for directory in (rgb_dir, label_dir, cost_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    id2label = json.loads(Path(id2label_path).read_text(encoding="utf-8"))
    mapping, cost_mapping = build_mapping(id2label)
    gray_mapping, gray_cost_mapping = build_grayscale_mapping()

    manifest = []
    total_limit = None if limit is None else int(limit)
    sample_idx = 0
    used_sources = []
    for parquet_path in parquet_paths:
        if total_limit is not None and sample_idx >= total_limit:
            break
        table = pq.read_table(parquet_path, columns=["image", "label"])
        rows = table.num_rows
        if total_limit is not None:
            rows = min(rows, total_limit - sample_idx)
        used_sources.append(str(parquet_path))
        for row_idx in range(rows):
            rgb = image_from_cell(table["image"][row_idx]).convert("RGB")
            raw_label = np.asarray(image_from_cell(table["label"][row_idx]))
            if raw_label.ndim == 3:
                raw_label = raw_label[:, :, 0]
            raw_label = raw_label.astype(np.uint8)
            if int(raw_label.max()) <= max(int(k) for k in id2label):
                trav_label = mapping[raw_label]
                trav_cost = cost_mapping[raw_label]
            else:
                trav_label = gray_mapping[raw_label]
                trav_cost = gray_cost_mapping[raw_label]

            stem = "%06d" % sample_idx
            rgb_path = rgb_dir / (stem + ".jpg")
            label_path = label_dir / (stem + ".png")
            cost_path = cost_dir / (stem + ".png")
            rgb.save(rgb_path, quality=92)
            Image.fromarray(trav_label).save(label_path)
            Image.fromarray(trav_cost).save(cost_path)

            if sample_idx < 12:
                overlay = np.asarray(rgb).copy()
                colors = np.zeros((*trav_label.shape, 3), dtype=np.uint8)
                colors[trav_label == 0] = (40, 180, 60)
                colors[trav_label == 1] = (240, 190, 40)
                colors[trav_label == 2] = (220, 50, 50)
                mask = trav_label != 255
                overlay[mask] = (0.55 * overlay[mask] + 0.45 * colors[mask]).astype(np.uint8)
                Image.fromarray(overlay).save(preview_dir / (stem + "_overlay.jpg"), quality=92)

            unique, counts = np.unique(trav_label, return_counts=True)
            hist = {int(k): int(v) for k, v in zip(unique, counts)}
            manifest.append({
                "rgb": str(rgb_path),
                "label_trav": str(label_path),
                "cost": str(cost_path),
                "histogram": hist,
            })
            sample_idx += 1

    (output / "manifest.json").write_text(
        json.dumps({
            "source_parquet": used_sources,
            "id2label": id2label,
            "classes": {
                "0": "low-risk traversable",
                "1": "medium-risk traversable",
                "2": "non-traversable obstacle",
                "255": "ignore",
            },
            "samples": manifest,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("Wrote %d samples to %s" % (sample_idx, output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", nargs="+", required=True)
    parser.add_argument("--id2label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=256)
    args = parser.parse_args()
    convert(args.parquet, args.id2label, args.output, args.limit)


if __name__ == "__main__":
    main()

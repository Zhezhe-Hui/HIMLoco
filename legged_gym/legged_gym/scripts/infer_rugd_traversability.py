"""Export RGB traversability predictions as label, cost, and preview images."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from train_rugd_traversability import TinyUNet, IGNORE_INDEX


COLORS = np.array([
    [40, 180, 60],
    [240, 190, 40],
    [220, 50, 50],
], dtype=np.uint8)
COST_VALUES = np.array([0, 128, 255], dtype=np.uint8)


def load_rgb(path, image_size):
    rgb = Image.open(path).convert("RGB")
    original_size = rgb.size
    resized = rgb.resize(image_size[::-1], Image.BILINEAR)
    image = np.asarray(resized, dtype=np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    return rgb, tensor, original_size


def update_confusion(confusion, pred, label):
    valid = label != IGNORE_INDEX
    for target, output in zip(label[valid].reshape(-1), pred[valid].reshape(-1)):
        confusion[int(target), int(output)] += 1


def metrics_from_confusion(confusion):
    ious = []
    for cls in range(3):
        tp = confusion[cls, cls]
        denom = confusion[cls, :].sum() + confusion[:, cls].sum() - tp
        ious.append(float(tp / denom) if denom > 0 else 0.0)
    acc = float(np.trace(confusion) / np.sum(confusion)) if np.sum(confusion) > 0 else 0.0
    return {
        "pixel_acc": acc,
        "miou": float(sum(ious) / len(ious)),
        "iou_low": ious[0],
        "iou_medium": ious[1],
        "iou_obstacle": ious[2],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Directory containing manifest.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--preview-count", type=int, default=24)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    height = int(args.height or ckpt_args.get("height", 96))
    width = int(args.width or ckpt_args.get("width", 144))
    width_mult = int(ckpt_args.get("width_mult", 8))

    output = Path(args.output)
    label_dir = output / "pred_label"
    cost_dir = output / "pred_cost"
    preview_dir = output / "preview"
    for directory in (label_dir, cost_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model = TinyUNet(width=width_mult)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    manifest_path = Path(args.dataset) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if args.limit is not None and args.limit > 0:
        samples = samples[: int(args.limit)]

    confusion = np.zeros((3, 3), dtype=np.float64)
    records = []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            rgb, tensor, original_size = load_rgb(sample["rgb"], (height, width))
            logits = model(tensor.to(device))
            pred_small = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred = np.asarray(
                Image.fromarray(pred_small).resize(original_size, Image.NEAREST),
                dtype=np.uint8,
            )
            cost = COST_VALUES[pred]

            stem = "%06d" % i
            label_path = label_dir / (stem + ".png")
            cost_path = cost_dir / (stem + ".png")
            Image.fromarray(pred).save(label_path)
            Image.fromarray(cost).save(cost_path)

            if "label_trav" in sample:
                label = np.asarray(Image.open(sample["label_trav"]), dtype=np.uint8)
                update_confusion(confusion, pred, label)

            if i < args.preview_count:
                rgb_np = np.asarray(rgb)
                pred_color = COLORS[pred]
                if "label_trav" in sample:
                    label = np.asarray(Image.open(sample["label_trav"]), dtype=np.uint8)
                    target_color = np.zeros_like(pred_color)
                    valid = label != IGNORE_INDEX
                    target_color[valid] = COLORS[label[valid]]
                else:
                    target_color = np.zeros_like(pred_color)
                canvas = np.concatenate([rgb_np, target_color, pred_color], axis=1)
                Image.fromarray(canvas).save(preview_dir / (stem + "_rgb_gt_pred.jpg"), quality=92)

            records.append({
                "rgb": sample["rgb"],
                "pred_label": str(label_path),
                "pred_cost": str(cost_path),
            })
            if (i + 1) % 50 == 0 or i + 1 == len(samples):
                print("exported %d/%d" % (i + 1, len(samples)), flush=True)

    summary = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "samples": len(samples),
        "device": str(device),
        "height": height,
        "width": width,
        "width_mult": width_mult,
        "metrics": metrics_from_confusion(confusion),
        "records": records,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote %s" % output)


if __name__ == "__main__":
    main()

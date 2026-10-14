"""Train a lightweight RGB traversability segmentation model on RUGD.

The model predicts coarse traversability classes:
0 low-risk traversable, 1 medium-risk traversable, 2 obstacle.
Label value 255 is ignored during training and evaluation.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IGNORE_INDEX = 255


class RUGDTraversabilityDataset(Dataset):
    def __init__(self, manifest_path, indices, image_size, cache_in_memory=False,
                 cache_name="dataset"):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.samples = [manifest["samples"][i] for i in indices]
        self.image_size = tuple(image_size)
        self.cache = None
        if cache_in_memory:
            self.cache = []
            for i in range(len(self.samples)):
                self.cache.append(self._load_sample(i))
                if (i + 1) % 100 == 0 or i + 1 == len(self.samples):
                    print(
                        "cached %s %d/%d" % (cache_name, i + 1, len(self.samples)),
                        flush=True,
                    )

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, idx):
        sample = self.samples[idx]
        rgb = Image.open(sample["rgb"]).convert("RGB")
        label = Image.open(sample["label_trav"])
        rgb = rgb.resize(self.image_size[::-1], Image.BILINEAR)
        label = label.resize(self.image_size[::-1], Image.NEAREST)

        image = np.asarray(rgb, dtype=np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        image = torch.from_numpy(image.transpose(2, 0, 1))
        target = torch.from_numpy(np.asarray(label, dtype=np.int64))
        return image, target

    def __getitem__(self, idx):
        if self.cache is not None:
            return self.cache[idx]
        return self._load_sample(idx)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, num_classes=3, width=24):
        super().__init__()
        self.enc1 = ConvBlock(3, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.enc3 = ConvBlock(width * 2, width * 4)
        self.bottleneck = ConvBlock(width * 4, width * 8)
        self.dec3 = ConvBlock(width * 8 + width * 4, width * 4)
        self.dec2 = ConvBlock(width * 4 + width * 2, width * 2)
        self.dec1 = ConvBlock(width * 2 + width, width)
        self.head = nn.Conv2d(width, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def split_indices(num_samples, val_ratio, seed, limit=None):
    indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if limit is not None:
        indices = indices[: int(limit)]
    val_count = max(1, int(round(len(indices) * val_ratio)))
    return indices[val_count:], indices[:val_count]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_pixels = 0
    confusion = torch.zeros(3, 3, dtype=torch.float64)
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX)
        valid = targets != IGNORE_INDEX
        total_loss += float(loss.item()) * int(valid.sum().item())
        total_pixels += int(valid.sum().item())

        pred = logits.argmax(dim=1)
        for t, p in zip(targets[valid].view(-1), pred[valid].view(-1)):
            confusion[int(t), int(p)] += 1

    ious = []
    for cls in range(3):
        tp = confusion[cls, cls]
        denom = confusion[cls, :].sum() + confusion[:, cls].sum() - tp
        ious.append(float(tp / denom) if denom > 0 else 0.0)
    acc = float(torch.diag(confusion).sum() / confusion.sum()) if confusion.sum() > 0 else 0.0
    return {
        "loss": total_loss / max(1, total_pixels),
        "pixel_acc": acc,
        "miou": float(sum(ious) / len(ious)),
        "iou_low": ious[0],
        "iou_medium": ious[1],
        "iou_obstacle": ious[2],
    }


@torch.no_grad()
def save_preview(model, dataset, output_dir, device, count=6):
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = np.array([[40, 180, 60], [240, 190, 40], [220, 50, 50]], dtype=np.uint8)
    for i in range(min(count, len(dataset))):
        image, target = dataset[i]
        logits = model(image.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        rgb = image.numpy().transpose(1, 2, 0)
        rgb = rgb * np.array([0.229, 0.224, 0.225], dtype=np.float32) + np.array(
            [0.485, 0.456, 0.406], dtype=np.float32
        )
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        pred_color = colors[pred]
        target_np = target.numpy()
        target_color = np.zeros_like(pred_color)
        valid = target_np != IGNORE_INDEX
        target_color[valid] = colors[target_np[valid]]
        canvas = np.concatenate([rgb, target_color, pred_color], axis=1)
        Image.fromarray(canvas).save(output_dir / ("%03d_rgb_gt_pred.jpg" % i), quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Directory containing manifest.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--width-mult", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--cache-in-memory", action="store_true")
    parser.add_argument(
        "--class-weights",
        default="1.0,1.3,1.2",
        help="comma-separated cross-entropy weights for low, medium, obstacle",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.dataset) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_idx, val_idx = split_indices(len(manifest["samples"]), args.val_ratio, args.seed, args.limit)

    train_set = RUGDTraversabilityDataset(
        manifest_path, train_idx, (args.height, args.width),
        cache_in_memory=args.cache_in_memory,
        cache_name="train")
    val_set = RUGDTraversabilityDataset(
        manifest_path, val_idx, (args.height, args.width),
        cache_in_memory=args.cache_in_memory,
        cache_name="val")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet(width=args.width_mult).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    class_weights = torch.tensor(
        [float(x.strip()) for x in args.class_weights.split(",")],
        device=device,
    )
    if class_weights.numel() != 3:
        raise ValueError("--class-weights must contain exactly three values")

    history = []
    best_miou = -1.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_pixels = 0
        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, targets, weight=class_weights, ignore_index=IGNORE_INDEX)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            valid_pixels = int((targets != IGNORE_INDEX).sum().item())
            running_loss += float(loss.item()) * valid_pixels
            running_pixels += valid_pixels
            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                print(
                    "epoch=%02d batch=%03d/%03d train_loss=%.4f"
                    % (epoch, batch_idx, len(train_loader), running_loss / max(1, running_pixels)),
                    flush=True,
                )

        metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, running_pixels),
            **metrics,
        }
        history.append(row)
        print(
            "epoch=%02d train_loss=%.4f val_loss=%.4f acc=%.3f miou=%.3f "
            "iou=[%.3f, %.3f, %.3f]"
            % (
                epoch,
                row["train_loss"],
                row["loss"],
                row["pixel_acc"],
                row["miou"],
                row["iou_low"],
                row["iou_medium"],
                row["iou_obstacle"],
            ),
            flush=True,
        )
        if row["miou"] > best_miou:
            best_miou = row["miou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "metrics": row,
                    "classes": manifest["classes"],
                },
                output_dir / "best_model.pt",
            )

    save_preview(model, val_set, output_dir / "preview", device)
    summary = {
        "dataset": str(args.dataset),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "device": str(device),
        "elapsed_sec": time.time() - start,
        "best_miou": best_miou,
        "history": history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote %s" % output_dir)


if __name__ == "__main__":
    main()

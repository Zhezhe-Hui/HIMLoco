#!/usr/bin/env python3
"""Fit static flat ground; output an UNAPPROVED calibration candidate.

Ground constrains height/roll/pitch only. X/Y and mounting yaw need verification.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from run_ours import rotation_rp, validate_calibration


def fit_ground(depth, intrinsics):
    fx, fy, cx, cy = intrinsics
    v, u = np.indices(depth.shape)
    good = (depth > 0.2) & (depth < 4) & (v > depth.shape[0]*0.5)
    z = depth[good]
    points = np.stack(((u[good]-cx)*z/fx, (v[good]-cy)*z/fy, z), axis=1)[::20]
    if len(points) < 300:
        raise ValueError("Not enough depth points for floor fitting")
    rng = np.random.RandomState(1)
    best = None
    for _ in range(500):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b-a, c-a)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n /= norm
        if n[1] > 0:
            n = -n
        offset = -n @ a
        if -n[1] < 0.8 or not 0.15 < offset < 0.9:
            continue
        hit = np.abs(points @ n + offset) < 0.015
        if best is None or hit.sum() > best[0].sum():
            best = (hit, n, offset)
    if best is None or best[0].mean() < 0.5:
        raise ValueError("No dominant plausible ground plane; keep motion disabled")
    inliers = points[best[0]]
    center = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers-center, full_matrices=False)
    n = vt[-1]
    if n[1] > 0:
        n = -n
    height = float(-n @ center)
    rms = float(np.sqrt(np.mean((inliers @ n + height)**2)))
    if rms > 0.02:
        raise ValueError("Ground fit residual too high")
    return n, height, float(best[0].mean()), rms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--camera-forward", type=float, default=0.30)
    parser.add_argument("--camera-left", type=float, default=0.0)
    args = parser.parse_args()
    f = np.load(args.run/"latest.npz")
    n, height, fraction, rms = fit_ground(f["depth"], f["intrinsics"])
    if "state_json" in f.files:
        state = json.loads(str(f["state_json"]))
    else:
        rows = [json.loads(line) for line in (args.run/"telemetry.jsonl").read_text().splitlines()]
        state = rows[-1]["state"]
    level = rotation_rp(state["roll"], state["pitch"])
    forward = np.array([0., 0., 1.])
    forward -= (forward @ n)*n
    forward /= np.linalg.norm(forward)
    left = np.cross(n, forward)
    rotation = level.T @ np.stack((forward, left, n))
    base_height = float(state["z"])
    tz = (height-base_height-level[2, 0]*args.camera_forward-level[2, 1]*args.camera_left)/level[2, 2]
    cal = dict(calibrated=False, xy_measured=False,
               note="Ground-fit candidate. Verify mounting yaw and physically measure X/Y before approval.",
               camera_translation_body_m=[args.camera_forward, args.camera_left, float(tz)],
               camera_rotation_optical_to_body=rotation.tolist(), standing_base_height_m=base_height,
               ground_camera_height_m=height, ground_inlier_fraction=fraction, ground_rms_m=rms,
               source_snapshot=str(args.run/"latest.npz"))
    validate_calibration(cal)
    args.out.write_text(json.dumps(cal, indent=2))
    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()

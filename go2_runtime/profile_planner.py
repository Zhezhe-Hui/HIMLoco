#!/usr/bin/env python3
"""Offline saved-frame benchmark: no camera, DDS, or command client."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import sys

import numpy as np
import run_ours


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--numpy-reference', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0,str(run_ours.ROOT/'go2_web'))
    from wired import WiredRuntime
    if args.numpy_reference:
        run_ours.bev_reducer = lambda: None
    else:
        assert run_ours.bev_reducer() is not None, 'Native backend missing'
    meta = json.loads(args.metadata.read_text())
    with np.load(args.snapshot) as z:
        depth, intrinsics = z['depth'].copy(), z['intrinsics'].copy()
        state = json.loads(str(z['state_json']))
    fake = SimpleNamespace(_module=lambda:run_ours, task={'kind':'shadow'})
    bev,fmm,mppi = WiredRuntime._controllers(fake, meta['parameters'], meta['calibration'])
    rows = []
    for i in range(22):
        now = time.monotonic()
        timing = {}
        result = run_ours.select_command(bev,fmm,mppi,depth,intrinsics,
            dict(state, received=now, state_age=0),
            dict(captured=now, received=now, clock_uncertainty=0),meta['calibration'],
            np.array([.5,0]),False,diagnostics=timing)
        if i>=2:
            rows.append(dict(total_ms=result[-1], **timing))
    report = dict(backend='numpy' if args.numpy_reference else 'native', samples=len(rows),
        timings={k:dict(mean=float(np.mean([r[k] for r in rows])),p95=float(np.percentile([r[k] for r in rows],95)))
                 for k in rows[0] if k.endswith('_ms')},
        obstacle_cells=int(result[2].sum()), samples_per_plan=mppi.num_samples,horizon=mppi.horizon)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

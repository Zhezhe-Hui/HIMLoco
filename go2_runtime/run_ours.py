#!/usr/bin/env python3
"""Real D435i -> existing dual-cost FMM -> standard footprint MPPI.

Default shadow mode never creates a velocity client. No Isaac Gym dependency.
"""
import argparse
import ctypes
from functools import lru_cache
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def algorithms():
    sys.path.insert(0, str(ROOT / "legged_gym"))
    folder = ROOT / "legged_gym/legged_gym/envs/go2"
    mapper = source_module("real_bev", folder / "bev_height_mapper.py")
    mppi = source_module("real_mppi", folder / "mppi_local_planner.py")
    cfg = source_module("real_viz", folder / "viz_config.py")
    return mapper, mppi, cfg


def rotation_rp(roll, pitch):
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    return np.array([[cp, sp*sr, sp*cr], [0, cr, -sr], [-sp, cp*sr, cp*cr]])


def validate_calibration(cal):
    rotation = np.asarray(cal["camera_rotation_optical_to_body"], dtype=float)
    translation = np.asarray(cal["camera_translation_body_m"], dtype=float)
    if (rotation.shape != (3, 3) or translation.shape != (3,) or
            not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or
            not np.isclose(np.linalg.det(rotation), 1, atol=1e-4) or
            not 0 < float(cal["standing_base_height_m"]) < 1):
        raise ValueError("Invalid calibration rotation/translation/height")


@lru_cache(maxsize=1)
def bev_reducer():
    """Optional CPU acceleration; missing library keeps the NumPy reference."""
    path = HERE / 'build/libbev_reduce.so'
    if not path.exists():
        return None
    lib = ctypes.CDLL(str(path))
    fn = lib.go2_bev_reduce
    array = np.ctypeslib.ndpointer
    fn.argtypes = [ctypes.c_size_t, ctypes.c_size_t,
                   array(np.int64, flags='C_CONTIGUOUS'), array(np.float64, flags='C_CONTIGUOUS'),
                   array(np.float32, flags='C_CONTIGUOUS'), array(np.float32, flags='C_CONTIGUOUS'),
                   array(np.int32, flags='C_CONTIGUOUS'), array(np.int32, flags='C_CONTIGUOUS')]
    fn.restype = ctypes.c_int
    return fn


def aggregate_points(bev, rows, cols, heights, use_native=True):
    elevated = np.zeros_like(bev.count)
    fn = bev_reducer() if use_native else None
    if fn is not None:
        ids = np.ascontiguousarray(rows.astype(np.int64) * bev.count.shape[1] + cols)
        heights = np.ascontiguousarray(heights, dtype=np.float64)
        if fn(len(ids), bev.count.size, ids, heights, bev.min_z, bev.max_z, bev.count, elevated):
            raise ValueError('BEV cell index outside grid')
    else:
        np.minimum.at(bev.min_z, (rows, cols), heights)
        np.maximum.at(bev.max_z, (rows, cols), heights)
        np.add.at(bev.count, (rows, cols), 1)
        mask = (heights > 0.15) & (heights < 1.20)
        np.add.at(elevated, (rows[mask], cols[mask]), 1)
    return elevated


def update_real_bev(bev, depth, intrinsics, state, cal):
    """Use actual optical intrinsics + SE(3); retain original BEV obstacle rules."""
    fx, fy, cx, cy = intrinsics
    if not np.all(np.isfinite(intrinsics)) or min(fx, fy) <= 0:
        raise ValueError("Invalid camera intrinsics")
    v, u = np.indices(depth.shape)
    keep = ((depth > bev.min_depth) & (depth < bev.max_depth) & np.isfinite(depth)
            & (v >= int(depth.shape[0] * bev.v_start_ratio)))
    z = depth[keep]
    optical = np.stack(((u[keep]-cx)*z/fx, (v[keep]-cy)*z/fy, z), axis=1)
    body = optical @ np.asarray(cal["camera_rotation_optical_to_body"]).T
    body += np.asarray(cal["camera_translation_body_m"])
    points = body @ rotation_rp(state["roll"], state["pitch"]).T
    points[:, 2] += float(cal["standing_base_height_m"])
    bev.reset()
    inside = ((points[:, 0] >= 0) & (points[:, 0] < bev.bev_x) &
              (np.abs(points[:, 1]) < bev.bev_y/2))
    points = points[inside]
    rows = np.floor(points[:, 0]/bev.bev_res).astype(int)
    cols = np.floor((points[:, 1]+bev.bev_y/2)/bev.bev_res).astype(int)
    obstacle_counts = aggregate_points(bev, rows, cols, points[:, 2])
    occ = bev.get_obstacle_map()
    # Real depth exposes a failure of the simulation height-range rule:
    # a vertical plane can occupy several cells with little within-cell height
    # variation. Ground-referenced elevated returns must also count as obstacles.
    occ[obstacle_counts >= bev.min_points] = 1
    return occ


def read_exact(stream, count):
    chunks = []
    while count:
        data = stream.read(count)
        if not data:
            raise EOFError("Camera SSH stream closed")
        chunks.append(data)
        count -= len(data)
    return b"".join(chunks)


class Latest:
    def __init__(self):
        self.value = None
        self.error = None
        self.lock = threading.Lock()

    def put(self, value):
        with self.lock:
            self.value = value

    def get(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            return self.value


def read_camera(process, latest, clock_offset, clock_uncertainty):
    previous = -1
    try:
        while True:
            header = read_exact(process.stdout, 56)
            magic, w, h, number, timestamp, captured, fx, fy, cx, cy, scale = struct.unpack("<IIIQddfffff", header)
            if magic != 0x48494d32 or (w, h) != (640, 480) or number <= previous or not 0 < scale < 1:
                raise ValueError("Invalid/repeated camera frame: magic=%x size=%sx%s frame=%s previous=%s scale=%s" %
                                 (magic, w, h, number, previous, scale))
            previous = number
            depth = np.frombuffer(read_exact(process.stdout, w*h*2), "<u2").reshape(h, w).astype(np.float32)*scale
            rgb = np.frombuffer(read_exact(process.stdout, w*h*3), np.uint8).reshape(h, w, 3)
            latest.put(dict(depth=depth, rgb=rgb, intrinsics=(fx, fy, cx, cy),
                            number=number, timestamp_ms=timestamp, received=time.monotonic(),
                            captured=captured+clock_offset, clock_uncertainty=clock_uncertainty))
    except Exception as exc:
        with latest.lock:
            latest.error = str(exc)


def read_state(process, latest):
    try:
        for line in process.stdout:
            # DDS may write initialization messages on stdout.
            if not line.startswith(b"{"):
                continue
            state = json.loads(line)
            if 'bridge_event' in state:
                raise RuntimeError('bridge_event:'+str(state['bridge_event']))
            required = [state[k] for k in ("x", "y", "z", "yaw", "roll", "pitch", "state_age", "vx", "vy", "yaw_speed")]
            if not np.all(np.isfinite(required)):
                raise ValueError("Non-finite robot state")
            state["received"] = time.monotonic()
            latest.put(state)
        raise EOFError("Robot bridge closed")
    except Exception as exc:
        with latest.lock:
            latest.error = str(exc)


def local_goal(goal_world, state):
    dx, dy = goal_world[0]-state["x"], goal_world[1]-state["y"]
    c, s = math.cos(state["yaw"]), math.sin(state["yaw"])
    return np.array([c*dx+s*dy, -s*dx+c*dy])


def select_command(bev, fmm, mppi, depth, intrinsics, state, frame, cal, goal, execute, diagnostics=None):
    start = time.monotonic()
    occ = update_real_bev(bev, depth, intrinsics, state, cal)
    bev_done = time.monotonic()
    fmm.set_goal(goal)
    fmm.update(occ)
    path = np.asarray([(r*fmm.bev_res, (c-fmm.W//2)*fmm.bev_res)
                       for r, c in fmm.extract_path()], dtype=float).reshape(-1, 2)
    fmm_done = time.monotonic()
    mppi_ms = check_ms = 0.0
    reason = "ok"
    command = (0.0, 0.0)
    valid_fraction = float(np.mean(np.isfinite(depth) & (depth > 0.05) & (depth < bev.max_depth)))
    h, w = depth.shape
    center = depth[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
    center_valid = center[np.isfinite(center) & (center > 0.05) & (center < bev.max_depth)]
    if center_valid.size < center.size * 0.3:
        reason = "insufficient_front_depth"
    elif float(np.percentile(center_valid, 5)) < 0.45:
        reason = "near_depth_stop"
    elif valid_fraction < 0.2:
        reason = "insufficient_depth"
    elif not fmm.is_reachable() or len(path) < 2:
        reason = "unreachable"
    elif goal[0] <= 0.1:
        reason = "goal_behind_or_too_close"
    else:
        mppi_start = time.monotonic()
        command = mppi.command(occ, path, goal_xy=goal)
        mppi_done = time.monotonic()
        controls = np.tile(command, (1, mppi.horizon, 1))
        if np.any(mppi.footprint_collision_mask(mppi._rollout(controls), occ)):
            command = (0.0, 0.0)
            reason = "command_footprint_collision"
        mppi_ms = (mppi_done-mppi_start)*1000
        check_ms = (time.monotonic()-mppi_done)*1000
    now = time.monotonic()
    camera_age = now-frame.get("captured", frame["received"])+frame.get("clock_uncertainty", 0)
    state_age = now-state["received"]+state["state_age"]
    if diagnostics is not None:
        diagnostics.update(bev_ms=(bev_done-start)*1000, fmm_ms=(fmm_done-bev_done)*1000,
                           mppi_ms=mppi_ms, command_check_ms=check_ms,
                           camera_age_start_ms=(start-frame.get('captured', frame['received'])+frame.get('clock_uncertainty',0))*1000,
                           state_age_start_ms=(start-state['received']+state['state_age'])*1000,
                           camera_age_end_ms=camera_age*1000, state_age_end_ms=state_age*1000,
                           stale_sources=[name for name, age in [('camera',camera_age),('state',state_age)] if age > .25])
    if (camera_age > 0.25 or state_age > 0.25):
        command, reason = (0.0, 0.0), "stale_input"
    elif abs(state["roll"]) > 0.45 or abs(state["pitch"]) > 0.45:
        command, reason = (0.0, 0.0), "excessive_tilt"
    elif execute and not 0.20 <= state["z"] <= 0.60:
        command, reason = (0.0, 0.0), "robot_not_standing"
    if not np.all(np.isfinite(command)):
        raise ValueError("Non-finite planner output")
    return command, reason, occ, path, valid_fraction, (time.monotonic()-start)*1000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interface", default="enx00e04c584b68")
    ap.add_argument("--host", default="unitree@192.168.123.18")
    ap.add_argument("--ssh-control", default="/tmp/himloco-ours-ssh")
    ap.add_argument("--calibration", type=Path, default=HERE/"calibration.example.json")
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--goal-forward", type=float, default=2)
    ap.add_argument("--goal-left", type=float, default=0)
    ap.add_argument("--max-vx", type=float, default=0.10)
    ap.add_argument("--min-vx", type=float, default=0.0,
                    help="Minimum non-stop MPPI sample speed; stopping overrides remain zero")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--preliminary-trial", action="store_true",
                    help="Operator-approved first trial only: <=0.5m, <=0.10m/s, <=10s; retains preliminary calibration status")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if not all(math.isfinite(v) for v in (args.duration, args.goal_forward, args.goal_left)) or not 0 < args.duration <= 120:
        ap.error("Finite goals and duration in (0,120] required")
    if not 0.3 <= args.goal_forward <= 5 or abs(args.goal_left) > 2:
        ap.error("Initial goal must be 0.3..5m forward, within 2m laterally")
    if not math.isfinite(args.max_vx) or not 0 < args.max_vx <= 0.20:
        ap.error("max-vx must be in (0, 0.20] m/s")
    if not math.isfinite(args.min_vx) or not 0 <= args.min_vx <= args.max_vx:
        ap.error("min-vx must be in [0, max-vx]")
    cal = json.loads(args.calibration.read_text())
    validate_calibration(cal)
    preliminary_trial = (args.preliminary_trial and args.duration <= 10 and
                         args.goal_forward <= 0.5 and args.goal_left == 0 and args.max_vx <= 0.10 and
                         cal.get("calibration_stage") == "preliminary_ground_fit_plus_user_tape_range" and
                         cal.get("extra_footprint_margin_m", 0) >= 0.04)
    if args.execute and cal.get("calibrated") is not True and not preliminary_trial:
        ap.error("Execution requires measured calibration with calibrated=true")
    if args.execute and cal.get("xy_measured") is not True:
        ap.error("Execution requires physically measured camera X/Y offsets")
    mapper, mppi_module, cfg = algorithms()
    bev = mapper.BEVMapper(bev_res=cfg.BEV_RES, bev_x=cfg.BEV_X, bev_y=cfg.BEV_Y,
                           height_range_thresh=cfg.BEV_HEIGHT_RANGE_THRESH,
                           min_points=cfg.BEV_MIN_POINTS, v_start_ratio=cfg.BEV_V_START_RATIO,
                           max_depth=cfg.BEV_MAX_DEPTH, min_depth=cfg.BEV_MIN_DEPTH,
                           enable_step_judge=cfg.BEV_ENABLE_STEP_JUDGE,
                           enable_memory_fusion=False)
    fmm = mapper.FMMGradientController(bev_res=cfg.BEV_RES, bev_x=cfg.BEV_X, bev_y=cfg.BEV_Y,
              use_soft_cost=True, hard_radius_m=cfg.FMM_HARD_RADIUS_M,
              soft_clearance_m=cfg.FMM_SOFT_CLEARANCE_M, soft_cost_weight=cfg.FMM_SOFT_COST_WEIGHT,
              soft_cost_power=cfg.FMM_SOFT_COST_POWER)
    mppi = mppi_module.FootprintMPPIController(bev_res=cfg.BEV_RES, bev_x=cfg.BEV_X, bev_y=cfg.BEV_Y,
              horizon=cfg.MPPI_HORIZON, num_samples=cfg.MPPI_NUM_SAMPLES,
              rollout_dt=cfg.MPPI_ROLLOUT_DT, max_vx=args.max_vx, min_vx=args.min_vx, max_wz=0.35,
              footprint_length_m=cfg.MPPI_FOOTPRINT_LENGTH_M,
              footprint_width_m=cfg.MPPI_FOOTPRINT_WIDTH_M,
              safety_margin_m=cfg.MPPI_FOOTPRINT_MARGIN_M+float(cal.get("extra_footprint_margin_m", 0.0)),
              temperature=cfg.MPPI_TEMPERATURE, noise_vx=cfg.MPPI_NOISE_VX,
              noise_wz=cfg.MPPI_NOISE_WZ, variant="standard")
    out = args.out or HERE/"runs"/time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(mode="execute" if args.execute else "shadow", calibration=cal,
                    preliminary_trial=preliminary_trial,
                    goal_relative=[args.goal_forward, args.goal_left], method="dual_cost_fmm_standard_footprint_mppi",
                    max_vx=args.max_vx, min_vx=args.min_vx, max_wz=0.35, temporal_memory=False, pose_source="Go2 SportModeState",
                    extra_footprint_margin_m=float(cal.get("extra_footprint_margin_m", 0.0)),
                    num_samples=mppi.num_samples, horizon=mppi.horizon, rollout_dt=mppi.rollout_dt,
                    hard_radius=fmm.hard_radius_m, soft_clearance=fmm.soft_clearance_m,
                    soft_weight=fmm.soft_cost_weight, soft_power=fmm.soft_cost_power,
                    note="Uncalibrated shadow runs are diagnostics, not navigation experiments")
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2))
    camera = robot = None
    frame_box, state_box = Latest(), Latest()
    rows = []
    failure = None
    stopped = threading.Event()
    def on_signal(*_):
        stopped.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        with (out/"camera.stderr.log").open("wb") as err:
            ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                   "-o", "ControlPath="+args.ssh_control, args.host]
            clock_samples = []
            for _ in range(3):
                before = time.monotonic()
                sample = subprocess.run(ssh+["python3 -c 'import time; print(time.monotonic())'"],
                                        stdout=subprocess.PIPE, stderr=err, check=True, timeout=8)
                after = time.monotonic()
                clock_samples.append(((after-before)/2, (after+before)/2-float(sample.stdout)))
            uncertainty, offset = min(clock_samples)
            if uncertainty > 0.10:
                raise RuntimeError("SSH clock synchronization uncertainty exceeds 100ms")
            metadata["clock_uncertainty_s"] = uncertainty
            (out/"metadata.json").write_text(json.dumps(metadata, indent=2))
            camera = subprocess.Popen(ssh+["exec /tmp/himloco_ours_camera_stream"], stdout=subprocess.PIPE, stderr=err)
            robot = subprocess.Popen([str(HERE/"build/robot_bridge"), args.interface] +
                                     (["--execute"] if args.execute else []), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            threading.Thread(target=read_camera, args=(camera, frame_box, offset, uncertainty), daemon=True).start()
            threading.Thread(target=read_state, args=(robot, state_box), daemon=True).start()
            deadline = time.monotonic()+15
            while frame_box.get() is None or state_box.get() is None:
                if stopped.is_set() or time.monotonic() > deadline:
                    raise TimeoutError("Waiting for camera/robot state")
                time.sleep(0.02)
            initial = state_box.get()
            goal_world = [initial["x"]+math.cos(initial["yaw"])*args.goal_forward-math.sin(initial["yaw"])*args.goal_left,
                          initial["y"]+math.sin(initial["yaw"])*args.goal_forward+math.cos(initial["yaw"])*args.goal_left]
            start = time.monotonic(); last_frame = -1; arrived = 0; last_print = -10
            with (out/"telemetry.jsonl").open("w", buffering=1) as log:
                while not stopped.is_set() and time.monotonic()-start < args.duration:
                    frame, state = frame_box.get(), state_box.get()
                    if time.monotonic()-frame["received"] > 0.25 or time.monotonic()-state["received"] > 0.25:
                        raise RuntimeError("Input stream became stale")
                    if frame["number"] == last_frame:
                        time.sleep(0.005); continue
                    last_frame = frame["number"]
                    goal = local_goal(goal_world, state)
                    command, reason, occ, path, valid, ms = select_command(
                        bev, fmm, mppi, frame["depth"], frame["intrinsics"], state, frame, cal, goal, args.execute)
                    near_goal = np.linalg.norm(goal) <= 0.20
                    settled = np.hypot(state["vx"], state["vy"]) < 0.05 and abs(state["yaw_speed"]) < 0.1
                    arrived = arrived+1 if near_goal and settled else 0
                    if near_goal:
                        command, reason = (0.0, 0.0), "goal_confirmation"
                    if args.execute:
                        robot.stdin.write(("%.6f %.6f\n" % command).encode()); robot.stdin.flush()
                    row = dict(elapsed=time.monotonic()-start, frame=last_frame, state=state,
                               goal=goal.tolist(), proposed_vx=command[0], proposed_wz=command[1],
                               sent=args.execute, reason=reason, obstacle_cells=int(occ.sum()),
                               valid_fraction=valid, plan_ms=ms,
                               frame_age_ms=(time.monotonic()-frame["received"])*1000)
                    row["capture_age_upper_ms"] = (time.monotonic()-frame["captured"]+frame["clock_uncertainty"])*1000
                    rows.append(row); log.write(json.dumps(row)+"\n")
                    if len(rows) == 1 or row["elapsed"]-last_print > 2:
                        print(json.dumps(row), flush=True); last_print = row["elapsed"]
                        np.savez_compressed(out/"latest.npz", depth=frame["depth"], rgb=frame["rgb"],
                                            occ=occ, path=path, min_z=bev.min_z, max_z=bev.max_z,
                                            intrinsics=frame["intrinsics"], soft_cost=fmm.soft_cost_map,
                                            state_json=json.dumps(state), mppi_plan=mppi.last_plan)
                    if arrived >= 10:
                        break
    except BaseException as exc:
        failure = str(exc)
        raise
    finally:
        if robot:
            try:
                if args.execute:
                    robot.stdin.write(b"0 0\n"); robot.stdin.flush()
                robot.stdin.close()
                robot.wait(timeout=2)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                robot.terminate()
        if camera:
            camera.terminate()
            try:
                camera.wait(timeout=3)
            except subprocess.TimeoutExpired:
                camera.kill(); camera.wait()
        summary = dict(status="failed" if failure else "completed", error=failure,
                       frames=len(rows), sent_velocity_commands=bool(args.execute and rows),
                       plan_ms_mean=float(np.mean([r["plan_ms"] for r in rows])) if rows else None,
                       plan_ms_p95=float(np.percentile([r["plan_ms"] for r in rows], 95)) if rows else None,
                       reasons={reason: sum(r["reason"] == reason for r in rows) for reason in set(r["reason"] for r in rows)})
        (out/"summary.json").write_text(json.dumps(summary, indent=2))
        print("RESULT", out, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Local experiment console. Offline control with optional read-only SSH camera.

Reads existing runs only. Simulation exercises UI/task recording, not MPPI physics.
"""
import argparse
import base64
import fcntl
import socket
import struct
from datetime import datetime
import io
import json
import math
from pathlib import Path
import secrets
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote
import zipfile

import numpy as np
from PIL import Image
from camera import ReadOnlyCamera
from access import Access, COOKIE, create_access

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "go2_runtime/runs"
DEFAULTS = dict(forward=0.5, lateral=0.0, max_vx=0.10, duration=10.0)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_bytes(value):
    return json.dumps(clean(value), ensure_ascii=False, allow_nan=False).encode()


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return fallback


def parameters(raw, max_vx_limit=0.20):
    if not isinstance(raw, dict) or set(raw)-set(DEFAULTS):
        raise ValueError("参数字段无效")
    values = dict(DEFAULTS, **raw)
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("参数必须是有限数字")
    if not 0.3 <= values["forward"] <= 5:
        raise ValueError("前方距离应为 0.3–5 米")
    if abs(values["lateral"]) > 2:
        raise ValueError("横向距离应在 ±2 米内")
    if not 0.01 <= values["max_vx"] <= max_vx_limit:
        raise ValueError("速度上限应为 0.01–%.2f m/s" % max_vx_limit)
    if not 1 <= values["duration"] <= 60:
        raise ValueError("模拟时长应为 1–60 秒")
    return {k: float(v) for k, v in values.items()}


def png_bytes(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


class OfflineConsole:
    def __init__(self, runs=RUNS, data=HERE/"data", camera=None, real=None):
        self.runs_root = Path(runs)
        self.data = Path(data)
        self.data.mkdir(parents=True, exist_ok=True)
        self.db = self.data/"console.sqlite3"
        self.lock = threading.RLock()
        self.active = None
        self.recording = True
        self.history = {}
        self.assets = {}
        self.camera = camera or ReadOnlyCamera(enabled=False)
        self.real = real
        with self.connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS simulations (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self.load_history()
        self.draft = parameters(self.setting("draft", DEFAULTS))

    def connection(self):
        return sqlite3.connect(str(self.db), timeout=10)

    def setting(self, key, default):
        with self.connection() as conn:
            row = conn.execute("SELECT payload FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def save_setting(self, key, value):
        with self.connection() as conn:
            conn.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, json_bytes(value).decode()))

    def load_history(self):
        for folder in sorted(self.runs_root.glob("*")):
            rows_file = folder/"telemetry.jsonl"
            if not rows_file.is_file():
                continue
            rows = []
            try:
                for line in rows_file.read_text().splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
            except (OSError, ValueError):
                continue
            if not rows:
                continue
            summary = read_json(folder/"summary.json", {})
            meta = read_json(folder/"metadata.json", {})
            motion = read_json(folder/"motion_analysis.json", {})
            is_walk = folder.name == "ours_first_walk_20261030"
            title = {"ours_ready_shadow_20261030": "初步标定 · ours 影子规划",
                     "ours_final_shadow_20261030": "短目标 · 数据时效检查",
                     "ours_first_walk_20261030": "首次速度试验 · 未迈步",
                     "shadow_groundfit_20261030": "地面拟合 · 感知检查"}.get(folder.name, folder.name)
            entry = dict(id=folder.name, title=title, source="historical", method="双层代价 FMM + 标准足迹 MPPI",
                         outcome="no_steps" if is_walk else "historical_only",
                         outcome_label="未迈步" if is_walk else "历史记录",
                         created="2026-10-30", duration=float(rows[-1].get("elapsed", 0)),
                         sample_count=len(rows), summary=summary, metadata=meta, motion=motion,
                         rows=rows, snapshot=False, snapshot_frame=None,
                         note="用户现场确认：仅前倾后收回，未迈步。里程计位移不代表实际行走。" if is_walk else
                              "原始实验遥测回放；画面为最近留存的一张快照，不是连续录像。")
            self.history[folder.name] = entry
            if folder.name.startswith('web_'):
                entry.update(title='网页实机影子规划' if meta.get('mode')=='shadow' else '网页实机短距离测试',
                             outcome=summary.get('status','historical_only'),outcome_label=summary.get('outcome_label','实机记录'),
                             created=meta.get('created',folder.name),note='真实采集与计算；写入速度桥不代表已迈步，里程计到达需现场核验。')
            npz = folder/"latest.npz"
            if not npz.exists():
                continue
            try:
                with np.load(npz, allow_pickle=False) as data:
                    self.assets[(folder.name, "rgb")] = png_bytes(data["rgb"])
                    depth = data["depth"].astype(float)
                    value = np.clip(depth/4, 0, 1)
                    # Blue/teal/amber depth preview; zero is invalid, not near ground.
                    color = np.stack((np.clip((value-.45)*2, 0, 1),
                                      np.sin(value*np.pi)*.85, 1-value*.85), axis=-1)
                    color[~np.isfinite(depth) | (depth <= 0)] = (0.04, 0.06, 0.10)
                    self.assets[(folder.name, "depth")] = png_bytes((np.nan_to_num(color)*255).astype(np.uint8))
                    cells = np.argwhere(data["occ"] > 0)
                    scene = dict(occupied=cells.tolist(), shape=list(data["occ"].shape),
                                 path=data["path"].tolist(),
                                 mppi=data["mppi_plan"].tolist() if "mppi_plan" in data.files else [],
                                 soft_cost=np.nan_to_num(data["soft_cost"][::2, ::2]).tolist(),
                                 resolution=0.05, extent=[6, 6])
                    entry["scene"] = scene
                    entry["snapshot"] = True
                    if "state_json" in data.files:
                        recorded = json.loads(str(data["state_json"]))
                        closest = min(rows, key=lambda row: abs(row.get("state", {}).get("received", -1)-recorded.get("received", 0)))
                        entry["snapshot_frame"] = closest.get("frame")
            except (OSError, ValueError, KeyError):
                # A broken snapshot does not hide its valuable telemetry.
                entry["snapshot"] = False

    def saved_simulations(self):
        with self.connection() as conn:
            return [json.loads(row[0]) for row in conn.execute("SELECT payload FROM simulations ORDER BY rowid DESC")]

    def get_experiment(self, experiment_id):
        with self.lock:
            if experiment_id in self.history:
                return self.history[experiment_id]
            if self.active and self.active["id"] == experiment_id:
                return clean(self.active)
            with self.connection() as conn:
                row = conn.execute("SELECT payload FROM simulations WHERE id=?", (experiment_id,)).fetchone()
            if row:
                return json.loads(row[0])
        raise KeyError("实验不存在")

    @staticmethod
    def brief(entry):
        return {k: clean(v) for k, v in entry.items() if k not in ("rows", "scene", "start_monotonic")}

    def experiments(self):
        with self.lock:
            items = list(self.history.values())+self.saved_simulations()
            if self.active and not any(v["id"] == self.active["id"] for v in items):
                items.append(self.active)
            priority = {"ours_ready_shadow_20261030": 3, "ours_first_walk_20261030": 2, "ours_final_shadow_20261030": 1}
            items.sort(key=lambda x: (x["source"] == "simulation", priority.get(x["id"], 0), x["id"]), reverse=True)
            return [self.brief(e) for e in items]

    def status(self):
        with self.lock:
            real_state=self.real.status() if self.real else dict(enabled=False)
            real_task=real_state.get('task') or {}
            camera_state=self.camera.status()
            if self.real:
                camera_state.update(recording=bool(real_task.get('record_rgbd') and real_task.get('outcome')=='running'),recording_mode='rgbd_snapshot_2hz')
            return dict(offline=not bool(self.real), robot_connected=real_state.get('connected',False),
                        motion_enabled=real_task.get('kind')=='execute' and real_task.get('outcome')=='running',
                        physical_commands_sent=real_state.get('physical_commands_sent',0),
                        recording=self.recording, draft=self.draft,
                        camera=camera_state,
                        real=real_state,
                        runtime=dict(location="robot" if self.camera.local_binary else "computer",
                                     camera_transport="local_usb" if self.camera.local_binary else "ssh"),
                        active=clean(self.active) if self.active else None,
                        history_count=len(self.history),
                        method=dict(name="双层代价 FMM + 标准足迹 MPPI", samples=384, horizon=20,
                                    rollout_dt=0.10, hard_radius=0.20, soft_clearance=0.65,
                                    soft_weight=4.0, resolution=0.05, footprint=[0.72, 0.42]),
                        capabilities=dict(history=True, simulation=True, physical_control=bool(self.real),
                                          rosbag=False, continuous_video=False),
                        notice=("实机模式：影子规划不发送速度；实际行走需要本次标定复核和现场确认。" if self.real else
                                "运动任务仅流程模拟；相机是否实时以相机状态为准，没有运动控制接口。"))

    def start(self, payload):
        if self.real and self.real.running(): raise ValueError('真实任务运行中，不能启动流程模拟')
        if not isinstance(payload, dict) or set(payload)-{"kind", "parameters"}:
            raise ValueError("任务字段无效")
        kind = payload.get("kind", "simulation")
        if kind not in ("simulation", "shadow"):
            raise ValueError("只支持离线流程模拟")
        params = parameters(payload.get("parameters", {}))
        with self.lock:
            if self.active and self.active["outcome"] == "running":
                raise RuntimeError("已有模拟任务，请先停止")
            self.draft = params
            self.save_setting("draft", params)
            task_id = "sim_"+datetime.now().strftime("%Y%m%d_%H%M%S_")+secrets.token_hex(3)
            self.active = dict(id=task_id, title="影子流程模拟" if kind == "shadow" else "短目标流程模拟",
                               source="simulation", kind=kind, parameters=params,
                               created=datetime.now().astimezone().isoformat(timespec="seconds"),
                               duration=0, sample_count=0, outcome="running", outcome_label="模拟运行中",
                               start_monotonic=time.monotonic(), rows=[], snapshot=False,
                               recording_enabled_at_start=self.recording,
                               note="纯流程模型生成的数据，不运行 FMM/MPPI，不代表算法性能或机器人运动。",
                               summary=dict(physical_commands_sent=0, recorded_samples=0))
            task = self.active
            threading.Thread(target=self.simulate, args=(task,), daemon=True).start()
            return self.brief(task)

    def persist(self, task):
        serializable = {k: v for k, v in task.items() if k != "start_monotonic"}
        with self.connection() as conn:
            conn.execute("INSERT OR REPLACE INTO simulations VALUES (?,?)", (task["id"], json_bytes(serializable).decode()))

    def simulate(self, task):
        while True:
            with self.lock:
                if task["outcome"] != "running":
                    return
                elapsed = min(time.monotonic()-task["start_monotonic"], task["parameters"]["duration"])
                params = task["parameters"]
                vx = params["max_vx"]*min(elapsed/0.5, 1)
                distance = params["max_vx"]*(elapsed*elapsed if elapsed < .5 else elapsed-.25)
                target = math.hypot(params["forward"], params["lateral"])
                shadow = task["kind"] == "shadow"
                progress = 0 if shadow else min(distance/target, 1)
                actual = 0 if shadow else vx
                row = dict(elapsed=elapsed, frame=task["sample_count"]+1,
                           proposed_vx=vx, proposed_wz=0, sent=False, reason="simulated",
                           state=dict(x=params["forward"]*progress, y=params["lateral"]*progress,
                                      yaw=0, roll=0, pitch=0, vx=actual, vy=0, mode=0, gait_type=0),
                           goal=[params["forward"]*(1-progress), params["lateral"]*(1-progress)],
                           # Performance fields intentionally absent: no planner was run.
                           source="simulation")
                task["current"] = row
                task["duration"] = elapsed
                task["progress"] = progress
                task["sample_count"] += 1
                if self.recording:
                    task["rows"].append(row)
                done = (not shadow and progress >= 1) or elapsed >= params["duration"]
                if done:
                    task["outcome"] = "simulated_complete" if shadow or progress >= 1 else "simulated_timeout"
                    task["outcome_label"] = "模拟完成" if shadow or progress >= 1 else "模拟超时"
                    task["current"] = dict(row, proposed_vx=0, proposed_wz=0, state=dict(row["state"], vx=0))
                    task["summary"]["recorded_samples"] = len(task["rows"])
                    self.persist(task)
                    return
            time.sleep(0.1)

    def stop(self):
        with self.lock:
            if self.active and self.active["outcome"] == "running":
                self.active["outcome"] = "stopped"
                self.active["outcome_label"] = "模拟已停止"
                current = self.active.get("current", {})
                self.active["current"] = dict(current, proposed_vx=0, proposed_wz=0,
                                               state=dict(current.get("state", {}), vx=0, vy=0))
                self.active["summary"]["recorded_samples"] = len(self.active["rows"])
                self.persist(self.active)
            return dict(ok=True, physical_commands_sent=0, message="模拟任务已停止；未发送机器人指令")

    def export(self, experiment_id):
        entry = self.get_experiment(experiment_id)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("metadata.json", json_bytes(self.brief(entry)))
            archive.writestr("telemetry.jsonl", b"\n".join(json_bytes(row) for row in entry.get("rows", [])))
            archive.writestr("README.txt", ("source="+entry["source"]+"\n历史图像仅为留存快照；模拟数据不代表算法或实机表现。\n").encode())
            if entry["source"] == "historical":
                folder=self.runs_root/experiment_id
                if experiment_id.startswith('web_'):
                    for name in ('metadata.json','summary.json','latest.npz'):
                        path=folder/name
                        if path.is_file(): archive.write(path,'real/'+name)
                    for path in sorted((folder/'frames').glob('*.npz')):
                        archive.write(path,'real/frames/'+path.name)
                for kind in ("rgb", "depth"):
                    asset = self.assets.get((experiment_id, kind))
                    if asset:
                        archive.writestr(kind+"_snapshot.png", asset)
                if "scene" in entry:
                    archive.writestr("snapshot_scene.json", json_bytes(entry["scene"]))
        return stream.getvalue()

    def capture_desktop(self):
        frame = self.camera.latest_frame()
        if frame is None or not 0 <= time.monotonic()-frame['received'] <= 1:
            raise ValueError('没有新鲜的实时相机画面，请先连接相机')
        try:
            desktop = Path(subprocess.check_output(['xdg-user-dir', 'DESKTOP'], text=True).strip())
        except (OSError, subprocess.SubprocessError):
            desktop = Path.home()/'桌面'
        parent = self.data/'captures' if self.camera.local_binary else desktop/'GO2相机截图'
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
        target = parent/(stamp+'_frame_'+str(frame['number']))
        with tempfile.TemporaryDirectory(prefix='.capture-', dir=parent) as temporary:
            folder = Path(temporary)
            depth = frame['depth']
            Image.fromarray(frame['rgb']).save(folder/'rgb.png')
            # Store metric depth independently from the visible colour rendering.
            mm = np.clip(np.rint(np.nan_to_num(depth, nan=0, posinf=0, neginf=0)*1000), 0, 65535).astype(np.uint16)
            Image.fromarray(mm).save(folder/'depth_mm.png')
            value = np.clip(depth/4, 0, 1)
            color = np.stack((np.clip((value-.45)*2, 0, 1), np.sin(value*np.pi)*.85, 1-value*.85), axis=-1)
            color[~np.isfinite(depth) | (depth <= 0)] = (0.04, 0.06, 0.10)
            Image.fromarray((np.nan_to_num(color)*255).astype(np.uint8)).save(folder/'depth_color.png')
            (folder/'metadata.json').write_text(json.dumps(dict(
                captured_at=datetime.now().astimezone().isoformat(), frame=frame['number'],
                intrinsics=list(frame['intrinsics']), rgb_shape=list(frame['rgb'].shape),
                depth_shape=list(depth.shape), depth_unit='millimetres', invalid_depth=0,
                source='live_camera_same_frame'), ensure_ascii=False, indent=2))
            folder.rename(target)
        return dict(ok=True, directory=str(target), frame=frame['number'],
                    download='/api/camera/captures/'+target.name+'.zip' if self.camera.local_binary else None,
                    files=['rgb.png','depth_color.png','depth_mm.png','metadata.json'])

    def capture_archive(self, name):
        if not name.endswith('.zip') or not name[:-4] or any(c not in '0123456789_frame' for c in name[:-4]):
            raise KeyError('截图不存在')
        folder=self.data/'captures'/name[:-4]
        if not folder.is_dir(): raise KeyError('截图不存在')
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as archive:
            for filename in ('rgb.png','depth_color.png','depth_mm.png','metadata.json'):
                archive.write(folder/filename, filename)
        return stream.getvalue()


class ConsoleServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, access=None, allowed_hosts=(), allow_interface_addresses=False):
        self.access = access
        self.allowed_hosts = set(allowed_hosts)|{"localhost","127.0.0.1"}
        self.allow_interface_addresses = allow_interface_addresses
        self.interface_hosts = set()
        self.interface_checked = 0
        super().__init__(address, handler)

    def allowed_host(self, host):
        if host in self.allowed_hosts: return True
        if not self.allow_interface_addresses: return False
        now=time.monotonic()
        if now-self.interface_checked>3:
            addresses=set()
            with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:
                for _,name in socket.if_nameindex():
                    try:
                        result=fcntl.ioctl(probe.fileno(),0x8915,struct.pack('256s',name.encode()[:15]))
                        addresses.add(socket.inet_ntoa(result[20:24]))
                    except OSError:
                        continue
            self.interface_hosts=addresses
            self.interface_checked=now
        return host in self.interface_hosts


def handler_for(console):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, data, content_type="application/json; charset=utf-8", attachment=None, cookie=None):
            body = json_bytes(data) if content_type.startswith("application/json") else data
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if cookie:
                self.send_header("Set-Cookie",cookie)
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
            if attachment:
                self.send_header("Content-Disposition", 'attachment; filename="'+attachment+'"')
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            try:
                host = urlparse("http://"+self.headers.get("Host",""))
                valid = self.server.allowed_host(host.hostname) and (host.port or 80)==self.server.server_address[1]
            except ValueError:
                valid = False
            if not valid:
                self.respond(403,dict(error="访问地址未在服务器允许列表中"))
            return valid

        def authorized(self):
            return self.server.access is None or self.server.access.valid(self.headers.get("Cookie"))

        def do_GET(self):
            if not self.valid_host(): return
            route = unquote(urlparse(self.path).path)
            if route in ("/login.js","/login.css"):
                self.respond(200,(HERE/"static"/route[1:]).read_bytes(),"text/javascript" if route.endswith(".js") else "text/css")
                return
            if route == "/api/auth":
                self.respond(200,dict(required=self.server.access is not None,authenticated=self.authorized()))
                return
            if not self.authorized():
                if route == "/":
                    self.respond(200,(HERE/"static/login.html").read_bytes(),"text/html; charset=utf-8")
                else:
                    self.respond(401,dict(error="请先登录实验台"))
                return
            try:
                static = {"/": ("index.html", "text/html; charset=utf-8"),
                          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                          "/style.css": ("style.css", "text/css; charset=utf-8")}
                if route in static:
                    filename, mime = static[route]
                    self.respond(200, (HERE/"static"/filename).read_bytes(), mime)
                elif route == "/api/state":
                    self.respond(200, console.status())
                elif route == "/api/experiments":
                    if console.real and console.real.task and not console.real.running() and console.real.task['id'] not in console.history:
                        with console.lock: console.load_history()
                    self.respond(200, dict(items=console.experiments()))
                elif route.startswith('/api/camera/captures/'):
                    name=route.rsplit('/',1)[1]
                    self.respond(200,console.capture_archive(name),'application/zip',name)
                elif route in ("/api/camera/rgb","/api/camera/depth"):
                    self.respond(200,console.camera.image(route.rsplit("/",1)[1]),"image/jpeg")
                elif route == '/api/camera/preview':
                    number, images = console.camera.preview_pair()
                    if not images:
                        self.respond(200, dict(frame=number, images={}))
                    elif self.headers.get('If-None-Match') == '"'+str(number)+'"':
                        self.send_response(304);self.send_header('Content-Length','0');self.end_headers()
                    else:
                        body=json_bytes(dict(frame=number,images={k:'data:image/jpeg;base64,'+base64.b64encode(v).decode('ascii') for k,v in images.items()}))
                        self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store')
                        self.send_header('ETag','"'+str(number)+'"');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
                elif route.startswith("/api/experiments/"):
                    parts = route.split("/")
                    experiment_id = parts[3]
                    if len(parts) == 4:
                        self.respond(200, console.get_experiment(experiment_id))
                    elif len(parts) == 5 and parts[4] == "export":
                        self.respond(200, console.export(experiment_id), "application/zip", "himloco_"+experiment_id+".zip")
                    else:
                        self.respond(404, dict(error="接口不存在"))
                elif route.startswith("/api/assets/"):
                    parts = route.split("/")
                    if len(parts) != 5 or parts[4] not in ("rgb", "depth"):
                        raise KeyError("图像不存在")
                    self.respond(200, console.assets[(parts[3], parts[4])], "image/png")
                else:
                    self.respond(404, dict(error="接口不存在"))
            except KeyError:
                self.respond(404, dict(error="数据不存在"))

        def do_POST(self):
            if not self.valid_host(): return
            origin = self.headers.get("Origin")
            if origin:
                try:
                    parsed = urlparse(origin)
                    current = urlparse("http://"+self.headers.get("Host",""))
                    valid = parsed.scheme=="http" and parsed.hostname==current.hostname and (parsed.port or 80)==(current.port or 80)
                except ValueError:
                    valid = False
                if not valid:
                    self.respond(403, dict(error="只接受当前页面的同源请求")); return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 0 or size > 8192:
                    raise ValueError("请求过大")
                if not self.headers.get("Content-Type", "").startswith("application/json"):
                    raise ValueError("请使用 JSON 请求")
                def reject_constant(value):
                    raise ValueError("不允许非有限数字")
                payload = json.loads(self.rfile.read(size), parse_constant=reject_constant)
                route = urlparse(self.path).path
                if route.startswith('/api/real/'):
                    if not self.authorized():
                        self.respond(401,dict(error='请先登录'));return
                    if not console.real:
                        raise ValueError('真实功能未启用，请从电脑运行 go2_runtime/manage.sh web-wired')
                    if route=='/api/real/connect':
                        if payload!={}: raise ValueError('不接受网络或命令参数')
                        result=console.real.connect()
                    elif route=='/api/real/disconnect':
                        if payload!={}: raise ValueError('断开请求应为空')
                        result=console.real.disconnect()
                    elif route=='/api/real/start':
                        if console.active and console.active['outcome']=='running': raise ValueError('先停止流程模拟')
                        result=console.real.start(payload)
                    elif route=='/api/real/stop':
                        if payload!={}: raise ValueError('停止请求应为空')
                        result=console.real.stop()
                    elif route=='/api/real/heartbeat':
                        if not isinstance(payload,dict) or set(payload)!={'lease'}: raise ValueError('会话字段无效')
                        result=console.real.heartbeat(payload['lease'])
                    elif route=='/api/real/calibration/save': result=console.real.save_measurements(payload)
                    elif route=='/api/real/calibration/fit':
                        if payload!={}: raise ValueError('拟合请求应为空')
                        result=console.real.fit_ground()
                    elif route=='/api/real/calibration/approve': result=console.real.approve_calibration(payload)
                    else: raise ValueError('实机接口不存在')
                    self.respond(200,dict(ok=True,result=result));return
                if route == "/api/auth/login":
                    if self.server.access is None:
                        self.respond(200,dict(ok=True)); return
                    if not isinstance(payload,dict) or set(payload)!={"username","password"}:
                        raise ValueError("请输入账号和密码")
                    token = self.server.access.login(payload["username"],payload["password"],self.client_address[0])
                    self.respond(200,dict(ok=True),cookie=COOKIE+"="+token+"; Path=/; HttpOnly; SameSite=Strict; Max-Age="+str(self.server.access.lifetime))
                    return
                if not self.authorized():
                    self.respond(401,dict(error="登录已失效，请重新登录")); return
                if route == "/api/auth/logout":
                    if self.server.access: self.server.access.logout(self.headers.get("Cookie"))
                    self.respond(200,dict(ok=True),cookie=COOKIE+"=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0")
                elif route == "/api/tasks/start":
                    self.respond(200, dict(ok=True, task=console.start(payload)))
                elif route == "/api/tasks/stop":
                    self.respond(200, console.stop())
                elif route == "/api/recording":
                    if not isinstance(payload, dict) or set(payload) != {"enabled"} or type(payload["enabled"]) is not bool:
                        raise ValueError("enabled 应为布尔值")
                    with console.lock:
                        console.recording = payload["enabled"]
                    self.respond(200, dict(ok=True, enabled=console.recording, source="simulation"))
                elif route == '/api/camera/capture':
                    if payload != {}: raise ValueError('截图请求应为空，不接受保存路径')
                    self.respond(200, console.capture_desktop())
                elif route == "/api/parameters":
                    draft = parameters(payload)
                    with console.lock:
                        console.draft = draft
                        console.save_setting("draft", draft)
                    self.respond(200, dict(ok=True, parameters=draft, scope="offline_only"))
                elif route in ("/api/camera/connect","/api/camera/disconnect"):
                    if console.real: raise ValueError('有线模式请使用真实链路的连接/断开入口')
                    if payload != {}:
                        raise ValueError("相机地址和命令由服务端固定配置，不接受网页参数")
                    result = console.camera.connect() if route.endswith("/connect") else console.camera.disconnect()
                    self.respond(200,dict(ok=True,camera=result,physical_commands_sent=0))
                else:
                    self.respond(403, dict(error="离线版不提供机器人控制或系统命令接口"))
            except (ValueError, TypeError) as exc:
                self.respond(400, dict(error=str(exc)))
            except RuntimeError as exc:
                self.respond(409, dict(error=str(exc)))
    return Handler


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bind",default="127.0.0.1")
    parser.add_argument("--allow-host",action="append",default=[],help="Allowed browser host/IP, without port")
    parser.add_argument("--allow-interface-addresses",action="store_true",help="Allow IPv4 addresses actually assigned to this Linux host")
    parser.add_argument("--auth-file",type=Path)
    parser.add_argument("--init-access",action="store_true")
    parser.add_argument("--runs",type=Path,default=RUNS)
    parser.add_argument("--camera-local",type=Path,help="Run fixed camera binary locally on the robot")
    parser.add_argument("--data", type=Path, default=HERE/"data")
    parser.add_argument("--enable-camera",action="store_true",help="Enable explicitly requested read-only camera connection")
    parser.add_argument("--wired",action="store_true",help="Laptop-only actual planning and bounded execution")
    parser.add_argument("--board",action="store_true",help="Robot-local USB, perception and bounded execution")
    parser.add_argument("--autoconnect",action="store_true",help="Connect read-only camera/state at startup, never start a task")
    parser.add_argument("--interface",default="enx00e04c584b68")
    parser.add_argument("--camera-host",default="unitree@192.168.123.18")
    parser.add_argument("--ssh-control",default="/tmp/himloco-ours-ssh")
    args = parser.parse_args()
    if args.init_access:
        args.auth_file = create_access(args.data)
    if args.bind not in ("127.0.0.1","localhost") and not args.auth_file and not args.board:
        parser.error("局域网访问必须配置 --auth-file 或 --init-access")
    access = Access(args.auth_file) if args.auth_file else None
    if args.board and (args.wired or not args.camera_local):
        parser.error('板载实机模式需要本地相机程序，不能同时使用--wired')
    if args.wired and (args.bind!='127.0.0.1' or access is not None or args.camera_local):
        parser.error('有线实机模式仅允许电脑127.0.0.1、无登录、不使用板载本地相机模式')
    camera=ReadOnlyCamera(args.enable_camera or args.wired or args.board,args.camera_host,args.ssh_control,args.camera_local,sync_clock=args.wired or args.board)
    real=None
    if args.wired or args.board:
        from wired import WiredRuntime
        real=WiredRuntime(camera,args.data,interface=args.interface)
    console = OfflineConsole(runs=args.runs,data=args.data,camera=camera,real=real)
    server = ConsoleServer((args.bind, args.port), handler_for(console),access=access,allowed_hosts=args.allow_host,allow_interface_addresses=args.allow_interface_addresses)
    print(f"HIMLoco offline console: http://127.0.0.1:{server.server_address[1]}", flush=True)
    print(f"历史实验 {len(console.history)} 组；{'有线真实规划，实际执行需现场手动启动' if real else '无运动接口'}；相机入口 {'启用' if camera.enabled else '禁用'}；Ctrl+C 停止。", flush=True)
    if args.autoconnect:
        if real is None: parser.error('--autoconnect需要实机模式')
        try: real.connect()
        except Exception as exc: print('只读连接暂不可用，可从网页重试：'+str(exc),flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        console.stop()
        if real: real.close()
        console.camera.disconnect()
        server.server_close()


if __name__ == "__main__":
    main()

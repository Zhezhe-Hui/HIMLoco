"""Optional read-only camera preview. Fixed SSH command, no robot control APIs."""
from collections import deque
import io
import struct
import subprocess
import tempfile
import threading
import time

import numpy as np
from PIL import Image


def read_exact(stream, size):
    result = bytearray()
    while len(result) < size:
        block = stream.read(size-len(result))
        if not block:
            raise EOFError("相机流已结束")
        result.extend(block)
    return bytes(result)


def encode_previews(depth, rgb):
    value = np.clip(depth/4, 0, 1)
    color = np.stack((np.clip((value-.45)*2, 0, 1), np.sin(value*np.pi)*.85, 1-value*.85), axis=-1)
    color[~np.isfinite(depth) | (depth <= 0)] = (0.04, 0.06, 0.10)
    result = {}
    for kind, array in (("rgb",rgb),("depth",(np.nan_to_num(color)*255).astype(np.uint8))):
        out = io.BytesIO()
        Image.fromarray(array).save(out, "JPEG", quality=95, subsampling=0)
        result[kind] = out.getvalue()
    return result


class ReadOnlyCamera:
    def __init__(self, enabled=False, host="unitree@192.168.123.18", control="/tmp/himloco-ours-ssh", local_binary=None, sync_clock=False):
        self.enabled, self.host, self.control = enabled, host, control
        self.local_binary = local_binary
        self.sync_clock = sync_clock
        self.raw_frame = None
        self.lock = threading.RLock()
        self.process = self.thread = None
        self.stop_event = threading.Event()
        self.images = {}
        self.received = None
        self.preview_number = self.frame_number = 0
        self.times = deque(maxlen=30)
        self.phase = "idle"
        self.error = ""

    def status(self):
        with self.lock:
            age = (time.monotonic()-self.received)*1000 if self.received is not None else None
            phase = self.phase
            if phase == "streaming" and age is not None and age > 1500:
                phase = "stale"
            fps = (len(self.times)-1)/(self.times[-1]-self.times[0]) if len(self.times)>1 and self.times[-1]>self.times[0] else None
            return dict(enabled=self.enabled,phase=phase if self.enabled else "disabled",
                        connected=phase=="streaming",has_frame=bool(self.images),
                        frame=self.frame_number,preview_frame=self.preview_number,
                        received_age_ms=age,received_fps=fps,error=self.error,
                        native_resolution=[640,480],preview_resolution=[640,480],preview_target_fps=15,
                        read_only=True,recording=False)

    def connect(self):
        if not self.enabled:
            raise ValueError("实时预览未启用，请使用 go2_runtime/manage.sh web-wired 启动")
        with self.lock:
            if self.thread and self.thread.is_alive():
                return self.status()
            self.stop_event = threading.Event()
            self.images = {}
            self.raw_frame = None
            self.received = None
            self.times.clear()
            self.error = ""
            self.phase = "connecting"
            self.frame_number = self.preview_number = 0
            self.thread = threading.Thread(target=self._receive,daemon=True)
            self.thread.start()
            return self.status()

    def disconnect(self):
        self.stop_event.set()
        with self.lock:
            process, thread = self.process, self.thread
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=2)
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3)
        with self.lock:
            self.phase = "idle"
            self.images = {}
            self.raw_frame = None
            self.error = ""
        return self.status()

    def image(self, kind):
        with self.lock:
            if kind not in ("rgb","depth") or kind not in self.images:
                raise KeyError("尚无实时图像")
            return self.images[kind]

    def latest_frame(self):
        with self.lock:
            return dict(self.raw_frame) if self.raw_frame is not None else None

    def preview_pair(self):
        with self.lock:
            return self.preview_number, dict(self.images)

    def _receive(self):
        process = None
        error = None
        with tempfile.TemporaryFile() as errors:
            try:
                # Both address and command are server configuration, never browser input.
                command = ["ssh","-T","-o","BatchMode=yes","-o","ConnectTimeout=5",
                           "-o","StrictHostKeyChecking=yes","-o","ControlPath="+self.control,
                           self.host,"exec /home/unitree/.local/lib/himloco/camera_stream"]
                if self.local_binary:
                    command = [str(self.local_binary)]
                offset, uncertainty = 0., 0. if self.local_binary else float('inf')
                if self.sync_clock and not self.local_binary:
                    samples=[]
                    for _ in range(3):
                        before=time.monotonic()
                        sample=subprocess.run(command[:-1]+["python3 -c 'import time; print(time.monotonic())'"],stdout=subprocess.PIPE,stderr=errors,check=True,timeout=8)
                        after=time.monotonic()
                        samples.append(((after-before)/2,(after+before)/2-float(sample.stdout)))
                    uncertainty,offset=min(samples)
                    if uncertainty>.10:
                        raise ValueError('相机时钟同步不确定性超过100ms')
                process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=errors)
                with self.lock:
                    self.process = process
                previous = -1
                while not self.stop_event.is_set():
                    fields = struct.unpack("<IIIQddfffff",read_exact(process.stdout,56))
                    magic,w,h,number,timestamp,captured,fx,fy,cx,cy,scale = fields
                    if magic != 0x48494d32 or (w,h)!=(640,480) or number<=previous or not 0<scale<1:
                        raise ValueError("相机帧格式或序号异常，请重新准备相机程序")
                    previous = number
                    raw_depth = read_exact(process.stdout,w*h*2)
                    raw_rgb = read_exact(process.stdout,w*h*3)
                    now = time.monotonic()
                    with self.lock:
                        self.received = now
                        self.frame_number = number
                        self.times.append(now)
                    depth = np.frombuffer(raw_depth,"<u2").reshape(h,w).astype(np.float32)*scale
                    rgb = np.frombuffer(raw_rgb,np.uint8).reshape(h,w,3)
                    with self.lock:
                        self.raw_frame=dict(depth=depth,rgb=rgb,intrinsics=(fx,fy,cx,cy),number=number,
                                            received=now,captured=captured+offset,clock_uncertainty=uncertainty)
                    images = encode_previews(depth,rgb)
                    with self.lock:
                        self.images = images
                        self.preview_number = number
                        self.phase = "streaming"
            except Exception as exc:
                error = str(exc)
            finally:
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait(timeout=2)
                errors.seek(0)
                detail = errors.read(1500).decode(errors="replace").strip()
                with self.lock:
                    self.process = None
                    if self.stop_event.is_set():
                        self.phase = "idle"
                    else:
                        self.phase = "error"
                        self.error = (error or "相机连接结束")+("；"+detail if detail else "")

"""Board/laptop D435i/Go2 planning. No generic command/velocity API."""
from collections import deque
from datetime import datetime
import importlib.util
import io
import json
import math
from pathlib import Path
import queue
import secrets
import signal
import subprocess
import sys
import threading
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'go2_runtime'


def load_runtime():
    sys.path.insert(0,str(RUNTIME))
    import run_ours
    return run_ours


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    temporary.replace(path)


def fresh_inputs(frame,state,now=None):
    now=time.monotonic() if now is None else now
    if frame is None or state is None: return False
    age=now-frame['captured']+frame['clock_uncertainty']
    return (0<=age<=.25 and 0<=now-frame['received']<=.25 and
            0<=now-state['received']+state['state_age']<=.25)


def planning_inputs_ready(frame,state,now=None):
    """Reserve compute time: wait for newer input rather than use old timestamps."""
    now=time.monotonic() if now is None else now
    return (fresh_inputs(frame,state,now) and
            now-frame['captured']+frame['clock_uncertainty']<=.10 and
            now-state['received']+state['state_age']<=.10)


def remote_status(state,now=None):
    now=time.monotonic() if now is None else now
    state=state or {}
    age=now-state.get('received',now)+state.get('remote_age',1e9)
    fresh=bool(state.get('remote_seen') and state.get('remote_valid') and 0<=age<=.5 and
               0<=now-state.get('received',now)+state.get('state_age',1e9)<=.25)
    verified=fresh and now-state.get('received',now)+state.get('remote_activity_age',1e9)<=120
    return dict(seen=bool(state.get('remote_seen')),fresh=fresh,ready=bool(fresh and state.get('remote_ready')),
                input_verified=bool(verified),source=state.get('remote_source'),
                age_ms=age*1000 if state.get('remote_seen') else None,
                keys=state.get('remote_keys',0),axes=state.get('remote_axes',[]))


class WiredRuntime:
    def __init__(self,camera,data,interface='enx00e04c584b68'):
        self.camera,self.data,self.interface=camera,Path(data),interface
        self.data.mkdir(parents=True,exist_ok=True)
        self.cal_path=self.data/'calibration.web.json'
        self.cal=json.loads((self.cal_path if self.cal_path.exists() else RUNTIME/'calibration.local.json').read_text())
        # The old xy_measured field included a user estimate; do not treat it as a tape measurement.
        self.cal.setdefault('xy_physical_confirmed',False)
        self.cal['xy_measured']=self.cal['xy_physical_confirmed']
        self.cal.setdefault('mounting_yaw_confirmed',False)
        self.cal.setdefault('mounting_yaw_deg',0.)
        self.lock=threading.RLock()
        self.action_lock=threading.RLock()
        self.module=None
        self.state_box=None
        self.states=deque(maxlen=160)
        self.bridge=None
        self.bridge_error=''
        self.bridge_log=None
        self.task=None
        self.worker=None
        self.stop_event=threading.Event()
        self.closed=threading.Event()
        self.scene=None
        self.current=None
        self.rows=deque(maxlen=300)
        self.ground_job=dict(phase='idle')
        self.ground_verified=0.
        self.approved_session=False
        self.lease_time=0.
        self.lease_token=None
        self.snapshot_queue=queue.Queue(maxsize=4)
        self.dropped_snapshots=0
        self.record_thread=threading.Thread(target=self._record_loop,daemon=True)
        self.watchdog_thread=threading.Thread(target=self._watchdog,daemon=True)
        self.record_thread.start();self.watchdog_thread.start()

    def _module(self):
        if self.module is None: self.module=load_runtime()
        return self.module

    def state(self):
        with self.lock:
            if self.state_box is None: return None
            try: return self.state_box.get()
            except RuntimeError as exc:
                self.bridge_error=str(exc)
                return None

    def _open_bridge(self,execute=False,speed_calibration=False,takeover_test=False):
        if (speed_calibration or takeover_test) and not execute: raise ValueError('测试必须显式执行')
        if speed_calibration and takeover_test: raise ValueError('测试模式冲突')
        self._close_bridge()
        mod=self._module()
        if not (RUNTIME/'build/robot_bridge').is_file():
            raise ValueError('通信桥未准备，请运行 go2_runtime/manage.sh web-wired')
        box=mod.Latest()
        self.bridge_log=(self.data/'bridge.stderr.log').open('ab')
        flag='--execute-takeover' if takeover_test else '--execute-calibration' if speed_calibration else '--execute'
        process=subprocess.Popen([str(RUNTIME/'build/robot_bridge'),self.interface]+([flag] if execute else []),
                                 stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.bridge_log)
        self.state_box,self.bridge=box,process
        self.bridge_error=''
        def read():
            mod.read_state(process,box)
        threading.Thread(target=read,daemon=True).start()

    def _close_bridge(self):
        process=self.bridge
        self.bridge=None
        if process:
            try:
                if process.poll() is None:
                    process.stdin.write(b'0 0\n');process.stdin.flush()
                process.stdin.close()
                process.wait(timeout=1)
            except (BrokenPipeError,OSError,subprocess.TimeoutExpired):
                process.terminate()
                try: process.wait(timeout=1)
                except subprocess.TimeoutExpired: process.kill();process.wait()
        if self.bridge_log:
            self.bridge_log.close();self.bridge_log=None

    def connect(self):
        with self.action_lock:
            if self.running(): raise ValueError('任务运行时不能重新连接')
            if self.bridge is None or self.bridge.poll() is not None or self.bridge_error:
                self._open_bridge(False)
            self.camera.connect()
        return self.status()

    def disconnect(self):
        self.stop('disconnected')
        with self.action_lock:
            self.camera.disconnect();self._close_bridge();self.state_box=None
        return self.status()

    def running(self):
        return self.worker is not None and self.worker.is_alive()

    def calibration_status(self):
        return dict(values=self.cal,job=self.ground_job,ground_checked_this_session=bool(self.ground_verified and time.monotonic()-self.ground_verified<600),
                    approved_this_session=self.approved_session,
                    needs=['实际测量相机前后/左右位置','核对相机安装偏航','本次静态地面拟合并确认'])

    def status(self):
        with self.lock:
            state=self.state()
            now=time.monotonic()
            age=now-state['received']+state['state_age'] if state else None
            frame=self.camera.latest_frame()
            return dict(enabled=True,connected=bool(state and age<=.25),state=state,remote=remote_status(state,now),state_age_ms=age*1000 if age is not None else None,
                        inputs_fresh=fresh_inputs(frame,state),bridge_error=self.bridge_error,
                        task=self.task,current=self.current,rows=list(self.rows),scene=self.scene,
                        calibration=self.calibration_status(),motion_ready=bool(fresh_inputs(frame,state) and self.approved_session),
                        bridge_writes=(self.task or {}).get('bridge_writes',0),
                        physical_commands_sent=None if (self.task or {}).get('kind')=='execute' else 0)

    def save_measurements(self,payload):
        if self.running() or self.ground_job['phase']=='running': raise ValueError('任务或地面拟合运行时不能改标定')
        expected={'forward_m','left_m','yaw_deg','physically_measured','yaw_confirmed'}
        if not isinstance(payload,dict) or set(payload)!=expected: raise ValueError('标定字段无效')
        for key in ('forward_m','left_m','yaw_deg'):
            if type(payload[key]) not in (float,int) or not math.isfinite(payload[key]): raise ValueError('标定参数必须是有限数字')
        if not .1<=payload['forward_m']<=.6 or abs(payload['left_m'])>.25 or abs(payload['yaw_deg'])>30:
            raise ValueError('标定位置或偏航超出本安装方式支持范围')
        if type(payload['physically_measured']) is not bool or type(payload['yaw_confirmed']) is not bool:
            raise ValueError('测量确认应为布尔值')
        self.cal=dict(self.cal,calibrated=False,xy_measured=payload['physically_measured'],
                      xy_physical_confirmed=payload['physically_measured'],mounting_yaw_confirmed=payload['yaw_confirmed'],
                      mounting_yaw_deg=float(payload['yaw_deg']))
        self.cal['camera_translation_body_m'][:2]=[float(payload['forward_m']),float(payload['left_m'])]
        self.approved_session=False;self.ground_verified=0
        atomic_json(self.cal_path,self.cal)
        return self.calibration_status()

    def _stationary(self):
        now=time.monotonic()
        samples=[s for s in self.states if now-s['received']<2]
        if len(samples)<15: raise ValueError('请先连接并静止站立至少2秒')
        if any(not .20<=s['z']<=.60 or math.hypot(s['vx'],s['vy'])>.025 or abs(s['yaw_speed'])>.04 or
               abs(s['roll'])>.12 or abs(s['pitch'])>.12 for s in samples):
            raise ValueError('需要在水平地面正常站立并保持静止')
        if np.ptp([s['z'] for s in samples])>.02: raise ValueError('本体高度变化过大')
        return samples[-1]

    def fit_ground(self):
        if self.running() or self.ground_job['phase']=='running': raise ValueError('已有任务运行')
        self._stationary()
        if not fresh_inputs(self.camera.latest_frame(),self.state()): raise ValueError('相机/状态尚未新鲜')
        self.ground_job=dict(phase='running',message='采集3个静态地面拟合样本')
        self.approved_session=False
        threading.Thread(target=self._fit_worker,daemon=True).start()
        return self.calibration_status()

    def _fit_worker(self):
        try:
            from estimate_floor import fit_ground
            mod=self._module();samples=[];last=-1
            deadline=time.monotonic()+8
            while len(samples)<3:
                state=self._stationary();frame=self.camera.latest_frame()
                if time.monotonic()>deadline: raise ValueError('采样超时')
                if not fresh_inputs(frame,state): raise ValueError('地面采样期间数据过期')
                if frame['number']==last: time.sleep(.02);continue
                n,height,fraction,rms=fit_ground(frame['depth'],frame['intrinsics'])
                if fraction<.70 or rms>.015: raise ValueError('地面占比/拟合残差不满足标定条件')
                samples.append((n,height,fraction,rms));last=frame['number']
            if np.ptp([s[1] for s in samples])>.01: raise ValueError('多帧地面高度不一致')
            if any(np.dot(s[0],samples[0][0])<math.cos(math.radians(1.5)) for s in samples):
                raise ValueError('多帧地面法向不一致')
            n=np.mean([s[0] for s in samples],axis=0);n/=np.linalg.norm(n)
            height=float(np.mean([s[1] for s in samples]));level=mod.rotation_rp(state['roll'],state['pitch'])
            forward=np.array([0.,0.,1.]);forward-=(forward@n)*n;forward/=np.linalg.norm(forward)
            yaw=math.radians(self.cal['mounting_yaw_deg']);c,s=math.cos(yaw),math.sin(yaw)
            rotation=level.T@np.array([[c,-s,0],[s,c,0],[0,0,1]])@np.stack((forward,np.cross(n,forward),n))
            x,y=self.cal['camera_translation_body_m'][:2]
            z=(height-state['z']-level[2,0]*x-level[2,1]*y)/level[2,2]
            self.cal=dict(self.cal,calibrated=False,camera_translation_body_m=[x,y,float(z)],
                          camera_rotation_optical_to_body=rotation.tolist(),standing_base_height_m=state['z'],
                          ground_camera_height_m=height,ground_inlier_fraction=min(s[2] for s in samples),
                          ground_rms_m=max(s[3] for s in samples),extra_footprint_margin_m=.04,
                          calibration_stage='web_ground_fit_measured' if self.cal['xy_physical_confirmed'] else 'preliminary_ground_fit_plus_user_tape_range',
                          source_snapshot=str(self.data/'ground-verification.npz'))
            mod.validate_calibration(self.cal)
            np.savez_compressed(self.data/'ground-verification.npz',depth=frame['depth'],intrinsics=frame['intrinsics'],state_json=json.dumps(state))
            atomic_json(self.cal_path,self.cal)
            self.ground_verified=time.monotonic()
            self.ground_job=dict(phase='complete',message='地面高度/倾角已复核；平移X/Y和偏航仍须人工核验')
        except Exception as exc:
            self.ground_verified=0
            self.ground_job=dict(phase='error',message=str(exc))

    def approve_calibration(self,payload):
        if payload!={'confirmed':True}: raise ValueError('请确认测量与安装方向')
        if self.running(): raise ValueError('任务运行时不能批准标定')
        if not self.ground_verified or time.monotonic()-self.ground_verified>600: raise ValueError('请先完成本次地面复核')
        if not self.cal['xy_physical_confirmed'] or not self.cal['mounting_yaw_confirmed']:
            raise ValueError('34cm/居中估计不能当作完整物理测量，请先实际测量并保存')
        self.cal['calibrated']=True;self.approved_session=True
        atomic_json(self.cal_path,self.cal)
        return self.calibration_status()

    def validate_start(self,payload):
        expected={'kind','parameters','operator_confirmed','preliminary_ack','record_rgbd','speed_calibration','takeover_test','remote_verified'}
        if not isinstance(payload,dict) or set(payload)-expected: raise ValueError('实机任务字段无效')
        kind=payload.get('kind')
        if kind not in ('shadow','execute'): raise ValueError('只支持真实影子规划或短距离实际测试')
        from server import parameters
        for flag in ('operator_confirmed','preliminary_ack','record_rgbd','speed_calibration','takeover_test','remote_verified'):
            if flag in payload and type(payload[flag]) is not bool: raise ValueError('确认字段应为布尔值')
        speed_calibration=payload.get('speed_calibration',False)
        takeover_test=payload.get('takeover_test',False)
        if takeover_test and (kind!='execute' or speed_calibration): raise ValueError('接管测试模式冲突')
        if speed_calibration and kind!='execute': raise ValueError('速度标定仅用于显式执行')
        params=parameters(payload.get('parameters',{}), max_vx_limit=.30 if speed_calibration or takeover_test else .20)
        if kind=='execute':
            if payload.get('operator_confirmed') is not True: raise ValueError('实际行走需要现场手动确认')
            if takeover_test:
                if not .5<=params['forward']<=2 or params['lateral']!=0 or params['max_vx']>.30 or params['duration']>8:
                    raise ValueError('手柄接管测试：前方0.5–2m、侧向0、最高0.30m/s、最多8秒')
                if payload.get('remote_verified') is not True: raise ValueError('请先只读核对手柄输入并确认')
                if not self.ground_verified or time.monotonic()-self.ground_verified>600: raise ValueError('请先重新地面复核')
                if not remote_status(self.state())['ready']:
                    raise ValueError('手柄数据过期或未归中，请松开按键并归中至少0.5秒')
                if not remote_status(self.state())['input_verified']:
                    raise ValueError('两分钟内未检测到真实手柄操作，请先只读核对输入并归中')
            elif speed_calibration:
                if params['forward']>.5 or params['lateral']!=0 or params['max_vx']>.30 or params['duration']>3:
                    raise ValueError('速度标定仅限前方0.3–0.5m、侧向0、最高0.30m/s、最多3秒')
                if not self.ground_verified or time.monotonic()-self.ground_verified>600:
                    raise ValueError('速度标定需要本次地面复核')
            elif params['forward']>.5 or params['lateral']!=0 or params['max_vx']>.10 or params['duration']>5:
                raise ValueError('首轮执行限制：前方0.3–0.5m、侧向0、速度≤0.10m/s、时长≤5s')
            if not self.approved_session:
                if payload.get('preliminary_ack') is not True or not self.ground_verified or time.monotonic()-self.ground_verified>600:
                    raise ValueError('完整标定未通过；初步标定试验需要本次地面复核和单独风险确认')
                if self.cal.get('extra_footprint_margin_m',0)<.04: raise ValueError('初步标定缺少额外足迹裕量')
            state=self.state()
            if state is None or not .20<=state['z']<=.60 or abs(state['roll'])>.45 or abs(state['pitch'])>.45:
                raise ValueError('狗尚未正常站立')
        if not fresh_inputs(self.camera.latest_frame(),self.state()): raise ValueError('相机或本体状态过期/未连接')
        return kind,params

    def start(self,payload):
        with self.action_lock:
            if self.running() or self.ground_job['phase']=='running': raise ValueError('已有实机任务或标定在运行')
            kind,params=self.validate_start(payload)
            mod=self._module()
            state=self.state()
            goal_world=[state['x']+math.cos(state['yaw'])*params['forward']-math.sin(state['yaw'])*params['lateral'],
                        state['y']+math.sin(state['yaw'])*params['forward']+math.cos(state['yaw'])*params['lateral']]
            task_id='web_'+kind+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_')+secrets.token_hex(3)
            out=RUNTIME/'runs'/task_id;out.mkdir(parents=True)
            self.lease_token=secrets.token_urlsafe(24);self.lease_time=time.monotonic()
            self.stop_event=threading.Event();self.rows.clear();self.scene=None;self.current=None
            self.task=dict(id=task_id,kind=kind,source='real',parameters=params,outcome='running',outcome_label='真实影子规划中' if kind=='shadow' else '实际执行中',
                           duration=0,sample_count=0,bridge_writes=0,record_rgbd=payload.get('record_rgbd',False),out=str(out),speed_calibration=payload.get('speed_calibration',False),takeover_test=payload.get('takeover_test',False),
                           goal_world=goal_world,goal_reference='start_body_frozen_in_sport_odometry',snapshot_drops=0,input_waits=0)
            atomic_json(out/'metadata.json',dict(mode=kind,source='real',calibration=self.cal,goal_relative=[params['forward'],params['lateral']],
                        goal_world=goal_world,parameters=params,method='dual_cost_fmm_standard_footprint_mppi',speed_calibration=payload.get('speed_calibration',False),takeover_test=payload.get('takeover_test',False),
                        preliminary_trial=kind=='execute' and not self.approved_session,pose_source='Go2 SportModeState',created=datetime.now().astimezone().isoformat(),
                        rgbd_recording_hz=2 if self.task['record_rgbd'] else 0))
            self.worker=threading.Thread(target=self._plan,args=(self.task,dict(self.cal)),daemon=True)
            self.worker.start()
            return dict(task=self.task,lease=self.lease_token if kind=='execute' else None)

    def heartbeat(self,token):
        if not secrets.compare_digest(str(token),str(self.lease_token)): raise ValueError('执行会话已失效')
        self.lease_time=time.monotonic()
        return dict(ok=True)

    def _watchdog(self):
        previous=None
        while not self.closed.wait(.05):
            state=self.state()
            if state and state['received']!=previous:
                self.states.append(state);previous=state['received']
            if self.running() and not self.stop_event.is_set() and self.task['kind']=='execute' and time.monotonic()-self.lease_time>.8:
                self.task['stop_reason']='网页心跳中断';self.stop_event.set()
                self._close_bridge()

    def _controllers(self,params,cal):
        mapper,mppi_module,cfg=self._module().algorithms()
        bev=mapper.BEVMapper(bev_res=cfg.BEV_RES,bev_x=cfg.BEV_X,bev_y=cfg.BEV_Y,
            height_range_thresh=cfg.BEV_HEIGHT_RANGE_THRESH,min_points=cfg.BEV_MIN_POINTS,v_start_ratio=cfg.BEV_V_START_RATIO,
            max_depth=cfg.BEV_MAX_DEPTH,min_depth=cfg.BEV_MIN_DEPTH,enable_step_judge=cfg.BEV_ENABLE_STEP_JUDGE,enable_memory_fusion=False)
        fmm=mapper.FMMGradientController(bev_res=cfg.BEV_RES,bev_x=cfg.BEV_X,bev_y=cfg.BEV_Y,
            use_soft_cost=True,hard_radius_m=cfg.FMM_HARD_RADIUS_M,soft_clearance_m=cfg.FMM_SOFT_CLEARANCE_M,
            soft_cost_weight=cfg.FMM_SOFT_COST_WEIGHT,soft_cost_power=cfg.FMM_SOFT_COST_POWER)
        mppi=mppi_module.FootprintMPPIController(bev_res=cfg.BEV_RES,bev_x=cfg.BEV_X,bev_y=cfg.BEV_Y,
            horizon=cfg.MPPI_HORIZON,num_samples=cfg.MPPI_NUM_SAMPLES,rollout_dt=cfg.MPPI_ROLLOUT_DT,
            max_vx=params['max_vx'],min_vx=params['max_vx'] if self.task['kind']=='execute' else 0,max_wz=0 if self.task.get('speed_calibration') or self.task.get('takeover_test') else .35,
            footprint_length_m=cfg.MPPI_FOOTPRINT_LENGTH_M,footprint_width_m=cfg.MPPI_FOOTPRINT_WIDTH_M,
            safety_margin_m=cfg.MPPI_FOOTPRINT_MARGIN_M+float(cal.get('extra_footprint_margin_m',0)),
            temperature=cfg.MPPI_TEMPERATURE,noise_vx=cfg.MPPI_NOISE_VX,noise_wz=cfg.MPPI_NOISE_WZ,variant='standard')
        return bev,fmm,mppi

    def _plan(self,task,cal):
        out=Path(task['out']);execute=task['kind']=='execute';started=time.monotonic();last=-1;last_snapshot=-1;arrived=0;plan_times=[];timings=[];reasons={}
        try:
            # Initialize all numerical code before creating a motion-capable bridge.
            bev,fmm,mppi=self._controllers(task['parameters'],cal)
            if self.stop_event.is_set(): return
            if execute:
                if time.monotonic()-self.lease_time>.8: raise ValueError('网页执行会话已过期')
                self._open_bridge(True,speed_calibration=task.get('speed_calibration',False),takeover_test=task.get('takeover_test',False))
                deadline=time.monotonic()+2
                while True:
                    current_state=self.state()
                    if current_state is not None and (not task.get('takeover_test') or current_state.get('remote_ready')): break
                    if self.stop_event.is_set() or time.monotonic()>deadline: raise ValueError('执行桥状态等待超时')
                    time.sleep(.02)
            started=time.monotonic()
            with (out/'telemetry.jsonl').open('w',buffering=1) as log:
                while not self.stop_event.is_set() and time.monotonic()-started<task['parameters']['duration']:
                    frame,state=self.camera.latest_frame(),self.state()
                    if not fresh_inputs(frame,state): raise ValueError('相机或状态过期，终止任务')
                    if frame['number']==last: time.sleep(.005);continue
                    if not planning_inputs_ready(frame,state):
                        task['input_waits']+=1;time.sleep(.002);continue
                    last=frame['number'];goal=self._module().local_goal(task['goal_world'],state)
                    timing={}
                    command,reason,occ,path,valid,ms=self._module().select_command(bev,fmm,mppi,frame['depth'],frame['intrinsics'],state,frame,cal,goal,execute,diagnostics=timing)
                    if np.linalg.norm(goal)<=.20:
                        command,reason=(0.,0.),'goal_confirmation'
                        arrived=arrived+1 if math.hypot(state['vx'],state['vy'])<.05 and abs(state['yaw_speed'])<.1 else 0
                    else: arrived=0
                    if self.stop_event.is_set(): break
                    if execute:
                        if not fresh_inputs(frame,self.state()) or time.monotonic()-self.lease_time>.8: raise ValueError('下发前检查失败，已终止')
                        self.bridge.stdin.write(('%.6f %.6f\n'%command).encode());self.bridge.stdin.flush()
                        task['bridge_writes']+=1
                    row=dict(elapsed=time.monotonic()-started,frame=last,state=state,goal=goal.tolist(),proposed_vx=float(command[0]),proposed_wz=float(command[1]),
                        sent=execute,reason=reason,obstacle_cells=int(occ.sum()),valid_fraction=valid,plan_ms=ms,
                        capture_age_upper_ms=(time.monotonic()-frame['captured']+frame['clock_uncertainty'])*1000,planning_timing=timing,source='real')
                    timings.append(timing);reasons[reason]=reasons.get(reason,0)+1
                    log.write(json.dumps(row)+'\n');plan_times.append(ms)
                    self.rows.append(row);self.current=row;task['duration']=row['elapsed'];task['sample_count']+=1
                    self.scene=dict(occupied=np.argwhere(occ>0).tolist(),shape=list(occ.shape),path=path.tolist(),
                        mppi=mppi.last_plan.tolist() if mppi.last_plan is not None else [],soft_cost=fmm.soft_cost_map[::2,::2].tolist(),resolution=.05,extent=[6,6],goal=goal.tolist())
                    if row['elapsed']-last_snapshot>=.5:
                        job=(out,dict(depth=frame['depth'],rgb=frame['rgb'],intrinsics=frame['intrinsics'],occ=occ.copy(),path=path.copy(),
                                     soft_cost=fmm.soft_cost_map.copy(),state_json=json.dumps(state),mppi_plan=np.array(mppi.last_plan),frame=last),task['record_rgbd'])
                        try: self.snapshot_queue.put_nowait(job)
                        except queue.Full: task['snapshot_drops']+=1
                        last_snapshot=row['elapsed']
                    if arrived>=10:
                        task['outcome']='odometry_arrived';task['outcome_label']='里程计到达，需现场核验';break
        except Exception as exc:
            task['error']=str(exc);task['outcome']='failed';task['outcome_label']='任务中止：'+str(exc)
            self.state() # Capture a latched native event, including pipe-close races.
            if 'bridge_event:' in self.bridge_error:
                event=self.bridge_error.split('bridge_event:',1)[1]
                labels={'remote_takeover':'手柄触发接管，程序已退出','remote_signal_lost':'手柄数据通道超时，已停止',
                        'remote_invalid':'手柄数据无效，已停止','remote_not_ready':'手柄未就绪，未启动运动',
                        'time_limit':'达到桥端时限，已停止','distance_limit':'达到桥端2米上限，已停止'}
                task['outcome']='taken_over' if event=='remote_takeover' else 'stopped'
                task['outcome_label']=labels.get(event,'通信桥停止：'+event);task['stop_reason']=task['outcome_label']
            if self.stop_event.is_set():
                task['outcome']='stopped';task['outcome_label']=task.get('stop_reason','已停止')
        finally:
            if execute:
                self._close_bridge()
                if not self.closed.is_set():
                    try: self._open_bridge(False)
                    except Exception as exc: self.bridge_error=str(exc)
            if task['outcome']=='running':
                task['outcome']='stopped' if self.stop_event.is_set() else 'timeout'
                task['outcome_label']=task.get('stop_reason','手动停止') if self.stop_event.is_set() else '达到时长上限，已停止'
                if not execute and not self.stop_event.is_set():
                    task['outcome']='shadow_complete';task['outcome_label']='影子规划完成，未发送速度'
            atomic_json(out/'summary.json',dict(status=task['outcome'],frames=task['sample_count'],error=task.get('error'),
                sent_velocity_commands=bool(task['bridge_writes']),bridge_writes=task['bridge_writes'],
                plan_ms_mean=float(np.mean(plan_times)) if plan_times else None,plan_ms_p95=float(np.percentile(plan_times,95)) if plan_times else None,
                snapshot_drops=task['snapshot_drops'],input_waits=task.get('input_waits',0),outcome_label=task['outcome_label'],reason_counts=reasons,
                timing_mean_ms={k:float(np.mean([t[k] for t in timings])) for k in timings[0] if k.endswith('_ms')} if timings else {}))

    def _record_loop(self):
        while True:
            job=self.snapshot_queue.get()
            if job is None: self.snapshot_queue.task_done();return
            out,data,raw=job
            try:
                buffer=io.BytesIO();np.savez_compressed(buffer,**data);blob=buffer.getvalue()
                temporary=out/'latest.tmp';temporary.write_bytes(blob);temporary.replace(out/'latest.npz')
                if raw:
                    (out/'frames').mkdir(exist_ok=True)
                    (out/'frames'/('frame_%08d.npz'%data['frame'])).write_bytes(blob)
            except Exception as exc:
                self.bridge_error='快照录制失败：'+str(exc)
            finally: self.snapshot_queue.task_done()

    def stop(self,reason='手动停止'):
        if self.task and self.running(): self.task['stop_reason']=reason
        self.stop_event.set()
        if self.worker and self.worker is not threading.current_thread(): self.worker.join(timeout=2)
        if self.running(): self._close_bridge()
        return dict(ok=True,message='已请求停止；不会自动站立或重启运动',bridge_writes=(self.task or {}).get('bridge_writes',0))

    def close(self):
        self.closed.set();self.stop('服务关闭');self.camera.disconnect();self._close_bridge()
        self.snapshot_queue.put(None,timeout=2)
        self.record_thread.join(timeout=2);self.watchdog_thread.join(timeout=1)

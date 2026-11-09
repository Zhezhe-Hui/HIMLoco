import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from wired import WiredRuntime,fresh_inputs,planning_inputs_ready,remote_status


class WiredTest(unittest.TestCase):
    def test_planning_waits_for_newer_input_without_weakening_expiry(self):
        now=time.monotonic()
        frame=dict(captured=now-.15,received=now-.14,clock_uncertainty=0)
        state=dict(received=now-.02,state_age=.001)
        self.assertTrue(fresh_inputs(frame,state,now))
        self.assertFalse(planning_inputs_ready(frame,state,now))
        frame.update(captured=now-.05,received=now-.04)
        self.assertTrue(planning_inputs_ready(frame,state,now))
        state['state_age']=.3
        self.assertFalse(fresh_inputs(frame,state,now))
        self.assertFalse(planning_inputs_ready(frame,state,now))

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.camera=Mock()
        self.runtime=WiredRuntime(self.camera,Path(self.temp.name))
        self.state=dict(x=1.,y=2.,yaw=math.pi/2,z=.31,roll=0.,pitch=0.,vx=0.,vy=0.,yaw_speed=0.,state_age=0.,received=time.monotonic())
        self.frame=dict(captured=time.monotonic(),received=time.monotonic(),clock_uncertainty=.001,number=1)
        self.runtime.state=Mock(return_value=self.state)
        self.camera.latest_frame.return_value=self.frame

    def tearDown(self):
        self.runtime.close();self.temp.cleanup()

    def payload(self,kind='execute',**kwargs):
        return dict(kind=kind,parameters=dict(forward=.5,lateral=0,max_vx=.1,duration=5),operator_confirmed=True,preliminary_ack=True,record_rgbd=False,**kwargs)

    def test_age_limits_and_clock_uncertainty(self):
        self.assertTrue(fresh_inputs(self.frame,self.state))
        self.assertFalse(fresh_inputs(dict(self.frame,captured=time.monotonic()-.3),self.state))
        self.assertFalse(fresh_inputs(dict(self.frame,clock_uncertainty=float('inf')),self.state))
        self.assertFalse(fresh_inputs(self.frame,dict(self.state,state_age=.3)))

    def test_motion_rejected_without_new_ground_fit_before_bridge_creation(self):
        with patch('wired.subprocess.Popen') as popen:
            with self.assertRaises(ValueError): self.runtime.start(self.payload())
            popen.assert_not_called()

    def test_preliminary_execute_restrictions_cannot_be_bypassed(self):
        self.runtime.ground_verified=time.monotonic()
        self.runtime.cal['extra_footprint_margin_m']=.04
        for field,value in [('forward',1),('lateral',.1),('max_vx',.2),('duration',10)]:
            p=self.payload();p['parameters'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): self.runtime.validate_start(p)
        p=self.payload();p['operator_confirmed']=False
        with self.assertRaises(ValueError): self.runtime.validate_start(p)
        self.assertEqual(self.runtime.validate_start(self.payload())[0],'execute')

    def test_estimated_xy_cannot_approve_full_calibration(self):
        self.runtime.ground_verified=time.monotonic()
        with self.assertRaises(ValueError): self.runtime.approve_calibration({'confirmed':True})
        self.runtime.save_measurements(dict(forward_m=.34,left_m=0,yaw_deg=0,physically_measured=False,yaw_confirmed=False))
        self.assertFalse(self.runtime.cal['calibrated'])
        self.assertFalse(self.runtime.cal['xy_measured'])
        self.assertEqual(self.runtime.ground_verified,0)

    def test_speed_calibration_requires_explicit_bounded_mode(self):
        self.runtime.ground_verified=time.monotonic()
        self.runtime.cal['extra_footprint_margin_m']=.04
        p=self.payload(speed_calibration=True)
        p['parameters'].update(max_vx=.30,duration=3)
        self.assertEqual(self.runtime.validate_start(p)[0],'execute')
        for field,value in [('forward',.6),('lateral',.01),('max_vx',.301),('duration',3.01)]:
            q=json.loads(json.dumps(p));q['parameters'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.runtime.validate_start(q)
        for field,value in [('speed_calibration',False),('operator_confirmed',False),('preliminary_ack',False),('kind','shadow')]:
            q=json.loads(json.dumps(p));q[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.runtime.validate_start(q)
        self.runtime.ground_verified=0
        with self.assertRaises(ValueError):self.runtime.validate_start(p)

    def test_takeover_profile_requires_fresh_neutral_verified_remote(self):
        self.runtime.ground_verified=time.monotonic()
        self.runtime.cal['extra_footprint_margin_m']=.04
        self.state.update(remote_seen=True,remote_valid=True,remote_ready=True,remote_age=.01,remote_activity_age=1.)
        p=self.payload(takeover_test=True,remote_verified=True)
        p['parameters'].update(forward=2,max_vx=.30,duration=8)
        self.assertEqual(self.runtime.validate_start(p)[0],'execute')
        for key,value in [('remote_verified',False),('takeover_test',False),('speed_calibration',True),('kind','shadow')]:
            q=json.loads(json.dumps(p));q[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.runtime.validate_start(q)
        for key,value in [('forward',2.1),('duration',8.1),('lateral',.1),('max_vx',.31)]:
            q=json.loads(json.dumps(p));q['parameters'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.runtime.validate_start(q)
        self.state['remote_age']=.6
        self.assertFalse(remote_status(self.state)['ready'])
        with self.assertRaises(ValueError):self.runtime.validate_start(p)
        self.state.update(remote_age=0,remote_ready=False)
        with self.assertRaises(ValueError):self.runtime.validate_start(p)

    def test_default_bridge_has_no_execute_flag(self):
        self.runtime.state=Mock(return_value=None)
        process=Mock();process.stdout=[];process.poll.return_value=0
        with patch('wired.subprocess.Popen',return_value=process) as popen:
            self.runtime._open_bridge(False)
            self.assertNotIn('--execute',popen.call_args.args[0])
            self.runtime._close_bridge()

    def test_manual_connect_rebuilds_failed_reader_without_execution(self):
        self.runtime.bridge = Mock()
        self.runtime.bridge.poll.return_value = None
        self.runtime.bridge_error = 'bad state JSON'
        self.runtime.camera.connect = Mock()
        with patch.object(self.runtime,'_open_bridge') as reopen:
            self.runtime.connect()
        reopen.assert_called_once_with(False)
        self.runtime.bridge = None

    def test_native_takeover_finishes_task_without_restarting_motion(self):
        out=Path(self.temp.name)/'takeover-event';out.mkdir()
        task=dict(kind='execute',out=str(out),parameters=dict(forward=2,lateral=0,max_vx=.3,duration=8),
                  outcome='running',outcome_label='',bridge_writes=0,sample_count=0,snapshot_drops=0)
        self.runtime.task=task
        self.runtime.bridge_error='bridge_event:remote_takeover'
        with patch.object(self.runtime,'_controllers',side_effect=RuntimeError('pipe closed')),patch.object(self.runtime,'_open_bridge') as reopen:
            self.runtime._plan(task,self.runtime.cal)
        self.assertEqual(task['outcome'],'taken_over')
        self.assertIn('手柄',task['outcome_label'])
        reopen.assert_called_once_with(False)

    def test_web_lease_timeout_closes_owned_bridge(self):
        self.runtime.task=dict(kind='execute')
        self.runtime.running=Mock(return_value=True)
        self.runtime._close_bridge=Mock()
        self.runtime.lease_time=time.monotonic()-2
        deadline=time.monotonic()+.3
        while not self.runtime.stop_event.is_set() and time.monotonic()<deadline: time.sleep(.01)
        self.assertTrue(self.runtime.stop_event.is_set())
        self.runtime._close_bridge.assert_called_once()
        self.assertEqual(self.runtime.task['stop_reason'],'网页心跳中断')
        self.runtime.running=Mock(return_value=False)

    def test_goal_fixed_at_start_in_odometry_frame(self):
        class Worker:
            def __init__(self,*args,**kwargs): pass
            def start(self): pass
            def is_alive(self): return False
        with patch('wired.threading.Thread',Worker),patch('wired.RUNTIME',Path(self.temp.name)),patch.object(self.runtime,'_module'):
            task=self.runtime.start(self.payload('shadow'))['task']
        np.testing.assert_allclose(task['goal_world'],[1,2.5],atol=1e-8)
        self.state['x']=1.1
        np.testing.assert_allclose(task['goal_world'],[1,2.5],atol=1e-8)
        self.runtime.worker=None

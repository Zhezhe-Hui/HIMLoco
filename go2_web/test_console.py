import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

import numpy as np

from server import ConsoleServer, OfflineConsole, handler_for, parameters


class OfflineConsoleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root/"runs"
        self.runs.mkdir()
        folder = self.runs/"ours_first_walk_20261030"
        folder.mkdir()
        row = dict(elapsed=1, frame=5, proposed_vx=.06, sent=True,
                   state=dict(x=0, y=0, z=.3, vx=0, received=100), reason="ok")
        self.original = json.dumps(row)
        (folder/"telemetry.jsonl").write_text(self.original+"\n")
        (folder/"summary.json").write_text(json.dumps(dict(frames=1, sent_velocity_commands=True)))
        np.savez(folder/"latest.npz", rgb=np.zeros((4,4,3),dtype=np.uint8),
                 depth=np.ones((4,4)), occ=np.zeros((120,120),dtype=np.uint8),
                 path=np.array([[0,0],[.5,0]]), soft_cost=np.zeros((120,120)),
                 state_json=json.dumps(row["state"]))
        self.console = OfflineConsole(self.runs, self.root/"data")

    def tearDown(self):
        self.console.stop()
        self.temp.cleanup()

    def test_reject_invalid_and_nonfinite_parameters(self):
        for raw in ({"forward":float("nan")},{"max_vx":True},{"duration":61},
                    {"forward":0},{"max_vx":.3},{"execute":True},[1,2]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parameters(raw)

    def test_history_retains_no_steps_and_preserves_original(self):
        history = self.console.get_experiment("ours_first_walk_20261030")
        self.assertEqual(history["outcome"], "no_steps")
        self.assertEqual(history["snapshot_frame"],5)
        self.assertEqual((self.runs/"ours_first_walk_20261030/telemetry.jsonl").read_text(),self.original+"\n")
        self.assertEqual(self.console.status()["physical_commands_sent"],0)

    def test_simulation_cannot_start_physical_task(self):
        with self.assertRaises(ValueError):
            self.console.start(dict(kind="execute", parameters={}))

    def test_single_active_task_stop_and_reload(self):
        task = self.console.start(dict(kind="simulation",parameters={"duration":5}))
        with self.assertRaises(RuntimeError):
            self.console.start(dict(kind="shadow",parameters={}))
        self.console.stop()
        stopped=self.console.get_experiment(task["id"])
        self.assertEqual(stopped["outcome"],"stopped")
        self.assertEqual(stopped["current"]["proposed_vx"],0)
        self.assertEqual(stopped["current"]["state"]["vx"],0)
        restarted = OfflineConsole(self.runs,self.root/"data")
        self.assertIsNone(restarted.status()["active"])
        self.assertEqual(restarted.get_experiment(task["id"])["source"],"simulation")

    def test_recording_off_does_not_fabricate_samples(self):
        self.console.recording=False
        task=self.console.start(dict(kind="shadow",parameters={}))
        self.console.stop()
        saved=self.console.get_experiment(task["id"])
        self.assertEqual(saved["rows"],[])
        self.assertEqual(saved["summary"]["recorded_samples"],0)

    def test_export_has_source_and_telemetry(self):
        archive=self.console.export("ours_first_walk_20261030")
        with zipfile.ZipFile(io.BytesIO(archive)) as result:
            meta=json.loads(result.read("metadata.json"))
            self.assertEqual(meta["source"],"historical")
            self.assertIn("rgb_snapshot.png",result.namelist())
            self.assertEqual(json.loads(result.read("telemetry.jsonl"))["sent"],True)

    def test_simulation_timeout_is_not_navigation_success(self):
        task=self.console.start(dict(kind="simulation",parameters={"forward":5,"duration":1}))
        deadline=time.monotonic()+2
        while self.console.get_experiment(task["id"])["outcome"]=="running" and time.monotonic()<deadline:
            time.sleep(.02)
        saved=self.console.get_experiment(task["id"])
        self.assertEqual(saved["outcome"],"simulated_timeout")
        self.assertTrue(all(r["sent"] is False for r in saved["rows"]))
        self.assertTrue(all("plan_ms" not in r for r in saved["rows"]))

    def test_http_rejects_cross_origin_robot_routes_and_traversal(self):
        server=ConsoleServer(("127.0.0.1",0),handler_for(self.console))
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        base="http://127.0.0.1:"+str(server.server_address[1])
        try:
            with urlopen(base+"/api/state",timeout=2) as result:
                self.assertFalse(json.load(result)["motion_enabled"])
            self.console.camera.images = dict(rgb=b'RGB42', depth=b'DEPTH42')
            self.console.camera.preview_number = 42
            with urlopen(base+'/api/camera/preview',timeout=2) as result:
                pair=json.load(result)
                self.assertEqual(pair['frame'],42)
                self.assertEqual(result.headers['ETag'],'"42"')
                self.assertEqual(set(pair['images']),{'rgb','depth'})
            with self.assertRaises(HTTPError) as unchanged:
                urlopen(Request(base+'/api/camera/preview',headers={'If-None-Match':'"42"'}),timeout=2)
            self.assertEqual(unchanged.exception.code,304)
            for path,headers,expected in [
                ("/api/tasks/start",{"Origin":"https://example.com","Content-Type":"application/json"},403),
                ("/api/robot/move",{"Content-Type":"application/json"},403)]:
                req=Request(base+path,data=b"{}",headers=headers,method="POST")
                with self.assertRaises(HTTPError) as error:
                    urlopen(req,timeout=2)
                self.assertEqual(error.exception.code,expected)
            for payload in ({}, {"host":"other-machine"}, {"command":"anything"}):
                req=Request(base+"/api/camera/connect",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
                with self.assertRaises(HTTPError) as error:
                    urlopen(req,timeout=2)
                self.assertEqual(error.exception.code,400)
            with self.assertRaises(HTTPError) as error:
                urlopen(base+"/api/experiments/../../etc/passwd",timeout=2)
            self.assertEqual(error.exception.code,404)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()

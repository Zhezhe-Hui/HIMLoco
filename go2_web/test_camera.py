import io
import struct
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from camera import ReadOnlyCamera, encode_previews, read_exact


class CameraTest(unittest.TestCase):
    def test_disabled_does_not_launch_ssh(self):
        camera = ReadOnlyCamera()
        with patch('camera.subprocess.Popen') as launch:
            with self.assertRaises(ValueError):
                camera.connect()
            launch.assert_not_called()

    def test_truncated_frame_rejected(self):
        with self.assertRaises(EOFError):
            read_exact(io.BytesIO(b'ab'), 3)

    def test_decode_real_protocol_and_fixed_readonly_command(self):
        header = struct.pack('<IIIQddfffff',0x48494d32,640,480,7,0,0,400,400,320,240,.001)
        payload = header + np.full((480,640),1000,dtype='<u2').tobytes() + bytes(480*640*3)
        class Process:
            stdout = io.BytesIO(payload)
            def poll(self): return 0
        camera = ReadOnlyCamera(enabled=True)
        with patch('camera.subprocess.Popen',return_value=Process()) as launch:
            camera._receive()
            command = launch.call_args.args[0]
            self.assertEqual(command[-1],'exec /home/unitree/.local/lib/himloco/camera_stream')
            self.assertIn('BatchMode=yes',command)
        status = camera.status()
        self.assertEqual(status['frame'],7)
        self.assertTrue(status['has_frame'])
        self.assertEqual(status['phase'],'error')  # EOF must not appear live.
        self.assertEqual(Image.open(io.BytesIO(camera.image('rgb'))).size,(640,480))
        self.assertTrue(status['read_only'])
        self.assertFalse(status['recording'])

    def test_stale_and_disconnect_clear_live_state(self):
        camera = ReadOnlyCamera(enabled=True)
        camera.images = {'rgb':b'last frame'}
        camera.phase = 'streaming'
        camera.received = time.monotonic()-2
        self.assertEqual(camera.status()['phase'],'stale')
        self.assertFalse(camera.status()['connected'])
        camera.disconnect()
        self.assertFalse(camera.status()['has_frame'])
        self.assertEqual(camera.status()['phase'],'idle')

    def test_preview_handles_invalid_depth(self):
        images = encode_previews(np.array([[0,np.nan],[np.inf,1.]]),np.zeros((2,2,3),dtype=np.uint8))
        self.assertEqual(set(images),{'rgb','depth'})
        self.assertEqual(Image.open(io.BytesIO(images['depth'])).size,(2,2))

    def test_preview_pair_is_one_atomic_frame(self):
        camera = ReadOnlyCamera(enabled=True)
        camera.preview_number = 42
        camera.images = dict(rgb=b'RGB42', depth=b'DEPTH42')
        number, pair = camera.preview_pair()
        camera.images = dict(rgb=b'RGB43', depth=b'DEPTH43')
        self.assertEqual((number, pair), (42, dict(rgb=b'RGB42', depth=b'DEPTH42')))

    def test_onboard_camera_uses_fixed_local_binary_without_ssh(self):
        class Process:
            stdout = io.BytesIO(b'')
            def poll(self): return 0
        camera=ReadOnlyCamera(enabled=True,local_binary='/home/unitree/himloco-web/bin/himloco_camera_stream')
        with patch('camera.subprocess.Popen',return_value=Process()) as launch:
            camera._receive()
            self.assertEqual(launch.call_args.args[0],[camera.local_binary])

    def test_local_capture_uses_same_monotonic_clock_without_ssh_sync(self):
        header=struct.pack('<IIIQddfffff',0x48494d32,640,480,1,0,time.monotonic(),400,400,320,240,.001)
        class Process:
            stdout=io.BytesIO(header+bytes(640*480*5))
            def poll(self):return 0
        camera=ReadOnlyCamera(enabled=True,local_binary='/fixed/camera',sync_clock=True)
        with patch('camera.subprocess.Popen',return_value=Process()),patch('camera.subprocess.run') as sync:
            camera._receive()
            sync.assert_not_called()
        self.assertEqual(camera.latest_frame()['clock_uncertainty'],0)

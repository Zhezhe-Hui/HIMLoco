import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from access import Access, COOKIE, create_access
from server import ConsoleServer, OfflineConsole, handler_for


class AccessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        path = create_access(self.root/'access')
        record = json.loads(path.read_text())
        record['digest'] = hashlib.pbkdf2_hmac('sha256',b'test-password',bytes.fromhex(record['salt']),record['iterations']).hex()
        path.write_text(json.dumps(record))
        self.access = Access(path)
        self.console = OfflineConsole(runs=self.root/'runs',data=self.root/'data')
        self.server = ConsoleServer(('127.0.0.1',0),handler_for(self.console),access=self.access,allowed_hosts=['192.168.123.18'])
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.base = 'http://127.0.0.1:'+str(self.server.server_address[1])
        self.host = '192.168.123.18:'+str(self.server.server_address[1])

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()
        self.console.stop();self.tmp.cleanup()

    def request(self,path,payload=None,cookie=None,origin=None,host=None):
        headers={'Host':host or self.host}
        if payload is not None: headers['Content-Type']='application/json'
        if cookie: headers['Cookie']=cookie
        if origin: headers['Origin']=origin
        return urlopen(Request(self.base+path,data=None if payload is None else json.dumps(payload).encode(),headers=headers),timeout=3)

    def login(self):
        with self.request('/api/auth/login',{'username':'himloco','password':'test-password'},origin='http://'+self.host) as response:
            header=response.headers['Set-Cookie']
            self.assertIn('HttpOnly',header);self.assertIn('SameSite=Strict',header)
            return header.split(';')[0]

    def test_no_data_or_camera_actions_before_login(self):
        with self.request('/') as response:
            self.assertIn('loginForm',response.read().decode())
        for path in ('/api/state','/api/experiments','/api/camera/rgb','/app.js'):
            with self.assertRaises(HTTPError) as error: self.request(path)
            self.assertEqual(error.exception.code,401)
        with self.assertRaises(HTTPError) as error: self.request('/api/camera/connect',{})
        self.assertEqual(error.exception.code,401)

    def test_login_state_and_logout(self):
        cookie=self.login()
        with self.request('/api/state',cookie=cookie) as response:
            self.assertFalse(json.load(response)['motion_enabled'])
        with self.request('/api/auth/logout',{},cookie=cookie) as response:
            self.assertIn('Max-Age=0',response.headers['Set-Cookie'])
        with self.assertRaises(HTTPError) as error: self.request('/api/state',cookie=cookie)
        self.assertEqual(error.exception.code,401)

    def test_wrong_password_unlisted_host_and_cross_origin(self):
        for payload,host,origin,status in [
            ({'username':'himloco','password':'wrong'},self.host,None,400),
            ({'username':'himloco','password':'test-password'},'evil.example:8765',None,403),
            ({'username':'himloco','password':'test-password'},self.host,'http://evil.example',403)]:
            with self.assertRaises(HTTPError) as error: self.request('/api/auth/login',payload,host=host,origin=origin)
            self.assertEqual(error.exception.code,status)

    def test_session_expiry_and_login_throttle(self):
        token=self.access.login('himloco','test-password','peer')
        key=self.access.token_key(COOKIE+'='+token)
        self.access.sessions[key]=time.monotonic()-1
        self.assertFalse(self.access.valid(COOKIE+'='+token))
        for _ in range(8):
            with self.assertRaises(ValueError): self.access.login('himloco','wrong','other-peer')
        with self.assertRaises(RuntimeError): self.access.login('himloco','test-password','other-peer')

    def test_credential_files_private_and_initialization_does_not_reset(self):
        folder=self.root/'private'
        path=create_access(folder)
        before=path.read_bytes()
        self.assertEqual(path.stat().st_mode & 0o777,0o600)
        self.assertEqual((folder/'login-password.txt').stat().st_mode & 0o777,0o600)
        create_access(folder)
        self.assertEqual(path.read_bytes(),before)

    def test_interface_host_policy_rejects_foreign_addresses(self):
        self.server.allow_interface_addresses=True
        self.assertTrue(self.server.allowed_host('127.0.0.1'))
        self.assertFalse(self.server.allowed_host('evil.example'))
        self.assertFalse(self.server.allowed_host('203.0.113.9'))

"""Password verification and expiring browser sessions for the local LAN console."""
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import threading
import time
from http.cookies import SimpleCookie, CookieError

COOKIE = "himloco_session"


def create_access(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "access.json"
    if path.exists():
        return path
    password = secrets.token_urlsafe(15)
    salt = secrets.token_hex(16)
    record = dict(username="himloco", salt=salt, iterations=260000,
                  digest=hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 260000).hex())
    for target, value in ((path, json.dumps(record)), (folder/"login-password.txt", "账号：himloco\n密码："+password+"\n")):
        fd = os.open(str(target), os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as output:
            output.write(value)
    return path


class Access:
    def __init__(self, path, lifetime=8*3600):
        self.record = json.loads(Path(path).read_text())
        self.lifetime = lifetime
        self.sessions = {}
        self.attempts = {}
        self.lock = threading.Lock()

    def login(self, username, password, peer):
        if not isinstance(username,str) or not isinstance(password,str) or len(password)>256:
            raise ValueError("请输入账号和密码")
        with self.lock:
            now = time.monotonic()
            self.attempts = {k:v for k,v in self.attempts.items() if now-v[0]<60}
            started, count = self.attempts.get(peer, (now,0))
            if count>=8 or len(self.attempts)>256:
                raise RuntimeError("尝试次数过多，请一分钟后重试")
            self.attempts[peer] = (started,count+1)
            record = self.record
            digest = hashlib.pbkdf2_hmac("sha256",password.encode(),bytes.fromhex(record["salt"]),record["iterations"]).hex()
            if not (hmac.compare_digest(username,record["username"]) and hmac.compare_digest(digest,record["digest"])):
                raise ValueError("账号或密码错误")
            self.attempts.pop(peer,None)
            self.sessions = {k:v for k,v in self.sessions.items() if v>now}
            if len(self.sessions)>=64:
                self.sessions.pop(next(iter(self.sessions)))
            token = secrets.token_urlsafe(32)
            self.sessions[hashlib.sha256(token.encode()).hexdigest()] = now+self.lifetime
            return token

    def token_key(self, header):
        try:
            cookie = SimpleCookie()
            cookie.load(header or "")
            token = cookie[COOKIE].value
            return hashlib.sha256(token.encode()).hexdigest()
        except (KeyError, ValueError, CookieError):
            return None

    def valid(self, header):
        with self.lock:
            return self.sessions.get(self.token_key(header),0)>time.monotonic()

    def logout(self, header):
        with self.lock:
            self.sessions.pop(self.token_key(header),None)

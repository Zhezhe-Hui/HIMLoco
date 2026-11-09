#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task_python="${HIMLOCO_REAL_PYTHON:-/home/lenovo/miniconda3/envs/isclab/bin/python}"
board_host="${GO2_HOST:-unitree@192.168.50.1}"
board_control="${GO2_SSH_CONTROL:-/tmp/himloco-board-wifi-ssh}"
wheel_dir="$PWD/deploy/.cache/board-wheels"
mkdir -p "$wheel_dir"
if ! ssh -S "$board_control" -O check "$board_host" >/dev/null 2>&1; then
  ssh -MNf -S "$board_control" -o ControlPersist=8h -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=yes "$board_host"
fi
"$task_python" -m pip download --dest "$wheel_dir" --only-binary=:all: \
  --platform manylinux2014_aarch64 --python-version 38 --implementation cp --abi cp38 \
  -r deploy/requirements.board.txt
"$task_python" - "$wheel_dir" <<'PY'
import hashlib,json,sys,urllib.request
from pathlib import Path
info=json.load(urllib.request.urlopen('https://pypi.org/pypi/scikit-fmm/2023.4.2/json'))
source=next(item for item in info['urls'] if item['packagetype']=='sdist')
target=Path(sys.argv[1])/source['filename']
if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=source['digests']['sha256']:
    payload=urllib.request.urlopen(source['url']).read()
    assert hashlib.sha256(payload).hexdigest()==source['digests']['sha256']
    target.write_bytes(payload)
PY
"$task_python" deploy/deploy_runtime.py --host "$board_host" --control "$board_control" --wheels "$wheel_dir"
"$task_python" tools/go2/update_records.py --action board_runtime_deployed

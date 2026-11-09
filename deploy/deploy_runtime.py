#!/usr/bin/env python3
"""Deploy GO2 web/runtime code; startup connects read-only inputs only."""
import argparse
import getpass
import inspect
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEST = '/home/unitree/himloco-runtime'


def migrate_layout(root):
    """Move legacy board folders without discarding calibration, runs or deps."""
    from pathlib import Path
    root = Path(root)
    moves = [(root/'deploy/ours', root/'go2_runtime'),
             (root/'real_robot_deploy/runtime', root/'go2_runtime'),
             (root/'real_robot_deploy/web_console', root/'go2_web')]
    pending = [(source, target) for source, target in moves if source.exists()]
    targets = [target for _, target in pending]
    if len(set(targets)) != len(targets) or any(target.exists() for target in targets):
        raise RuntimeError('Old and new layouts coexist; refusing to overwrite board data')
    for source, target in pending:
        source.rename(target)
    for folder in (root/'deploy', root/'real_robot_deploy'):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    # These deployment helpers now live in deploy/, not in service/runtime code.
    for relative in ('go2_runtime/build.sh', 'go2_runtime/prepare_camera.sh',
                     'go2_web/deploy_runtime.py', 'go2_web/go2-ours-web.service',
                     'go2_web/requirements.board.txt'):
        path = root/relative
        if path.is_file():
            path.unlink()


def runtime_files(root=ROOT):
    """Explicit source manifest: data, local calibration and runs are never uploaded."""
    root = Path(root)
    groups = {
        'go2_web': ('server.py', 'camera.py', 'access.py', 'wired.py', 'static',
                    'test_camera.py', 'test_console.py', 'test_access.py', 'test_wired.py'),
        'go2_runtime': ('run_ours.py', 'estimate_floor.py', 'robot_bridge.cpp',
                       'bridge_safety.hpp', 'remote_takeover.hpp', 'bev_reduce.cpp',
                       'camera_stream.cpp', 'calibration.example.json', 'test_runtime.py',
                       'test_remote_takeover.cpp', 'test_bridge_safety.cpp', 'profile_planner.py'),
        'deploy': ('build_runtime.sh', 'go2-ours-web.service'),
        'legged_gym/legged_gym': ('__init__.py',),
        'legged_gym/legged_gym/envs/go2': ('bev_height_mapper.py', 'mppi_local_planner.py', 'viz_config.py'),
    }
    return [root/folder/name for folder, names in groups.items() for name in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default=os.environ.get('GO2_HOST', 'unitree@192.168.50.1'))
    parser.add_argument('--control', default=os.environ.get('GO2_SSH_CONTROL', '/tmp/himloco-board-wifi-ssh'))
    parser.add_argument('--wheels', type=Path, required=True, help='Python3.8/aarch64 wheels and scikit-fmm source')
    parser.add_argument('--activate-only', action='store_true', help='Verify and start an uploaded/built runtime')
    args = parser.parse_args()
    files = runtime_files()
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing deployment sources: "+", ".join(missing))
    ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-o', 'ControlPath='+args.control, args.host]

    def remote(command, **kwargs):
        return subprocess.run(ssh+[command], check=True, **kwargs)

    remote('test "$(uname -m)" = aarch64 && python3 -c "import numpy"')
    # Refuse maintenance while the web backend reports an active real task.
    preflight = '''import json, urllib.request
try:
    state=json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/state', timeout=3))
except OSError:
    state={}
task=(state.get('real') or {}).get('task') or {}
if task.get('outcome')=='running':
    raise SystemExit('Stop the active robot task before deployment')
'''
    remote('python3 -c '+shlex.quote(preflight))
    remote('python3 -c '+shlex.quote(inspect.getsource(migrate_layout)+'\nmigrate_layout('+repr(DEST)+')'))
    sdk = Path(os.environ.get('UNITREE_SDK2_DIR', '/home/lenovo/unitree_sdk2'))
    if not args.activate_only:
        print('Uploading web/runtime sources, ARM64 dependencies and SDK...', flush=True)
        with tempfile.TemporaryFile() as bundle:
            with tarfile.open(fileobj=bundle, mode='w:gz') as archive:
                def stable(info):
                    info.mtime = 0
                    return info
                def add(path, arcname):
                    archive.add(path, arcname=arcname, filter=stable)
                for file in files:
                    add(file, str(file.relative_to(ROOT)))
                for name in ('include', 'thirdparty/include', 'lib/aarch64', 'thirdparty/lib/aarch64'):
                    add(sdk/name, 'vendor/unitree_sdk2/'+name)
                add(args.wheels, 'wheelhouse')
            bundle.seek(0)
            remote('umask 077; mkdir -p '+DEST+'; tar xzf - -C '+DEST, stdin=bundle)
        remote('mkdir -p '+DEST+'/go2_web/data '+DEST+'/go2_runtime/runs')
        # Seed each calibration only once; subsequent deployments preserve the board.
        for relative in ('go2_web/data/calibration.web.json', 'go2_runtime/calibration.local.json'):
            seed = ROOT/relative
            target = DEST+'/'+relative
            if seed.exists():
                remote('test -f '+target+' || cat > '+target, input=seed.read_bytes())
        print('Installing isolated dependencies and compiling ARM64 binaries...', flush=True)
        remote('cd '+DEST+' && python3 -m pip install --no-index --no-deps --target go2_runtime/.deps wheelhouse/*.whl')
        remote('cd '+DEST+' && PYTHONPATH='+DEST+'/go2_runtime/.deps python3 -m pip install --no-index --no-deps --no-build-isolation --target go2_runtime/.deps wheelhouse/scikit-fmm-*.tar.gz')
        remote('cd '+DEST+' && UNITREE_SDK2_DIR='+DEST+'/vendor/unitree_sdk2 bash deploy/build_runtime.sh && g++ -O2 -std=c++11 go2_runtime/camera_stream.cpp -lrealsense2 -o go2_runtime/build/camera_stream')
    remote('cd '+DEST+' && PYTHONPATH='+DEST+'/go2_runtime/.deps:'+DEST+'/go2_runtime MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 python3 -S -c '+shlex.quote('import numpy,scipy,skimage,skfmm,matplotlib.pyplot,PIL,run_ours; run_ours.algorithms(); print("Native algorithms OK")'))
    password = getpass.getpass('Box sudo password (not saved): ')
    def sudo(command):
        remote('sudo -S -p "" sh -c '+shlex.quote(command), input=(password+'\n').encode())
    sudo('systemctl disable --now himloco-web.service; install -m 644 '+DEST+'/deploy/go2-ours-web.service /etc/systemd/system/go2-ours-web.service; systemctl daemon-reload; systemctl enable go2-ours-web.service; systemctl restart go2-ours-web.service')
    del password
    remote('systemctl is-enabled go2-ours-web.service && systemctl is-active go2-ours-web.service')
    print('Deployed '+DEST+'. No navigation task was started.')


if __name__ == '__main__':
    main()

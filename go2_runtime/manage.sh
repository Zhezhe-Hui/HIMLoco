#!/usr/bin/env bash
# Default action is a local check. Only the explicit `trial` action can move.
set -euo pipefail
entry_dir="$(cd "$(dirname "$0")" && pwd)"
project_dir="$(dirname "$entry_dir")"
runtime_dir="$entry_dir"
task_python="${HIMLOCO_REAL_PYTHON:-/home/lenovo/miniconda3/envs/isclab/bin/python}"
export HIMLOCO_REAL_PYTHON="$task_python"
requested_go2_host="${GO2_HOST:-}"
export GO2_HOST="${GO2_HOST:-unitree@192.168.123.18}"
export GO2_SSH_CONTROL="${GO2_SSH_CONTROL:-/tmp/himloco-ours-ssh}"
robot_interface="${GO2_INTERFACE:-enx00e04c584b68}"
action="${1:-check}"
if (($#)); then shift; fi
case "$action" in
  --check) action=check;;
  -h|--help|help)
    cat <<'HELP'
用法：bash go2_runtime/manage.sh [操作] [--duration 秒]
  check         默认：仅检查本机依赖和文件，不连接机器人
  prepare       编译本机通信桥，SSH 登录并部署盒子相机程序，不发速度
  shadow        电脑备用：准备后运行只读影子规划，默认 20 秒
  trial         显式运动测试：0.10 m/s，前方 0.5 m，最多 5 秒
                需现场输入 START；不会自动站立或切换运动模式
  test          离线 Python 测试和 C++ 看门狗检查，不连接机器人
  records       重新生成桌面上的两份操作记录
  web           启动本地离线网页，不连接狗，不编译或部署机器人程序
  web-wired     电脑备用实机网页；使用前须停盒子网页，避免相机冲突
  web-deploy    部署当前完整板载网页（等同 deploy_board_web.sh）
  --help        显示本说明
仅 shadow 支持 --duration；其他参数不透传，避免意外启用速度发送。
HELP
    exit 0;;
  check|prepare|shadow|trial|test|records|web|web-wired|web-deploy) ;;
  *) echo "未知操作：$action；请使用 --help" >&2; exit 2;;
esac
duration=20
while (($#)); do
  case "$1" in
    --duration)
      [[ "$action" == shadow ]] || { echo "仅 shadow 可指定 --duration" >&2; exit 2; }
      duration="${2:?--duration 缺少秒数}"; shift 2;;
    *) echo "不支持的参数：$1" >&2; exit 2;;
  esac
done

record_exit() {
  result=$?
  trap - EXIT
  "$task_python" "$project_dir/tools/go2/update_records.py" --source go2_runtime/manage.sh --action "$action" --exit-code "$result" || \
    echo "桌面记录更新失败，请运行 records 重试。" >&2
  exit "$result"
}
trap record_exit EXIT

check_local() {
  [[ -x "$task_python" ]] || { echo "Python 不存在：$task_python" >&2; return 1; }
  for tool in ssh g++ ip; do command -v "$tool" >/dev/null; done
  [[ -f "$runtime_dir/run_ours.py" && -f "$runtime_dir/calibration.local.json" ]] || {
    echo "缺少运行入口或本机标定文件，请查看 docs/go2/README.md" >&2; return 1;
  }
  PYTHONPATH="$runtime_dir/.deps${PYTHONPATH:+:$PYTHONPATH}" "$task_python" -c \
    'import numpy, scipy, skimage, skfmm, matplotlib; print("本机 Python 依赖检查通过")'
  "$task_python" -c 'import math,sys; v=float(sys.argv[1]); assert math.isfinite(v) and 0<v<=120, "时长必须在 (0,120] 秒"' "$duration"
  echo "项目：$project_dir"
  echo "运行时：$runtime_dir"
  echo "算力盒子：$GO2_HOST；网卡：$robot_interface"
}

prepare_camera() {
  if ! ssh -S "$GO2_SSH_CONTROL" -O check "$GO2_HOST" >/dev/null 2>&1; then
    echo "建立 SSH 连接；如提示密码，请在终端输入。脚本不保存密码。"
    ssh -MNf -S "$GO2_SSH_CONTROL" -o ControlPersist=8h \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -o ConnectTimeout=5 -o StrictHostKeyChecking=yes "$GO2_HOST"
  fi
  if ssh -T -o BatchMode=yes -o "ControlPath=$GO2_SSH_CONTROL" "$GO2_HOST" \
    'systemctl is-active --quiet go2-ours-web.service || systemctl is-active --quiet himloco-web.service'; then
    echo "盒子网页正在采集相机；请使用板载网页。切换电脑运行前须先停止盒子网页服务。" >&2
    return 2
  fi
  bash "$project_dir/deploy/prepare_camera.sh"
}

prepare() {
  check_local
  ip link show "$robot_interface" >/dev/null
  bash "$project_dir/deploy/build_runtime.sh"
  prepare_camera
  echo "本机桥和盒子相机程序已准备完成；尚未发送速度。"
}

case "$action" in
  web-wired)
    prepare
    echo "网线开发网页：http://localhost:8765；无登录，仅当前电脑。启动不会让狗运动。"
    PYTHONPATH="$runtime_dir/.deps${PYTHONPATH:+:$PYTHONPATH}" MPLBACKEND=Agg \
      "$task_python" "$project_dir/go2_web/server.py" --wired --interface "$robot_interface" \
      --camera-host "$GO2_HOST" --ssh-control "$GO2_SSH_CONTROL";;
  web-deploy)
    GO2_HOST="${requested_go2_host:-unitree@192.168.50.1}" bash "$project_dir/deploy/deploy.sh";;
  web)
    exec_note="http://127.0.0.1:8765"
    echo "启动离线网页：$exec_note；按 Ctrl+C 结束。无需开启机器狗。"
    "$task_python" "$project_dir/go2_web/server.py";;
  check) check_local;;
  prepare) prepare;;
  records) :;;
  test)
    check_local
    PYTHONPATH="$runtime_dir/.deps${PYTHONPATH:+:$PYTHONPATH}" MPLBACKEND=Agg \
      "$task_python" -m unittest discover -s "$runtime_dir" -p 'test_*.py' -v
    MPLBACKEND=Agg PYTHONPATH="$runtime_dir/.deps${PYTHONPATH:+:$PYTHONPATH}" \
      "$task_python" -m unittest discover -s "$project_dir/go2_web" -p 'test_*.py' -v
    PYTHONPATH="$runtime_dir/.deps${PYTHONPATH:+:$PYTHONPATH}" MPLBACKEND=Agg \
      "$task_python" -m unittest discover -s "$project_dir/tools/go2" -p 'test_*.py' -v
    mkdir -p "$runtime_dir/build"
    g++ -std=c++17 "$runtime_dir/test_bridge_safety.cpp" -o "$runtime_dir/build/test_bridge_safety"
    "$runtime_dir/build/test_bridge_safety"
    g++ -std=c++17 "$runtime_dir/test_remote_takeover.cpp" -o "$runtime_dir/build/test_remote_takeover"
    "$runtime_dir/build/test_remote_takeover"
    echo "离线 Python 和 C++ 检查通过；未连接机器人。";;
  shadow)
    prepare
    bash "$runtime_dir/run.sh" --interface "$robot_interface" --host "$GO2_HOST" \
      --ssh-control "$GO2_SSH_CONTROL" --calibration "$runtime_dir/calibration.local.json" \
      --duration "$duration";;
  trial)
    prepare
    echo "本操作会发送速度：前方目标 0.5m，MPPI 非停车采样速度固定 0.10m/s，最多 5s。"
    echo "请让狗正常站立，保持空旷通道，并由现场人员持遥控器接管。"
    read -r -p "准备实际行走时输入 START，其他输入取消：" confirmation
    [[ "$confirmation" == START ]] || { echo "已取消，未发送速度。"; exit 3; }
    bash "$runtime_dir/run.sh" --interface "$robot_interface" --host "$GO2_HOST" \
      --ssh-control "$GO2_SSH_CONTROL" --calibration "$runtime_dir/calibration.local.json" \
      --execute --preliminary-trial --goal-forward 0.5 --min-vx 0.10 --max-vx 0.10 --duration 5;;
esac

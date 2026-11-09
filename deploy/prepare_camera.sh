#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../go2_runtime"
host="${GO2_HOST:-unitree@192.168.123.18}"
control="${GO2_SSH_CONTROL:-/tmp/himloco-ours-ssh}"
# Requires an authenticated SSH control connection; stores no password.
ssh -T -o BatchMode=yes -o "ControlPath=$control" "$host" \
  'cat > /tmp/himloco_ours_camera_stream.cpp' < camera_stream.cpp
ssh -T -o BatchMode=yes -o "ControlPath=$control" "$host" \
  'g++ -O2 -std=c++11 /tmp/himloco_ours_camera_stream.cpp -lrealsense2 -o /tmp/himloco_ours_camera_stream'
# Keep the web preview executable across robot reboots; CLI retains its /tmp path.
ssh -T -o BatchMode=yes -o "ControlPath=$control" "$host" \
  'mkdir -p /home/unitree/.local/lib/himloco && cp /tmp/himloco_ours_camera_stream /home/unitree/.local/lib/himloco/camera_stream.new && mv /home/unitree/.local/lib/himloco/camera_stream.new /home/unitree/.local/lib/himloco/camera_stream'

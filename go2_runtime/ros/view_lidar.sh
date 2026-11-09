#!/usr/bin/env bash
set -e
go2_view_dir="$(cd "$(dirname "$0")" && pwd)"
source "$go2_view_dir/ros2_pc_env.sh"
exec env -u QT_QPA_PLATFORM_PLUGIN_PATH -u QT_PLUGIN_PATH /opt/ros/humble/bin/rviz2 -d "$go2_view_dir/go2_lidar.rviz"

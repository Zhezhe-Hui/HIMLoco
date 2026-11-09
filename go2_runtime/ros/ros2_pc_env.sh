# Source this file in a PC bash terminal. Current mode: wired DDS, read-only tools.
source /opt/ros/humble/setup.bash
go2_ros_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
go2_ros_prefix="$go2_ros_root/go2_web/data/ros-pc-overlay/root/opt/ros/humble"
if [[ ! -f "$go2_ros_prefix/lib/librmw_cyclonedds_cpp.so" ]]; then
  echo 'CycloneDDS local overlay missing; see ROS_TOPICS_GUIDE.md' >&2
  return 1
fi
export AMENT_PREFIX_PATH="$go2_ros_prefix:$AMENT_PREFIX_PATH"
export LD_LIBRARY_PATH="$go2_ros_prefix/lib:/opt/ros/humble/lib:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${GO2_ROS_IFACE:-enx00e04c584b68}\"/></Interfaces><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>"

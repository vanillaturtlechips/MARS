# Source this in the ROS2 / Nav2 / ros2-CLI / supervisor shell (subscriber side).
#   source deploy/isaac/env_ros2.sh
#
# System ROS2 Humble (py3.10) + the SAME UDP-only FastDDS profile Isaac uses,
# so `ros2 topic echo`, Nav2 and mars.ros.ros2_node see Isaac's topics.

source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/MARS/deploy/isaac/fastdds_udp_only.xml

echo "[env_ros2] ROS_DISTRO=$ROS_DISTRO RMW=$RMW_IMPLEMENTATION"
echo "[env_ros2] profile=$FASTRTPS_DEFAULT_PROFILES_FILE"

# Source this in the ROS2 / Nav2 / ros2-CLI / supervisor shell (subscriber side).
#   source deploy/isaac/env_ros2.sh
#
# System ROS2 Humble (py3.10) + the SAME UDP-only FastDDS profile Isaac uses,
# so `ros2 topic echo`, Nav2 and mars.ros.ros2_node see Isaac's topics.

# If an Isaac py3.11 venv is active (e.g. run_keepout_demo.sh launched from an
# isaac_venv311 shell), its libs (fmt/spdlog) break ROS2 rclpy with
# "librcl_logging_spdlog.so: undefined symbol". Strip the venv before sourcing ROS2.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  PATH="$(echo "$PATH" | tr ':' '\n' | grep -v "$VIRTUAL_ENV" | paste -sd:)"; export PATH
  unset VIRTUAL_ENV
fi
export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v isaac_venv312 | paste -sd:)"
export PYTHONPATH="$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v isaac_venv312 | paste -sd:)"

source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/MARS/deploy/isaac/fastdds_udp_only.xml

echo "[env_ros2] ROS_DISTRO=$ROS_DISTRO RMW=$RMW_IMPLEMENTATION"
echo "[env_ros2] profile=$FASTRTPS_DEFAULT_PROFILES_FILE"

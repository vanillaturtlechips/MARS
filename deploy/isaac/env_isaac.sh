# Source this in the Isaac Sim shell (publisher side).
#   source deploy/isaac/env_isaac.sh
#
# Makes Isaac Sim 5.1 (py3.11) use its INTERNAL ROS2 Humble libs and the
# UDP-only FastDDS profile so its topics reach the system ROS2 (py3.10).

source /workspace/isaac_venv311/bin/activate

# Strip any auto-sourced system ROS2 (py3.10) — it shadows Isaac's internal rclpy.
unset AMENT_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION \
      COLCON_PREFIX_PATH AMENT_CURRENT_PREFIX
export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v '/opt/ros/' | paste -sd:)
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/opt/ros/' | paste -sd:)

_EXT=/workspace/isaac_venv311/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$_EXT/humble/lib:$LD_LIBRARY_PATH
export FASTRTPS_DEFAULT_PROFILES_FILE=/workspace/MARS/deploy/isaac/fastdds_udp_only.xml

echo "[env_isaac] py=$(python --version 2>&1) RMW=$RMW_IMPLEMENTATION"
echo "[env_isaac] profile=$FASTRTPS_DEFAULT_PROFILES_FILE"
echo "[env_isaac] tip: pkill -9 -f isaacsim; pkill -9 -f kit  (kill stale sims first)"

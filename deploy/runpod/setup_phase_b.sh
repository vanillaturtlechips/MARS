#!/usr/bin/env bash
# One-shot rebuild of the Phase B (Isaac Sim + Nav2 + ROS2 bridge) environment
# on a fresh RunPod pod. Idempotent-ish; safe to re-run.
#
# Needs: Ubuntu 22.04, NVIDIA GPU, /workspace writable, internet.
# Does NOT install Postgres or the RL isaac_venv (3.10) — those are only for the
# Phase A brain demo, not the Nav2 keepout demo.
#
#   bash deploy/runpod/setup_phase_b.sh
#
# Then per shell:
#   Isaac side : source deploy/isaac/env_isaac.sh && python deploy/isaac/isaac_warehouse_ros2.py
#   ROS2 side  : source deploy/isaac/env_ros2.sh  && ros2 launch deploy/nav2/bringup_keepout.launch.py
set -e

REPO="${REPO:-/workspace/MARS}"
VENV="${VENV:-/workspace/isaac_venv311}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv_cache}"   # / is small; cache on /workspace

echo "==[1/4] ROS2 Humble + Nav2 + colcon =="
if [ ! -d /opt/ros/humble ]; then
    bash "$REPO/deploy/runpod/setup_ros2.sh"
else
    echo "  /opt/ros/humble present — skipping ROS2 install"
fi
# Nav2 packages (idempotent)
apt-get install -y ros-humble-navigation2 ros-humble-nav2-bringup >/dev/null 2>&1 || true

echo "==[2/4] mars_msgs colcon build =="
if [ ! -f "$HOME/ros2_ws/install/setup.bash" ]; then
    mkdir -p "$HOME/ros2_ws/src"
    ln -sfn "$REPO/agents/mars/ros2_ws/src/mars_msgs" "$HOME/ros2_ws/src/mars_msgs"
    bash -c "source /opt/ros/humble/setup.bash && cd $HOME/ros2_ws && colcon build --packages-select mars_msgs"
else
    echo "  mars_msgs already built — skipping"
fi

echo "==[3/4] uv + py3.11 Isaac Sim 5.1 venv =="
command -v uv >/dev/null 2>&1 || pip install uv -q
[ -d "$VENV" ] || uv venv --python 3.11 "$VENV"   # create only if missing (idempotent)
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"  # pypi.nvidia.com is slow
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install \
    isaacsim==5.1.0 isaacsim-rl==5.1.0 isaacsim-replicator==5.1.0 \
    isaacsim-extscache-physics==5.1.0 isaacsim-extscache-kit==5.1.0 \
    isaacsim-extscache-kit-sdk==5.1.0 isaacsim-ros2==5.1.0 \
    --extra-index-url https://pypi.nvidia.com

echo "==[4/4] verify =="
python - <<'PY'
import isaacsim  # noqa: F401
print("  isaacsim import OK")
PY
ls -d "$VENV"/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge \
    && echo "  ros2 bridge ext present"

echo
echo "DONE. Generate the Nav2 map once:  python $REPO/deploy/nav2/make_empty_map.py"
echo "Then bring up Isaac + Nav2 (see deploy/isaac/env_isaac.sh / env_ros2.sh)."

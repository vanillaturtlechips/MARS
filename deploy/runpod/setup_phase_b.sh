#!/usr/bin/env bash
# One-shot rebuild of the Phase B (Isaac Sim 6.0 + Nav2 + ROS2 bridge) environment
# on a fresh RunPod pod. Idempotent-ish; safe to re-run.
#
# Needs: Ubuntu 24.04, NVIDIA GPU, /workspace writable, internet.
#
#   bash deploy/runpod/setup_phase_b.sh
#
# Then per shell:
#   Isaac side : source deploy/isaac/env_isaac.sh && python deploy/isaac/isaac_warehouse_ros2.py
#   ROS2 side  : source deploy/isaac/env_ros2.sh  && ros2 launch deploy/nav2/bringup_keepout.launch.py
set -e

REPO="${REPO:-/workspace/MARS}"
VENV="${VENV:-/workspace/isaac_venv312}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv_cache}"

echo "==[1/4] ROS2 Jazzy + Nav2 + colcon =="
if [ ! -d /opt/ros/jazzy ]; then
    bash "$REPO/deploy/runpod/setup_ros2.sh"
else
    echo "  /opt/ros/jazzy present — skipping ROS2 install"
fi
apt-get install -y ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-slam-toolbox >/dev/null 2>&1 || true

echo "==[2/4] mars_msgs colcon build =="
if [ ! -f "$HOME/ros2_ws/install/setup.bash" ]; then
    mkdir -p "$HOME/ros2_ws/src"
    ln -sfn "$REPO/agents/mars/ros2_ws/src/mars_msgs" "$HOME/ros2_ws/src/mars_msgs"
    bash -c "source /opt/ros/jazzy/setup.bash && cd $HOME/ros2_ws && colcon build --packages-select mars_msgs"
else
    echo "  mars_msgs already built — skipping"
fi

echo "==[3/4] uv + py3.12 Isaac Sim 6.0 venv =="
command -v uv >/dev/null 2>&1 || pip install uv -q
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# NOTE: install via the [all,extscache] extras — NOT a hand-listed set of
# subpackages. The bare meta-package only installs the Python wheels; the actual
# Kit extensions (isaacsim.app.about, .gui, sensors, ros2.bridge, ...) live in
# the extscache wheels. Without `extscache` Kit tries to pull them from the
# online registry at runtime and dies with "No versions of isaacsim.app.about".
uv pip install \
    "isaacsim[all,extscache]==6.0.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease=allow

echo "==[4/4] verify =="
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1
python - <<'PY'
import isaacsim  # noqa: F401
print("  isaacsim import OK")
PY
# Verify the extscache extensions actually landed locally (the bit that was
# missing before): isaacsim.app.about is the one Kit failed to resolve.
for _e in isaacsim.app.about isaacsim.ros2.bridge; do
    if ls -d "$VENV"/lib/python3.12/site-packages/isaacsim/exts/"$_e" >/dev/null 2>&1; then
        echo "  ext present: $_e"
    else
        echo "  [WARN] ext MISSING: $_e"
    fi
done

echo
echo "DONE. Isaac + ROS2 환경 준비 완료."
echo "Isaac : source deploy/isaac/env_isaac.sh && python deploy/isaac/isaac_warehouse_ros2.py"
echo "ROS2  : source deploy/isaac/env_ros2.sh  && ros2 launch deploy/nav2/bringup_keepout.launch.py"

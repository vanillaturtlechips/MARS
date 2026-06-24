#!/usr/bin/env bash
# Install Isaac Sim 6.0 (py3.12) reliably on a flaky network.
# Single concurrent download + long timeout + retry loop. uv caches every
# COMPLETED wheel in UV_CACHE_DIR, so each retry resumes (only re-fetches what
# didn't finish) and progress is monotonic until it succeeds.
#
#   nohup bash deploy/runpod/install_isaac_pip.sh > /tmp/isaac_pip.log 2>&1 &
#   tail -f /tmp/isaac_pip.log
set -u

VENV="${VENV:-/workspace/isaac_venv312}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.uv_cache}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-900}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_ALLOW_ROOT=1

command -v uv >/dev/null 2>&1 || pip install uv -q
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

PKGS="isaacsim==6.0.0 isaacsim-rl==6.0.0 isaacsim-replicator==6.0.0 \
isaacsim-extscache-physics==6.0.0 isaacsim-extscache-kit==6.0.0 \
isaacsim-extscache-kit-sdk==6.0.0 isaacsim-ros2==6.0.0"

for i in $(seq 1 50); do
    echo "===== attempt $i ($(date +%H:%M:%S)) ====="
    # shellcheck disable=SC2086
    if uv pip install $PKGS --extra-index-url https://pypi.nvidia.com; then
        echo "===== INSTALL DONE ====="
        python -c "import isaacsim; print('isaacsim import OK')"
        ls -d "$VENV"/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.bridge \
            && echo "ros2 bridge ext present"
        exit 0
    fi
    echo "attempt $i failed — retrying in 5s (cached wheels are kept)"
    sleep 5
done
echo "===== gave up after 50 attempts ====="
exit 1

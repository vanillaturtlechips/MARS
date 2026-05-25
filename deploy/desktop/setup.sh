#!/bin/bash
# MARS Desktop (RunPod Kasm) 환경 설치
# 사용법: bash deploy/desktop/setup.sh
# 예상 시간: 약 20분 (Isaac Sim 다운로드 포함)

set -e

BASE_DIR="/workspace"
VENV_PATH="$BASE_DIR/isaac_venv"
ISAACLAB_PATH="$BASE_DIR/IsaacLab"
MARS_PATH="$BASE_DIR/MARS"
ISAACLAB_VERSION="v2.3.2"

_CUDA_RT="${CUDA_VERSION:-$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo '12.4')}"
_CUDA_MM=$(echo "$_CUDA_RT" | cut -d. -f1,2)
_CUDA_TAG=$(echo "$_CUDA_MM" | tr -d '.')

echo "════════════════════════════════════════════"
echo " MARS Desktop 환경 설치"
echo " BASE_DIR   : $BASE_DIR"
echo " CUDA 런타임: $_CUDA_MM  →  cu${_CUDA_TAG}"
echo "════════════════════════════════════════════"

# ── 사전 확인 ────────────────────────────────────────────────────
echo ""
echo "▶ 사전 확인"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null \
    && echo "  GPU OK" || echo "  [경고] nvidia-smi 실패"

# ── 1. 시스템 라이브러리 ─────────────────────────────────────────
echo "[1/7] 시스템 라이브러리..."
apt-get update -q 2>/dev/null || true
apt-get install -y --no-install-recommends \
    software-properties-common \
    libxt6 libxrandr2 libxcursor1 libxinerama1 \
    libgl1-mesa-glx libglu1-mesa \
    libvulkan1 libegl1 libgles2 \
    libxkbcommon0 libdbus-1-3 \
    git curl ffmpeg \
    python3-pip 2>/dev/null || true

# Ubuntu 20.04 기본 Python은 3.8 → 3.11 설치
if ! python3.11 --version &>/dev/null; then
    echo "  Python 3.11 설치..."
    add-apt-repository ppa:deadsnakes/ppa -y 2>/dev/null || true
    apt-get update -q 2>/dev/null || true
    apt-get install -y python3.11 python3.11-venv python3.11-dev 2>/dev/null || true
fi
echo "  완료"

# ── 캐시 경로 설정 ────────────────────────────────────────────────
export UV_CACHE_DIR="$BASE_DIR/uv_cache"
export PIP_CACHE_DIR="$BASE_DIR/pip_cache"
export TMPDIR="$BASE_DIR/tmp"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"
export UV_HTTP_TIMEOUT=600

# ── 2. uv ────────────────────────────────────────────────────────
echo "[2/7] uv 설치..."
python3.11 -m pip install uv -q
echo "  완료"

# ── 3. venv ──────────────────────────────────────────────────────
echo "[3/7] 가상환경: $VENV_PATH"
if [ -d "$VENV_PATH" ]; then
    echo "  기존 venv 재사용"
else
    uv venv "$VENV_PATH" --python python3.11
fi
source "$VENV_PATH/bin/activate"
echo "  완료"

# ── 4. Isaac Sim 5.1.0 + PyTorch ─────────────────────────────────
echo "[4/7] Isaac Sim 5.1.0 설치 (10~15분)..."

uv pip install \
    isaacsim==5.1.0 \
    isaacsim-rl==5.1.0 \
    isaacsim-replicator==5.1.0 \
    isaacsim-extscache-physics==5.1.0 \
    isaacsim-extscache-kit==5.1.0 \
    isaacsim-extscache-kit-sdk==5.1.0 \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match

case "$_CUDA_TAG" in
    128) TORCH_TAG="cu128" ;;
    126) TORCH_TAG="cu126" ;;
    *)   TORCH_TAG="cu124" ;;
esac
echo "  torch==2.7.0+${TORCH_TAG} 설치..."
uv pip install "torch==2.7.0" "torchvision==0.22.0" "numpy==1.26.4" \
    --index-url "https://download.pytorch.org/whl/${TORCH_TAG}" \
    --extra-index-url "https://pypi.org/simple" \
    --reinstall

SITE_PKG="$VENV_PATH/lib/python3.11/site-packages"
PXR_DIR=$(find "$VENV_PATH" -maxdepth 12 -name "pxr" -type d 2>/dev/null \
          | grep -v "__pycache__" | head -1)
if [ -n "$PXR_DIR" ]; then
    dirname "$PXR_DIR" > "$SITE_PKG/pxr_path.pth"
    echo "  pxr 경로 등록: $(dirname $PXR_DIR)"
fi
echo "  완료"

# ── 4.5. flatdict ────────────────────────────────────────────────
SITE_PKG="$VENV_PATH/lib/python3.11/site-packages"
_FD_DIR="/tmp/flatdict_src"
mkdir -p "$_FD_DIR"
python3.11 -m pip download flatdict==4.0.1 --no-deps --no-binary :all: -d "$_FD_DIR" -q
tar xzf "$_FD_DIR/flatdict-4.0.1.tar.gz" -C "$_FD_DIR"
cp "$_FD_DIR/flatdict-4.0.1/flatdict.py" "$SITE_PKG/"
_DI="$SITE_PKG/flatdict-4.0.1.dist-info"
mkdir -p "$_DI"
printf "Metadata-Version: 2.1\nName: flatdict\nVersion: 4.0.1\n" > "$_DI/METADATA"
printf "Wheel-Version: 1.0\nGenerator: manual\nRoot-Is-Purelib: true\n" > "$_DI/WHEEL"
echo "" > "$_DI/RECORD"
echo "  flatdict 완료"

# ── 5. Isaac Lab v2.3.2 ───────────────────────────────────────────
echo "[5/7] Isaac Lab $ISAACLAB_VERSION..."
if [ ! -d "$ISAACLAB_PATH" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git \
        --branch "$ISAACLAB_VERSION" --depth 1 "$ISAACLAB_PATH"
else
    echo "  기존 클론 재사용"
fi

cd "$ISAACLAB_PATH"
mkdir -p _isaac_sim
ln -sf "$(which python)" _isaac_sim/python.sh 2>/dev/null || true

uv pip install \
    -e source/isaaclab \
    -e source/isaaclab_assets \
    -e source/isaaclab_rl \
    -e source/isaaclab_tasks

if [ ! -d "$BASE_DIR/rsl_rl" ]; then
    git clone https://github.com/leggedrobotics/rsl_rl.git "$BASE_DIR/rsl_rl"
fi
cd "$BASE_DIR/rsl_rl" && git checkout v3.1.2 && cd -
uv pip install -e "$BASE_DIR/rsl_rl"

uv pip install \
    tensordict onnxscript warp-lang tensorboard \
    "onnx>=1.14.0" "onnxruntime>=1.16.0" setuptools
uv pip install "gymnasium==1.2.1"
echo "  완료"

# ── 6. MARS 코드 ─────────────────────────────────────────────────
echo "[6/7] MARS 코드..."
if [ ! -d "$MARS_PATH" ]; then
    git clone https://github.com/vanillaturtlechips/MARS.git "$MARS_PATH"
else
    cd "$MARS_PATH" && git pull origin main
fi
echo "  완료"

# ── 7. 검증 ──────────────────────────────────────────────────────
echo "[7/7] 설치 확인..."
cd "$MARS_PATH"
python -c "
import sys
ok = True

import torch
cuda_ok = torch.cuda.is_available()
print(f'  torch         {torch.__version__}  (cuda={torch.version.cuda})')
print(f'  CUDA 사용가능  {\"✓\" if cuda_ok else \"✗\"}')
if cuda_ok:
    print(f'  GPU           {torch.cuda.get_device_name(0)}')

import numpy as np
print(f'  numpy         {np.__version__}')

import isaacsim
print(f'  isaacsim      OK')

try:
    from pxr import Usd
    print(f'  pxr           OK')
except ImportError:
    print(f'  pxr           (AppLauncher 실행 후 자동 로드)')

try:
    import rsl_rl
    ver = getattr(rsl_rl, '__version__', '(버전 미노출)')
    print(f'  rsl_rl        {ver}  ✓')
except ImportError as e:
    print(f'  rsl_rl        FAIL: {e}')
    ok = False

try:
    from isaaclab_rl.rsl_rl.runners import OnPolicyRunner
    print(f'  isaaclab_rl   OK  ✓')
except ImportError as e:
    print(f'  isaaclab_rl   FAIL: {e}')
    ok = False

sys.exit(0 if ok else 1)
"

echo ""
echo "════════════════════════════════════════════"
echo " 설치 완료"
echo "════════════════════════════════════════════"
echo ""
echo "데모 실행:"
echo "  bash $MARS_PATH/deploy/desktop/run_demo.sh"

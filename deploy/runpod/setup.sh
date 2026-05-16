#!/bin/bash
# MARS RunPod 원클릭 환경 설치
# 사용법: bash deploy/runpod/setup.sh
# 예상 시간: 약 20분 (Isaac Sim 다운로드 포함)

set -e

VENV_PATH="/workspace/isaac_venv"
ISAACLAB_PATH="/workspace/IsaacLab"
MARS_PATH="/workspace/MARS"
ISAACLAB_VERSION="v2.3.2"

echo "════════════════════════════════════════════"
echo " MARS RunPod 환경 설치"
echo "════════════════════════════════════════════"

# ── 사전 확인: CUDA 접근 가능 여부 ─────────────────────────────
echo ""
echo "▶ 사전 확인"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null \
    && echo "  GPU OK" || echo "  [경고] nvidia-smi 실패"

CUDA_INIT=$(python3 -c "
import ctypes
try:
    ret = ctypes.CDLL('libcuda.so.1').cuInit(0)
    print(ret)
except:
    print(999)
" 2>/dev/null)

if [ "$CUDA_INIT" != "0" ]; then
    echo ""
    echo "  ╔══════════════════════════════════════════════╗"
    echo "  ║  [경고] CUDA cuInit 실패 (error $CUDA_INIT)          ║"
    echo "  ║  /dev/nvidia-caps/ 가 비어있는 컨테이너임.    ║"
    echo "  ║  이 Pod를 삭제하고 RunPod 공식 PyTorch/CUDA   ║"
    echo "  ║  템플릿으로 새 Pod를 생성해야 함.              ║"
    echo "  ║  환경만 설치 후 종료함.                        ║"
    echo "  ╚══════════════════════════════════════════════╝"
    echo ""
fi

# ── 1. 시스템 라이브러리 ────────────────────────────────────────
echo "[1/7] 시스템 라이브러리..."
apt-get update -q 2>/dev/null
apt-get install -y --no-install-recommends \
    libxt6 libxrandr2 libxcursor1 libxinerama1 \
    libgl1-mesa-glx libglu1-mesa \
    libvulkan1 libegl1 libgles2 \
    libxkbcommon0 libdbus-1-3 \
    git curl 2>/dev/null || true
echo "  완료"

# ── 캐시 경로를 Volume disk로 ──────────────────────────────────
export UV_CACHE_DIR=/workspace/uv_cache
export PIP_CACHE_DIR=/workspace/pip_cache
export TMPDIR=/workspace/tmp
mkdir -p /workspace/uv_cache /workspace/pip_cache /workspace/tmp

# ── 2. uv ──────────────────────────────────────────────────────
echo "[2/7] uv 설치..."
pip install uv -q
echo "  완료"

# ── 3. venv ────────────────────────────────────────────────────
echo "[3/7] 가상환경: $VENV_PATH"
if [ -d "$VENV_PATH" ]; then
    echo "  기존 venv 재사용"
else
    uv venv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"
echo "  완료"

# ── 4. Isaac Sim 5.1.0 + PyTorch ───────────────────────────────
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

# PyTorch cu128 명시적 설치
# unsafe-best-match 사용 시 cu126이 설치될 수 있으므로 +cu128 suffix 명시
uv pip install \
    "torch==2.7.0+cu128" \
    "torchvision==0.22.0+cu128" \
    "numpy==1.26.4" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match

# pxr 경로 .pth 등록 (sitecustomize.py 사용 금지 — import isaacsim이 CUDA 컨텍스트 오염)
SITE_PKG="$VENV_PATH/lib/python3.11/site-packages"
PXR_DIR=$(find "$VENV_PATH" -maxdepth 12 -name "pxr" -type d 2>/dev/null \
          | grep -v "__pycache__" | head -1)
if [ -n "$PXR_DIR" ]; then
    dirname "$PXR_DIR" > "$SITE_PKG/pxr_path.pth"
    echo "  pxr 경로 등록: $(dirname $PXR_DIR)"
else
    echo "  pxr: AppLauncher 실행 시 자동 추가됨"
fi

echo "  완료"

# ── 5. Isaac Lab v2.3.2 ────────────────────────────────────────
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
    -e source/isaaclab_tasks \
    --no-deps

# isaaclab_rl 의존성 (standalone rsl-rl 필수)
uv pip install \
    rsl-rl \
    tensorboard \
    "gymnasium>=0.29,<1.0" \
    "onnx>=1.14.0" \
    "onnxruntime>=1.16.0" \
    2>/dev/null || true

echo "  완료"

# ── 6. MARS 코드 ───────────────────────────────────────────────
echo "[6/7] MARS 코드..."
if [ ! -d "$MARS_PATH" ]; then
    git clone https://github.com/vanillaturtlechips/MARS.git "$MARS_PATH"
else
    cd "$MARS_PATH" && git pull origin main
fi
echo "  완료"

# ── 7. 검증 ───────────────────────────────────────────────────
echo "[7/7] 설치 확인..."
cd "$MARS_PATH"
python -c "
import sys
ok = True

import torch
cuda_ok = torch.cuda.is_available()
print(f'  torch         {torch.__version__}  (cuda={torch.version.cuda})')
print(f'  CUDA 사용가능  {\"✓\" if cuda_ok else \"✗  [컨테이너 재생성 필요]\"}')
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
    print(f'  rsl_rl        {rsl_rl.__version__}  ✓')
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
echo "훈련 실행:"
echo "  source $VENV_PATH/bin/activate"
echo "  cd $MARS_PATH"
echo "  python training/multi_robot/train_ippo.py --headless --num_envs 256"
echo "  python training/single_robot/train_manipulation.py --headless --num_envs 512"

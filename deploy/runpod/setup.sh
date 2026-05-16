#!/bin/bash
# RunPod Isaac Lab 환경 설치 스크립트
# 사용법: bash setup.sh

set -e  # 오류 시 즉시 중단

VENV_PATH="/workspace/isaac_venv"
ISAACLAB_PATH="/workspace/IsaacLab"
MARS_PATH="/workspace/MARS"
ISAACLAB_VERSION="v2.3.2"

echo "========================================"
echo " MARS RunPod 환경 설치"
echo "========================================"

# 1. uv 설치
echo "[1/6] uv 설치..."
pip install uv -q
echo "      완료"

# 2. venv 생성
echo "[2/6] 가상환경 생성: $VENV_PATH"
uv venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
echo "      완료"

# 3. Isaac Sim 설치
echo "[3/6] Isaac Sim 5.1.0 설치 (시간 걸림)..."
uv pip install \
    isaacsim==5.1.0 \
    isaacsim-rl==5.1.0 \
    isaacsim-replicator==5.1.0 \
    isaacsim-extscache-physics==5.1.0 \
    isaacsim-extscache-kit==5.1.0 \
    isaacsim-extscache-kit-sdk==5.1.0 \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match

# numpy 고정 (isaacsim 요구사항)
uv pip install "numpy==1.26.4" "torch==2.7.0" "torchvision==0.22.0" \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match
echo "      완료"

# 4. Isaac Lab 설치
echo "[4/6] Isaac Lab $ISAACLAB_VERSION 설치..."
if [ ! -d "$ISAACLAB_PATH" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git \
        --branch "$ISAACLAB_VERSION" \
        --depth 1 \
        "$ISAACLAB_PATH"
fi

cd "$ISAACLAB_PATH"
# isaaclab.sh 가 pip 을 찾을 수 있도록 symlink
mkdir -p _isaac_sim
ln -sf "$(which python)" _isaac_sim/python.sh 2>/dev/null || true

uv pip install \
    -e source/isaaclab \
    -e source/isaaclab_assets \
    -e source/isaaclab_rl \
    -e source/isaaclab_tasks \
    --no-deps
echo "      완료"

# 5. MARS 코드 클론
echo "[5/6] MARS 코드 클론..."
if [ ! -d "$MARS_PATH" ]; then
    git clone https://github.com/vanillaturtlechips/MARS.git "$MARS_PATH"
else
    cd "$MARS_PATH" && git pull origin main
fi
echo "      완료"

# 6. 동작 확인
echo "[6/6] 동작 확인..."
cd "$MARS_PATH"
python -c "
import torch
print(f'  torch:       {torch.__version__}')
print(f'  cuda:        {torch.cuda.is_available()}')
print(f'  GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"없음\"}')
import numpy as np
print(f'  numpy:       {np.__version__}')
import isaacsim
print(f'  isaacsim:    OK')
"

echo ""
echo "========================================"
echo " 설치 완료"
echo "========================================"
echo ""
echo "훈련 실행:"
echo "  source $VENV_PATH/bin/activate"
echo "  cd $MARS_PATH"
echo "  python training/single_robot/train_manipulation.py --headless --num_envs 512"
echo "  python training/multi_robot/train_ippo.py --headless --num_envs 256"

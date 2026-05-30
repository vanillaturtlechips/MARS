#!/bin/bash
# Paperspace Ubuntu 22.04 Desktop — Isaac Lab 설치 스크립트
# GPU: Quadro RTX 4000 (8GB), CUDA 12.2 드라이버
# 용도: GUI 데모 (훈련은 RunPod에서)
#
# 실행:
#   chmod +x scripts/setup_paperspace.sh
#   ./scripts/setup_paperspace.sh

set -e  # 오류 시 즉시 중단

echo "========================================"
echo "  MARS — Paperspace Isaac Lab 설치"
echo "========================================"

# ── 1. Python 3.10 ────────────────────────────────────────────────────
echo ""
echo "[1/6] Python 3.10 설치..."
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update -q
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils

python3.10 --version

# ── 2. venv ───────────────────────────────────────────────────────────
echo ""
echo "[2/6] 가상환경 생성: /workspace/isaac_venv"
python3.10 -m venv /workspace/isaac_venv
source /workspace/isaac_venv/bin/activate
pip install --upgrade pip setuptools wheel

# ── 3. Isaac Sim ──────────────────────────────────────────────────────
echo ""
echo "[3/6] Isaac Sim 4.5.0 설치 (20GB+, 시간 소요)..."
pip install isaacsim==4.5.0 --extra-index-url https://pypi.nvidia.com
pip install \
    isaacsim-rl \
    isaacsim-replicator \
    isaacsim-extscache-physics \
    isaacsim-extscache-kit-sdk \
    isaacsim-extscache-kit \
    isaacsim-app \
    --extra-index-url https://pypi.nvidia.com

# ── 4. Isaac Lab ──────────────────────────────────────────────────────
echo ""
echo "[4/6] Isaac Lab 설치..."
cd /workspace
if [ ! -d "IsaacLab" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git
fi
cd IsaacLab
pip install -e source/isaaclab
pip install -e source/isaaclab_assets
pip install -e source/isaaclab_rl

# ── 5. rsl_rl ─────────────────────────────────────────────────────────
echo ""
echo "[5/6] rsl_rl 설치..."
cd /workspace
if [ ! -d "rsl_rl" ]; then
    git clone https://github.com/leggedrobotics/rsl_rl.git
fi
cd rsl_rl
pip install -e .

# ── 6. MARS ───────────────────────────────────────────────────────────
echo ""
echo "[6/6] MARS 의존성 설치..."
cd /workspace/MARS
pip install torch numpy

echo ""
echo "========================================"
echo "  설치 완료!"
echo ""
echo "  이후 사용:"
echo "  source /workspace/isaac_venv/bin/activate"
echo ""
echo "  데모 실행 (GUI, 4 envs):"
echo "  python training/single_robot/train_pickplace.py \\"
echo "      --num_envs 4 \\"
echo "      --teacher_ckpt logs/warehouse_manipulation_teacher/model_2999.pt"
echo "========================================"

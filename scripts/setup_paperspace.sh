#!/bin/bash
# Paperspace Ubuntu 22.04 Desktop — Isaac Lab 데모 환경 설치
# GPU: Quadro RTX 4000 (8GB), CUDA 12.2 드라이버
# 용도: GUI 데모 (훈련은 RunPod에서)
#
# 버전 매칭 (중요):
#   Isaac Sim 4.5.0  ↔  Isaac Lab v2.0.0  ↔  rsl-rl-lib v2.2.x
#   (코드가 `import isaaclab` 신규 네이밍을 쓰므로 Isaac Lab 2.0+ 필수.
#    v1.x는 `omni.isaac.lab`이라 import부터 깨짐)
#
# 실행 (새 머신에서):
#   sudo apt-get update && sudo apt-get install -y git
#   git clone https://github.com/vanillaturtlechips/MARS.git /workspace/MARS
#   bash /workspace/MARS/scripts/setup_paperspace.sh
#
# 재실행 안전(idempotent): 이미 끝난 단계는 건너뜀. 중간에 깨지면 다시 돌려도 됨.

set -euo pipefail

VENV=/workspace/isaac_venv
ISAACLAB_DIR=/workspace/IsaacLab
RSL_RL_DIR=/workspace/rsl_rl
MARS_DIR=/workspace/MARS
ISAACLAB_TAG=v2.0.0          # Isaac Sim 4.5.0 호환
RSL_RL_TAG=v2.2.0            # Isaac Lab 2.0과 호환
LOG=/workspace/setup_paperspace.log

# ── 깨진 지점 즉시 보고 ────────────────────────────────────────────────
trap 'echo ""; echo "❌ 설치 실패 — 위 마지막 출력과 라인 $LINENO 확인. 로그: $LOG"; exit 1' ERR

exec > >(tee -a "$LOG") 2>&1
echo "========================================"
echo "  MARS — Paperspace Isaac Lab 설치 ($(date))"
echo "========================================"

# ── 0. 커널/드라이버 hold (재부팅 시 GUI 블랙스크린 방지) ──────────────
echo ""
echo "[0/7] 커널/드라이버 자동 업데이트 차단..."
sudo apt-mark hold linux-image-generic linux-headers-generic || true
sudo apt-mark hold 'nvidia-driver-*' 'libnvidia-*' 'xserver-xorg-video-nvidia-*' || true
sudo systemctl disable unattended-upgrades 2>/dev/null || true
sudo systemctl stop unattended-upgrades 2>/dev/null || true

# ── 디스크 여유 확인 (Isaac Sim 20GB+) ────────────────────────────────
echo ""
echo "[디스크] /workspace 여유 공간:"
df -h /workspace | tail -1
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
if [ "${AVAIL_GB:-0}" -lt 40 ]; then
    echo "⚠️  여유 공간 ${AVAIL_GB}GB < 40GB 권장. Isaac Sim 설치 중 디스크 부족 가능."
fi

# ── 1. Python 3.10 (+ venv 모듈) ──────────────────────────────────────
echo ""
echo "[1/7] Python 3.10 + venv..."
# python3.10 명령이 있어도 venv 모듈(ensurepip)이 없을 수 있음 → 모듈 작동 여부로 판단
if ! command -v python3.10 &>/dev/null; then
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -q
fi
if ! python3.10 -c "import ensurepip" &>/dev/null; then
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils
fi
python3.10 --version
python3.10 -c "import ensurepip; print('  ✅ venv 모듈 OK')"

# ── 2. venv ───────────────────────────────────────────────────────────
echo ""
echo "[2/7] 가상환경: $VENV"
if [ ! -f "$VENV/bin/activate" ]; then
    python3.10 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel

# ── 3. Isaac Sim 4.5.0 ────────────────────────────────────────────────
echo ""
echo "[3/7] Isaac Sim 4.5.0 (20GB+, 시간 소요)..."
if python -c "import isaacsim" 2>/dev/null; then
    echo "  이미 설치됨 — 건너뜀"
else
    pip install isaacsim==4.5.0 --extra-index-url https://pypi.nvidia.com
    pip install \
        isaacsim-rl isaacsim-replicator isaacsim-extscache-physics \
        isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app \
        --extra-index-url https://pypi.nvidia.com
fi
# 검증 게이트 — 여기서 막혀야 다음 단계가 헛돌지 않음
python -c "import isaacsim; print('  ✅ isaacsim import OK')"

# ── 4. Isaac Lab v2.0.0 ───────────────────────────────────────────────
echo ""
echo "[4/7] Isaac Lab $ISAACLAB_TAG..."
if [ ! -d "$ISAACLAB_DIR/source/isaaclab" ]; then
    rm -rf "$ISAACLAB_DIR"
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
cd "$ISAACLAB_DIR"
git fetch --tags --quiet || true
git checkout "$ISAACLAB_TAG"
pip install -e source/isaaclab
pip install -e source/isaaclab_assets
pip install -e source/isaaclab_rl
# 검증 게이트 — isaaclab_rl 래퍼는 rsl_rl 의존이라 5단계 이후에 검증(아래)
python -c "import isaaclab; from isaaclab.app import AppLauncher; print('  ✅ isaaclab import OK')"

# ── 5. rsl_rl (코드가 /workspace/rsl_rl 소스 경로를 sys.path에 주입) ───
echo ""
echo "[5/7] rsl_rl $RSL_RL_TAG..."
if [ ! -d "$RSL_RL_DIR/.git" ]; then
    rm -rf "$RSL_RL_DIR"
    git clone https://github.com/leggedrobotics/rsl_rl.git "$RSL_RL_DIR"
fi
cd "$RSL_RL_DIR"
git fetch --tags --quiet || true
git checkout "$RSL_RL_TAG" 2>/dev/null || echo "  (태그 $RSL_RL_TAG 없음 — 기본 브랜치 사용)"
pip install -e .
python -c "from rsl_rl.runners import OnPolicyRunner; print('  ✅ rsl_rl import OK')"
# 주의: isaaclab_rl 래퍼/isaaclab.envs 는 omni.kit 의존이라 AppLauncher 없이 import 불가.
#       따라서 런타임 import 대신 pip 설치 여부만 확인 (실제 검증은 데모 실행으로).
pip show isaaclab_rl >/dev/null 2>&1 && echo "  ✅ isaaclab_rl 설치됨 (런타임 검증은 데모 실행 시)"

# ── 6. MARS 의존성 + git 권한 ─────────────────────────────────────────
echo ""
echo "[6/7] MARS 의존성..."
git config --global --add safe.directory "$MARS_DIR" || true
cd "$MARS_DIR"
pip install torch numpy tensordict   # tensordict: ippo_wrapper의 obs TensorDict 묶음용

# ── 7. 최종 검증 ──────────────────────────────────────────────────────
echo ""
echo "[7/7] 데모 모델 존재 확인..."
for m in logs/warehouse_pickplace/model_300.pt logs/warehouse_mappo/model_10998.pt; do
    if [ -f "$m" ]; then echo "  ✅ $m ($(du -h "$m" | cut -f1))"; else echo "  ❌ $m 없음 — git pull 필요"; fi
done

echo ""
echo "========================================"
echo "  ✅ 설치 완료!"
echo ""
echo "  사용:"
echo "    source $VENV/bin/activate"
echo "    cd $MARS_DIR"
echo ""
echo "  MARL 비주얼 데모 (iw.hub + 회피):"
echo "    python training/multi_robot/demo_play.py \\"
echo "      --checkpoint logs/warehouse_mappo/model_10998.pt \\"
echo "      --num_envs 1 --max_steps 1500"
echo ""
echo "  PickPlace 데모:"
echo "    python training/single_robot/eval_manipulation.py \\"
echo "      --checkpoint logs/warehouse_pickplace/model_300.pt --num_envs 4"
echo "========================================"

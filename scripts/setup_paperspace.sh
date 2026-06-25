#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
#  MARS — Paperspace (Quadro RTX 4000, 드라이버 535) Isaac Lab 데모 환경
# ════════════════════════════════════════════════════════════════════════
#  용도: GUI 데모 (훈련은 RunPod). 한 방에 끝나도록 설계 — 중간 땜질 없음.
#
#  ※ 핵심 제약 (이걸 어기면 전부 깨짐) ※
#    드라이버 535  →  Isaac Sim 6.0 불가  →  Isaac Sim 4.5.0 고정
#    Isaac Sim 4.5 →  torch == 2.5.1 고정 (2.6+ 깔면 isaacsim 깨짐)
#    모델은 rsl_rl 3.x 산출물 → rsl_rl 필요. 단 3.1.2는 torch를 2.6+로
#    끌어올려 충돌 → rsl_rl 3.0.1 을 --no-deps 로 설치해 torch 보존.
#
#  검증된 버전 조합:
#    Python 3.10 · Isaac Sim 4.5.0 · Isaac Lab v2.0.0 · torch 2.5.1
#    · rsl_rl 3.0.1(--no-deps) · tensordict
#
#  실행 (새 머신):
#    sudo mkdir -p /workspace && sudo chown -R "$USER":"$USER" /workspace
#    sudo apt-get update && sudo apt-get install -y git tmux
#    git clone https://github.com/vanillaturtlechips/MARS.git /workspace/MARS
#    tmux new -s setup            # SSH 끊겨도 살아남게
#    bash /workspace/MARS/scripts/setup_paperspace.sh
#
#  GUI는 SSH가 아니라 Paperspace '데스크탑 화면'의 터미널에서 실행해야 창이 뜸.
# ════════════════════════════════════════════════════════════════════════

set -euo pipefail

VENV=/workspace/isaac_venv
ISAACLAB_DIR=/workspace/IsaacLab
RSL_RL_DIR=/workspace/rsl_rl
MARS_DIR=/workspace/MARS
ISAACLAB_TAG=v2.0.0          # Isaac Sim 4.5.0 호환 (드라이버 535로 돌릴 수 있는 최신 묶음)
RSL_RL_TAG=v3.0.1           # torch 2.5.1과 호환되는 rsl_rl 3.x (3.1.2는 torch 2.6+ 강제)
TORCH_VER=2.5.1
TV_VER=0.20.1
LOG=/workspace/setup_paperspace.log

trap 'echo ""; echo "❌ 설치 실패 — 위 마지막 출력과 라인 $LINENO 확인. 로그: $LOG"; exit 1' ERR
exec > >(tee -a "$LOG") 2>&1

echo "════════════════════════════════════════════"
echo "  MARS Paperspace 설치 ($(date))"
echo "════════════════════════════════════════════"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "[경고] nvidia-smi 실패"

# ── 0. 커널/드라이버 hold (재부팅 시 GUI 블랙스크린 방지) ──────────────
echo ""; echo "[0/7] 커널/드라이버 자동 업데이트 차단..."
sudo apt-mark hold linux-image-generic linux-headers-generic 2>/dev/null || true
sudo apt-mark hold 'nvidia-driver-*' 'libnvidia-*' 'xserver-xorg-video-nvidia-*' 2>/dev/null || true
sudo systemctl disable --now unattended-upgrades 2>/dev/null || true

# ── 1. Python 3.10 (+ venv 모듈) ──────────────────────────────────────
echo ""; echo "[1/7] Python 3.10 + venv..."
if ! python3.12 -c "import ensurepip" &>/dev/null; then
    sudo add-apt-repository ppa:deadsnakes/ppa -y 2>/dev/null || true
    sudo apt-get update -q
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev python3.12-distutils
fi
python3.12 --version

# ── 2. venv (깨진 상태면 깨끗이 재생성) ───────────────────────────────
echo ""; echo "[2/7] venv: $VENV"
if [ ! -f "$VENV/bin/activate" ]; then
    python3.12 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel -q

# ── 3. Isaac Sim 4.5.0 (이미 있으면 건너뜀) ───────────────────────────
echo ""; echo "[3/7] Isaac Sim 4.5.0 (20GB+, 최초 1회만)..."
if python -c "import isaacsim" 2>/dev/null; then
    echo "  이미 설치됨 — 건너뜀"
else
    pip install isaacsim==4.5.0 --extra-index-url https://pypi.nvidia.com
    pip install \
        isaacsim-rl isaacsim-replicator isaacsim-extscache-physics \
        isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app \
        --extra-index-url https://pypi.nvidia.com
fi

# ── 4. torch 2.5.1 정밀 고정 (★ 깨진 nccl/torch 복구의 핵심 ★) ────────
echo ""; echo "[4/7] torch $TORCH_VER 고정 (force-reinstall로 nccl 심볼 깨짐 복구)..."
NEED_TORCH_FIX=1
if python - <<PY 2>/dev/null
import torch, sys
sys.exit(0 if torch.__version__.startswith("$TORCH_VER") and torch.cuda.is_available() else 1)
PY
then NEED_TORCH_FIX=0; fi
if [ "$NEED_TORCH_FIX" = "1" ]; then
    pip install --force-reinstall "torch==$TORCH_VER" "torchvision==$TV_VER"
else
    echo "  torch $TORCH_VER + CUDA 정상 — 건너뜀"
fi
python -c "import torch; assert torch.__version__.startswith('$TORCH_VER'), torch.__version__; print(f'  ✅ torch {torch.__version__} CUDA={torch.cuda.is_available()}')"

# ── 5. Isaac Lab v2.0.0 ───────────────────────────────────────────────
echo ""; echo "[5/7] Isaac Lab $ISAACLAB_TAG..."
if [ ! -d "$ISAACLAB_DIR/source/isaaclab" ]; then
    rm -rf "$ISAACLAB_DIR"
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
cd "$ISAACLAB_DIR"
git fetch --tags --quiet || true
git checkout "$ISAACLAB_TAG"
# isaaclab은 deps와 함께 설치 — torch==2.5.1을 '요구'하므로 torch를 안 바꾸고,
# 오히려 prettytable·pkg_resources 등 필요한 패키지를 알아서 끌어옴.
# (torch를 깨는 범인은 rsl_rl뿐 → --no-deps는 6단계 rsl_rl에만 적용)
pip install -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_rl
pip install tensordict gitpython          # rsl_rl(--no-deps)가 쓰는 부수 패키지
# isaaclab 설치가 torch를 안 건드렸는지 재확인
python -c "import torch; assert torch.__version__.startswith('$TORCH_VER'), f'torch 오염: {torch.__version__}'; print(f'  ✅ torch {torch.__version__} 유지')"
python -c "import isaaclab; from isaaclab.app import AppLauncher; print('  ✅ isaaclab import OK')"

# ── 6. rsl_rl 3.0.1 (--no-deps 로 torch 보존) ─────────────────────────
echo ""; echo "[6/7] rsl_rl $RSL_RL_TAG (--no-deps: torch 2.5.1 보존)..."
rm -rf /home/rsl_rl                       # 이전에 엉뚱한 위치에 받은 것 제거
if [ ! -d "$RSL_RL_DIR/.git" ]; then
    rm -rf "$RSL_RL_DIR"
    git clone https://github.com/leggedrobotics/rsl_rl.git "$RSL_RL_DIR"
fi
cd "$RSL_RL_DIR"
git fetch --tags --quiet || true
git checkout "$RSL_RL_TAG"
pip install --no-deps -e .                # ★ --no-deps 가 torch 업그레이드 차단 ★
python -c "from rsl_rl.runners import OnPolicyRunner; print('  ✅ rsl_rl import OK')"
# torch가 여전히 2.5.1인지 재확인 (rsl_rl이 못 건드렸는지)
python -c "import torch; assert torch.__version__.startswith('$TORCH_VER'), f'torch 오염됨: {torch.__version__}'; print(f'  ✅ torch {torch.__version__} 유지됨')"

# ── 7. MARS + 최종 확인 ───────────────────────────────────────────────
echo ""; echo "[7/7] MARS 의존성 + 권한..."
git config --global --add safe.directory "$MARS_DIR" || true
cd "$MARS_DIR"
git config --global --add safe.directory "$ISAACLAB_DIR" 2>/dev/null || true
git config --global --add safe.directory "$RSL_RL_DIR" 2>/dev/null || true
echo ""
echo "데모 모델 확인:"
for m in logs/warehouse_pickplace/model_300.pt logs/warehouse_mappo/model_10998.pt; do
    [ -f "$m" ] && echo "  ✅ $m ($(du -h "$m"|cut -f1))" || echo "  ❌ $m 없음 — git pull"
done

echo ""
echo "════════════════════════════════════════════"
echo "  ✅ 설치 완료!  버전 요약:"
python - <<'PY'
import torch, rsl_rl, isaacsim
print(f"    torch    {torch.__version__}  (CUDA {torch.cuda.is_available()})")
try:
    import importlib.metadata as M
    print(f"    rsl-rl   {M.version('rsl-rl-lib')}")
except Exception: pass
PY
echo ""
echo "  ⚠️ GUI는 SSH가 아니라 Paperspace '데스크탑 화면'의 터미널에서 실행:"
echo ""
echo "    source $VENV/bin/activate && cd $MARS_DIR"
echo ""
echo "    # MARL 비주얼 데모 (iw.hub + 회피)"
echo "    python training/multi_robot/demo_play.py \\"
echo "      --checkpoint logs/warehouse_mappo/model_10998.pt --num_envs 1 --max_steps 1500"
echo "════════════════════════════════════════════"

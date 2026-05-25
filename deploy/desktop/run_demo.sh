#!/bin/bash
# MARS Desktop 데모 실행 (GUI 모드 — Xvfb 불필요)
#
# 사용법:
#   bash deploy/desktop/run_demo.sh [ckpt] [num_envs] [num_episodes]
#
# 예시:
#   bash deploy/desktop/run_demo.sh                          # 기본값
#   bash deploy/desktop/run_demo.sh ~/MARS/logs/warehouse_manipulation_teacher/model_2999.pt 4 10

BASE_DIR="/workspace"
VENV_PATH="$BASE_DIR/isaac_venv"
MARS_PATH="$BASE_DIR/MARS"

CKPT="${1:-$MARS_PATH/logs/warehouse_manipulation_teacher/model_2999.pt}"
NUM_ENVS="${2:-4}"
NUM_EPISODES="${3:-0}"

echo "============================================"
echo " MARS Desktop 데모"
echo " ckpt      : $CKPT"
echo " num_envs  : $NUM_ENVS"
echo " episodes  : $NUM_EPISODES (0=무한)"
echo "============================================"

source "$VENV_PATH/bin/activate"
cd "$MARS_PATH"

python training/single_robot/demo_manipulation.py \
    --ckpt "$CKPT" \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES"

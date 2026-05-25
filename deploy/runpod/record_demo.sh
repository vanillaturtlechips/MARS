#!/bin/bash
# Phase 2 데모 녹화 — Xvfb 가상 디스플레이 + ffmpeg x11grab
#
# 사용법:
#   bash deploy/runpod/record_demo.sh
#
# 출력: /workspace/phase2_demo.mp4

set -e

CKPT="${1:-logs/warehouse_manipulation/model_2999.pt}"
OUTPUT="${2:-/workspace/phase2_demo.mp4}"
NUM_EPISODES="${3:-5}"
RES="1280x720"

echo "============================================"
echo " Phase 2 데모 녹화"
echo " ckpt       : $CKPT"
echo " output     : $OUTPUT"
echo " episodes   : $NUM_EPISODES"
echo "============================================"

# 1. 의존성 설치
echo "[1/4] 패키지 설치..."
apt-get install -y --quiet xvfb ffmpeg 2>/dev/null || true

# 2. 가상 X11 디스플레이 시작
echo "[2/4] Xvfb 가상 디스플레이 시작 (:99)..."
pkill Xvfb 2>/dev/null || true
sleep 1
Xvfb :99 -screen 0 "${RES}x24" -ac +extension GLX +render &
XVFB_PID=$!
sleep 3
export DISPLAY=:99
echo "  DISPLAY=$DISPLAY  PID=$XVFB_PID"

# 3. ffmpeg 녹화 시작 (백그라운드)
echo "[3/4] ffmpeg 녹화 시작..."
ffmpeg -y \
  -f x11grab -r 30 -s "$RES" -i ":99" \
  -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
  "$OUTPUT" \
  > /tmp/ffmpeg_record.log 2>&1 &
FFMPEG_PID=$!
sleep 2
echo "  ffmpeg PID=$FFMPEG_PID"

# 4. 데모 실행 (디스플레이 있음 → GLFW 성공)
echo "[4/4] Isaac Sim 데모 실행..."
source /workspace/isaac_venv/bin/activate
cd /workspace/MARS

python training/single_robot/demo_manipulation.py \
  --ckpt "$CKPT" \
  --num_envs 4 \
  --num_episodes "$NUM_EPISODES" \
  2>&1

# 5. 정리
echo ""
echo "[완료] 녹화 종료..."
kill $FFMPEG_PID 2>/dev/null || true
wait $FFMPEG_PID 2>/dev/null || true
kill $XVFB_PID  2>/dev/null || true

if [ -f "$OUTPUT" ]; then
  SIZE=$(du -sh "$OUTPUT" | cut -f1)
  echo "  → $OUTPUT  ($SIZE)"
  echo ""
  echo "로컬 다운로드:"
  echo "  scp -i ~/.ssh/id_ed25519 2fbfjj97hq7ooo-6441169a@ssh.runpod.io:$OUTPUT ~/Downloads/"
else
  echo "[오류] 파일 생성 안 됨 — /tmp/ffmpeg_record.log 확인"
fi

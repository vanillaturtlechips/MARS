#!/bin/bash
# 팔(Franka) 픽앤플레이스 데모 녹화.
# [변경] Xvfb 화면캡처(x11grab) → 헤드리스 카메라 직접 녹화. RTX 뷰포트가 가상화면에 합성 안 돼
#        검게 찍히던 문제 해결 + Isaac Sim UI 안 들어감(nav 데모와 동일 방식, view_camera 출력).
#
# 사용: bash deploy/runpod/record_demo.sh [ckpt] [output] [episodes]
set -e
cd /workspace/MARS

CKPT="${1:-logs/warehouse_pickplace/model_300.pt}"   # obs23/act4 — WarehouseManipulationEnv 규격 일치
OUTPUT="${2:-/workspace/phase2_demo.mp4}"
NUM_EPISODES="${3:-5}"

echo "[record] ckpt=$CKPT  out=$OUTPUT  episodes=$NUM_EPISODES  (헤드리스 카메라 녹화, UI 없음)"
rm -f "$OUTPUT"   # stale 파일이 ✅로 오인되지 않게 먼저 삭제
LOG=/tmp/arm_record.log
python training/single_robot/demo_manipulation.py \
  --ckpt "$CKPT" --num_envs 4 --num_episodes "$NUM_EPISODES" \
  --record --video_out "$OUTPUT" --headless --enable_cameras > "$LOG" 2>&1 || true
echo "=== 핵심 로그 ==="
grep -aE "\[Actor\]|place_rate|녹화 저장|프레임|Error|Traceback" "$LOG" | tail -25
echo "=== 마지막 15줄(에러 확인용) ==="
tail -15 "$LOG"
echo "================="

if [ -f "$OUTPUT" ]; then
  echo "[record] ✅ $OUTPUT ($(du -sh "$OUTPUT" | cut -f1))"
  echo "[record] 다운로드: scp -P {PORT} -i ~/.ssh/id_ed25519 root@{POD_IP}:$OUTPUT ."
else
  echo "[record] ❌ 파일 없음 — 위 로그 확인 (프레임 0개면 카메라 렌더 문제)"
fi

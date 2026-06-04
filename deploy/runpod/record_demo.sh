#!/bin/bash
# 팔(Franka) 픽앤플레이스 데모 녹화 — 검증된 demo_record.py 사용.
#   demo_record.py가 headless+enable_cameras를 내부에서 강제하고, env.reset 후 camera.reset로
#   초기화한 뒤 카메라 프레임을 PNG→MP4로 만듦(UI 없음, 검은화면 없음). nav 데모와 동일 원리.
#
# 사용: bash deploy/runpod/record_demo.sh [ckpt] [output] [episodes]
set -e
cd /workspace/MARS

CKPT="${1:-logs/warehouse_pickplace/model_300.pt}"   # obs23/act4 — WarehouseManipulationEnv 규격 일치
OUTPUT="${2:-/workspace/phase2_demo.mp4}"
NUM_EPISODES="${3:-5}"
CAM_EYE="${4:-2.0,1.8,1.6}"        # 카메라 위치 (조정: 4번째 인자)
CAM_TARGET="${5:-0.5,0.0,0.95}"   # 카메라 타겟 (작업공간 높이 z≈0.95)

# demo_record.py는 frames_to_video에서 ffmpeg(subprocess) 사용 → 없으면 설치
command -v ffmpeg >/dev/null 2>&1 || { echo "[record] ffmpeg 설치..."; apt-get install -y --quiet ffmpeg >/dev/null 2>&1 || true; }

rm -f "$OUTPUT"   # stale 파일 오인 방지
LOG=/tmp/arm_record.log
echo "[record] demo_manipulation.py --record (env.render+RecordVideo, nav 데모와 동일 경로)  ckpt=$CKPT  ep=$NUM_EPISODES"
python training/single_robot/demo_manipulation.py \
  --ckpt "$CKPT" --num_envs 1 --num_episodes "$NUM_EPISODES" \
  --record --video_out "$OUTPUT" --cam_eye "$CAM_EYE" --cam_target "$CAM_TARGET" \
  --headless > "$LOG" 2>&1 || true

echo "=== 핵심 로그 ==="
grep -aE "\[Actor\]|에피소드|place_rate|프레임|frame|Error|Traceback|mp4|saved|저장" "$LOG" | tail -25
echo "=== 마지막 12줄 ==="
tail -12 "$LOG"
echo "================="

if [ -f "$OUTPUT" ]; then
  echo "[record] ✅ $OUTPUT ($(du -sh "$OUTPUT" | cut -f1))"
  echo "[record] 다운로드: scp -P {PORT} -i ~/.ssh/id_ed25519 root@{POD_IP}:$OUTPUT ."
else
  echo "[record] ❌ 파일 없음 — 위 마지막 줄 확인"
fi

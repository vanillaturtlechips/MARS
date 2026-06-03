#!/bin/bash
# 데모 렌더 헬퍼 — 긴 명령 줄바꿈 깨짐 방지용. 사용: bash deploy/runpod/render_demo.sh [video_length]
# 결과: logs/demo_videos/demo-step-0.mp4 + 첫 프레임 /tmp/check.png
set -e
cd /workspace/MARS

CKPT="logs/warehouse_mappo_extobs/model_shelf_final.pt"
VLEN="${1:-60}"   # 기본 60스텝(빠른 확인). 풀 영상은: bash ... 1500

echo "[render] ckpt=$CKPT  video_length=$VLEN"
python training/multi_robot/demo_play.py \
  --task --extended_obs \
  --checkpoint "$CKPT" \
  --video --video_length "$VLEN" --headless > /tmp/d.log 2>&1

echo "=== 랙/적재 관련 로그 ==="
grep -iE "절차적|랙|선반|적재|프롭|박스|RACK" /tmp/d.log || echo "(관련 로그 없음 — tail로 확인)"
echo "=== 마지막 5줄 ==="
tail -5 /tmp/d.log

# 첫 프레임 추출(ffmpeg 있으면)
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -i logs/demo_videos/demo-step-0.mp4 -vframes 1 /tmp/check.png >/dev/null 2>&1 && echo "[render] 스샷 저장: /tmp/check.png"
else
  echo "[render] ffmpeg 없음 → logs/demo_videos/demo-step-0.mp4 직접 열어 스샷"
fi

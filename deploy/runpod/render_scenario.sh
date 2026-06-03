#!/bin/bash
# 시나리오 쇼케이스 렌더. 사용:
#   bash deploy/runpod/render_scenario.sh [video_length] [scn_only]
#   - video_length: 프레임수(=초×15). 기본 900(~60s)
#   - scn_only: -1=전체순차(기본) / 0=S1교행 / 1=S2삼각 / 2=S4스테이션 / 3=S5혼잡 (그것만 반복재생)
# 예) 전체:        bash deploy/runpod/render_scenario.sh 900
#     S1만 600:    bash deploy/runpod/render_scenario.sh 600 0
set -e
cd /workspace/MARS

CKPT="logs/warehouse_mappo_extobs/model_shelf_final.pt"
VLEN="${1:-900}"
ONLY="${2:--1}"

echo "[render-scn] ckpt=$CKPT  video_length=$VLEN  scn_only=$ONLY"
python training/multi_robot/demo_play.py \
  --scenario_demo --extended_obs \
  --checkpoint "$CKPT" --scn_only "$ONLY" \
  --video --video_length "$VLEN" --headless > /tmp/scn.log 2>&1

echo "=== 시나리오 결과(성공/시간초과) ==="
grep -iE "Scenario.*(성공|시간초과)" /tmp/scn.log || echo "(stdout 가로채짐)"
echo "=== 타이밍 tsv ==="
cat logs/demo_videos/scenario_schedule.tsv 2>/dev/null || echo "(없음)"

if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -i logs/demo_videos/demo-step-0.mp4 -vframes 1 /tmp/scn_check.png >/dev/null 2>&1 && echo "[render-scn] 스샷: /tmp/scn_check.png"
fi

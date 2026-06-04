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

# 시나리오별로 따로 보존(덮어쓰기 방지). scn_only -1=all, 0~3=시나리오별.
OUT="logs/demo_videos/scn_${ONLY}.mp4"
[ "$ONLY" = "-1" ] && OUT="logs/demo_videos/scn_all.mp4"
cp -f logs/demo_videos/demo-step-0.mp4 "$OUT" 2>/dev/null && echo "[render-scn] 저장: $OUT"

echo "=== 시나리오 결과(성공/시간초과) ==="
grep -iE "Scenario.*(성공|시간초과)" /tmp/scn.log || echo "(stdout 가로채짐)"
echo "=== 타이밍 tsv ==="
cat logs/demo_videos/scenario_schedule.tsv 2>/dev/null || echo "(없음)"

# 편집 기준점: 각 회차(take) 시작 시점을 mm:ss로 저장(15fps). tsv는 다음 렌더에 덮어쓰이므로
#   scn별로 보존(scn_${ONLY}_cuts.txt). 끝 시점 = 다음 행 시작(마지막은 영상 끝).
CUTS="logs/demo_videos/scn_${ONLY}_cuts.txt"
awk -F'\t' 'NR>1{sec=$1/15; printf "%02d:%02d\tstep %d\t%s\n", int(sec/60), int(sec)%60, $1, $2}' \
  logs/demo_videos/scenario_schedule.tsv > "$CUTS" 2>/dev/null
echo "=== 편집 기준점(회차 시작 mm:ss) → $CUTS ==="
cat "$CUTS" 2>/dev/null || echo "(없음)"

if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -i logs/demo_videos/demo-step-0.mp4 -vframes 1 /tmp/scn_check.png >/dev/null 2>&1 && echo "[render-scn] 스샷: /tmp/scn_check.png"
fi

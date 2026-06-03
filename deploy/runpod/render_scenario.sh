#!/bin/bash
# 시나리오 쇼케이스 렌더 — S1/S2/S4/S5 순차 재생. 사용: bash deploy/runpod/render_scenario.sh [video_length]
# 결과: logs/demo_videos/demo-step-0.mp4 + 시나리오 타이밍 logs/demo_videos/scenario_schedule.tsv
set -e
cd /workspace/MARS

CKPT="logs/warehouse_mappo_extobs/model_shelf_final.pt"
VLEN="${1:-900}"   # 기본 900(~60s) — S1~S4 한 사이클 담김. 더 길게: 1500 등

echo "[render-scn] ckpt=$CKPT  video_length=$VLEN  (시나리오 4종 순차)"
python training/multi_robot/demo_play.py \
  --scenario_demo --extended_obs \
  --checkpoint "$CKPT" \
  --video --video_length "$VLEN" --headless > /tmp/scn.log 2>&1

echo "=== 시나리오 전환 로그 ==="
grep -iE "Scenario|시나리오|성공|시간초과" /tmp/scn.log || echo "(stdout 가로채짐 — tsv로 확인)"
echo "=== 시나리오 타이밍(자막용 tsv) ==="
cat logs/demo_videos/scenario_schedule.tsv 2>/dev/null || echo "(tsv 없음)"

if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -i logs/demo_videos/demo-step-0.mp4 -vframes 1 /tmp/scn_check.png >/dev/null 2>&1 && echo "[render-scn] 스샷: /tmp/scn_check.png"
fi

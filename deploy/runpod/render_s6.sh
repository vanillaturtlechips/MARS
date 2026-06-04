#!/bin/bash
# S6(동적 장애물 회피) 전용 렌더. scn_0~3과 달리 17D model_10998 + 장애물 활성화로 따로 뽑는다.
#   (시나리오 모델 model_shelf_final은 S6 ~32%, 17D model_10998은 ~98% → S6만 17D로 렌더)
# 사용:
#   bash deploy/runpod/render_s6.sh [video_length]
#   - video_length: 프레임수(=초×15). 기본 900(~60s)
# 예) bash deploy/runpod/render_s6.sh 900   # → logs/demo_videos/scn_4.mp4 (S6 반복재생)
set -e
cd /workspace/MARS

CKPT="logs/warehouse_mappo/model_10998.pt"   # 17D, S6 동적장애물 ~98%
VLEN="${1:-900}"

echo "[render-s6] ckpt=$CKPT  video_length=$VLEN  (17D + 동적장애물)"
# 주의: 17D 모델이므로 --extended_obs 넣지 않음(19D 입력층과 불일치). --enable_obstacles 필수.
python training/multi_robot/demo_play.py \
  --scenario_demo --enable_obstacles \
  --checkpoint "$CKPT" --scn_only 4 \
  --video --video_length "$VLEN" --headless > /tmp/s6.log 2>&1

# scn_only 4 = S6. scn_0~3과 같은 명명 규칙으로 보존(덮어쓰기 방지).
OUT="logs/demo_videos/scn_4.mp4"
cp -f logs/demo_videos/demo-step-0.mp4 "$OUT" 2>/dev/null && echo "[render-s6] 저장: $OUT"

echo "=== S6 결과(성공/시간초과) ==="
grep -iE "Scenario.*(성공|시간초과)" /tmp/s6.log || echo "(stdout 가로채짐)"
echo "=== 장애물 배치 로그 ==="
grep -iE "\[S6\]" /tmp/s6.log | head -5 || echo "(없음 — 장애물 미배치?)"
echo "=== 타이밍 tsv ==="
cat logs/demo_videos/scenario_schedule.tsv 2>/dev/null || echo "(없음)"

# 편집 기준점: 각 회차(take) 시작 시점을 mm:ss로 저장(15fps). tsv는 다음 렌더에 덮어쓰이므로
#   scn별로 보존. 끝 시점 = 다음 행 시작(마지막은 영상 끝).
CUTS="logs/demo_videos/scn_4_cuts.txt"
awk -F'\t' 'NR>1{sec=$1/15; printf "%02d:%02d\tstep %d\t%s\n", int(sec/60), int(sec)%60, $1, $2}' \
  logs/demo_videos/scenario_schedule.tsv > "$CUTS" 2>/dev/null
echo "=== 편집 기준점(회차 시작 mm:ss) → $CUTS ==="
cat "$CUTS" 2>/dev/null || echo "(없음)"

if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -i logs/demo_videos/demo-step-0.mp4 -vframes 1 /tmp/s6_check.png >/dev/null 2>&1 && echo "[render-s6] 스샷: /tmp/s6_check.png"
fi

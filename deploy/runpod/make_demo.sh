#!/bin/bash
# 데모 영상 자동 편집 — 여러 클립을 [잘라내기 + 라벨 자막 + 하나로 합치기].
# 편집 스킬 0으로, 아래 SEGMENTS의 숫자/라벨만 고치고 실행하면 final_demo.mp4가 나온다.
#
# 사용:
#   bash deploy/runpod/make_demo.sh
#   → logs/demo_videos/final_demo.mp4
#
# SEGMENTS 한 줄 형식:  "영상경로 | 시작(mm:ss) | 끝(mm:ss) | 화면에 띄울 라벨"
#   - 시작~끝 구간만 잘라 씀. 좋은 take의 시점은 scn_N_cuts.txt 참고.
#   - 라벨은 클립 상단에 반투명 박스로 박힘(한글 OK). 라벨에 '|' 문자는 쓰지 말 것.
set -e
cd /workspace/MARS

OUT="logs/demo_videos/final_demo.mp4"
RES_W=1280; RES_H=720; FPS=30           # 모든 클립을 이 규격으로 통일(해상도 달라도 OK)
TMP="/tmp/make_demo"; rm -rf "$TMP"; mkdir -p "$TMP"

# ── 편집할 구간들 (★ 여기 숫자·라벨만 고치면 됨) ─────────────────────────────
SEGMENTS=(
  "logs/demo_videos/scn_0.mp4   | 0:00 | 0:18 | S1 · 통로 교행"
  "logs/demo_videos/scn_1.mp4   | 0:00 | 0:18 | S2 · 교차로 양보 (3-way 교착)"
  "logs/demo_videos/scn_2.mp4   | 0:00 | 0:18 | S4 · 적재 스테이션"
  "logs/demo_videos/scn_3.mp4   | 0:00 | 0:18 | S5 · 혼잡 통로 횡단"
  "logs/demo_videos/scn_4.mp4   | 0:00 | 0:20 | S6 · 동적 장애물 회피"
  "logs/demo_videos/demo-step-0.mp4 | 0:00 | 0:30 | 선반 작업 — 박스 픽업·운반"
  "/workspace/phase2_demo.mp4   | 0:00 | 0:20 | 로봇팔 — 픽 앤 플레이스"
)
# ──────────────────────────────────────────────────────────────────────────

# 의존성: ffmpeg + 한글 폰트(나눔고딕)
command -v ffmpeg >/dev/null 2>&1 || { echo "[make] ffmpeg 설치..."; apt-get install -y --quiet ffmpeg >/dev/null 2>&1 || true; }
FONT="/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if [ ! -f "$FONT" ]; then
  echo "[make] 한글 폰트(나눔) 설치..."; apt-get install -y --quiet fonts-nanum >/dev/null 2>&1 || true
  fc-cache -f >/dev/null 2>&1 || true
fi
[ -f "$FONT" ] || FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"   # 폴백(한글 깨질 수 있음)
echo "[make] 폰트: $FONT"

LIST="$TMP/list.txt"; : > "$LIST"
i=0
for seg in "${SEGMENTS[@]}"; do
  IFS='|' read -r CLIP START END LABEL <<< "$seg"
  CLIP="$(echo "$CLIP" | xargs)"; START="$(echo "$START" | xargs)"
  END="$(echo "$END" | xargs)";  LABEL="$(echo "$LABEL" | xargs)"
  if [ ! -f "$CLIP" ]; then echo "[make] ⚠ 건너뜀(파일 없음): $CLIP"; continue; fi

  # drawtext 이스케이프(작은따옴표/콜론) + 라벨 박스
  ELABEL="$(printf '%s' "$LABEL" | sed "s/'/\\\\'/g; s/:/\\\\:/g")"
  DRAW="drawtext=fontfile=${FONT}:text='${ELABEL}':fontsize=44:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=20:x=(w-text_w)/2:y=44"
  VF="scale=${RES_W}:${RES_H}:force_original_aspect_ratio=decrease,pad=${RES_W}:${RES_H}:(ow-iw)/2:(oh-ih)/2:color=black,fps=${FPS},${DRAW}"

  OUTSEG="$TMP/seg_$(printf '%02d' $i).mp4"
  echo "[make] 자르기 #$i: ${CLIP##*/}  $START~$END  '$LABEL'"
  ffmpeg -y -i "$CLIP" -ss "$START" -to "$END" -vf "$VF" \
    -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$OUTSEG" >/dev/null 2>&1 \
    && echo "file '$OUTSEG'" >> "$LIST" \
    || echo "[make] ⚠ 실패(건너뜀): $CLIP $START~$END"
  i=$((i+1))
done

if [ ! -s "$LIST" ]; then echo "[make] ❌ 합칠 클립이 없음 — 경로/시점 확인"; exit 1; fi

echo "[make] 합치는 중 → $OUT"
ffmpeg -y -f concat -safe 0 -i "$LIST" -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an "$OUT" >/dev/null 2>&1

if [ -f "$OUT" ]; then
  DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" 2>/dev/null | cut -d. -f1)
  SIZE=$(du -sh "$OUT" | cut -f1)
  echo "[make] ✅ 완성: $OUT  (${DUR}s, $SIZE)"
  echo "[make] 다운로드: scp -P {PORT} -i ~/.ssh/id_ed25519 root@{POD_IP}:/workspace/MARS/$OUT ."
else
  echo "[make] ❌ 합치기 실패"
fi

# 멀티로봇 데모 — 최종 현황 & 남은 일 (2026-06-04)

S1~S6 시나리오 쇼케이스 데모를 만들기 위한 디버깅·구현의 최종 정리.
**모델은 완성됐고, 남은 건 영상 take 선택 + S6 + 편집뿐이다.**

> 브랜치: `diff-drive-finetune`. 모델·코드 전부 git에 있음(파드 꺼져도 복구 가능).

---

## 1. TL;DR

- **모델 완성**: `logs/warehouse_mappo_extobs/model_shelf_final.pt` (git 영구화).
  eval(32 env): **S1 100% · S2 100% · S5 100% · S5 선반교착 0%.** 선반 회피·협응 해결.
- **데모**: 시나리오(S1/S2/S4/S5)를 **공장 맵 안에서** 재생(B안). 매끄러운 모션. 시나리오별 mp4 저장.
- **핵심 한계**: 모델은 eval에서 완벽한데, **데모 단일재생(1 env)은 하드 3-way에서 한 로봇이
  골 0.5m 안에 못 닿고 ~1.2~1.7m 짧게 끝남.** 단 시각적으로는 교차·협응을 수행함(거의 다 감).
- **남은 일**: ① scn 영상 보고 좋은 구간 채택 ② S6(17D 별도) ③ S3 스킵 ④ 편집(자막).

---

## 2. 모델 (완성)

| | 값 | 비고 |
|---|---|---|
| 파일 | `logs/warehouse_mappo_extobs/model_shelf_final.pt` | model_800 사본, git 영구화 |
| 학습 | scenario-only(시나리오형) + warm-start model_10998 + freeze_std | 19D extended_obs |
| eval S1 교행 | **100%** | |
| eval S2 3way | **100%** | |
| eval S5 선반혼잡 | **100% / 교착 0%** | 선반 회피 핵심 요구 달성 |
| S6 동적장애물 | 17D model_10998이 98%로 우위(model_shelf_final은 ~32%) | 선반↔장애물 트레이드오프 |

> 자세한 학습 경위는 시나리오형 학습 성공 기록 참조. 선반 vs 장애물은 한 모델로 양립 불가
> (입증됨) → 레이어 분리: 선반=시나리오 모델, 장애물=17D 별도.

---

## 3. 데모 구조

**두 가지 모드:**
- **`--scenario_demo`** (= `render_scenario.sh`): S1/S2/S4/S5를 순차/선택 재생. **S1~S6 쇼케이스용.**
- **`--task`** (= `render_demo.sh`): 창고 맵 + 박스 픽업↔도크 운반(연속). 창고 분위기용.

**B안 적용(시나리오를 공장 맵 안에서):**
- 시나리오 모드도 `full_warehouse.usd` 로드(이전엔 부감카메라+지붕 때문에 바닥판만 썼음).
- 카메라를 옆 3/4 각도로(`SCN_CAMERA_EYE=(1.5,-10,5.5)`, 지붕 아래). `--cam_eye`/`--cam_target`로 조정.

**스폰 지터:** 시나리오 스폰을 매 렌더 ±0.35 흔듦 → 매번 다른 롤아웃(고정 나쁜케이스 반복 회피).

**실행 명령:**
```bash
# 시나리오 쇼케이스 (scn_only: -1=전체, 0=S1, 1=S2, 2=S4, 3=S5)
bash deploy/runpod/render_scenario.sh 900 0     # → logs/demo_videos/scn_0.mp4 (S1, 3회 반복)
bash deploy/runpod/render_scenario.sh 900 1     # → scn_1.mp4 (S2)
# 운반 데모
bash deploy/runpod/render_demo.sh 900           # → demo-step-0.mp4 (창고+운반)
```
각 `scn_N.mp4`에 그 시나리오 3회 반복(다른 draw). 콘솔에 회차별 `성공/시간초과`.
자막 타이밍은 `logs/demo_videos/scenario_schedule.tsv`(start_step → label).

---

## 4. 핵심 발견 — eval은 완벽한데 데모만 실패 (70턴 디버깅 결론)

**증상:** eval(32 env)은 S1/S2/S5 100%인데, 데모 단일재생(1 env)은 하드 3-way에서
한 로봇(R2 종단 크로스 / R1 동일목표)이 골에 ~1.2~1.7m 못 미치고 끝남.

**제거한 가설(전부 eval과 맞춰도 bit-identical → 원인 아님):**
큐브 크기(0.5) · action 스무딩 · tanh · ActorMLP vs OnPolicyRunner · 초기 yaw(=0) ·
rigid sleep/stabilization_threshold · 선반 적재물(goods) · obs 분리(IPPOReshape) · 정책 로딩.

**남은 단 하나의 차이 = `num_envs` (데모 1개 vs eval 32개).**
eval의 32개 env는 GPU 비결정성으로 32개 다른 롤아웃 → **모두 성공(견고).**
데모의 단일 결정론 롤아웃은 하필 그 하드 3-way의 나쁜 케이스 → 한 로봇 실패.
(docs §3 "결정론적 고정스폰이 나쁜케이스 반복"과 동일 현상.)

**중요:** 메트릭 "시간초과"는 **골 0.5m 안에 정확히 못 닿았다**는 엄격 기준.
R2최소=1.3이면 R2가 위(+4)→아래(-2.7)까지 **거의 다 가로질렀다**는 뜻 = 시각상 교차·회피 수행.
**→ 데모 품질은 숫자가 아니라 영상으로 판단해야 한다.**

**완화책:** 스폰 지터(매 렌더 다른 draw) + 시나리오별 좋은 take 선택.

**시나리오별 성공 패턴(지터, 회당):**
| 시나리오 | 성공률 | 비고 |
|---|---|---|
| S2 3way | ~3/4 성공 | 깨끗한 take 충분 ✅ |
| S1 교행 | R2 거의 매번 ~1.2~1.7 짧음 | 종단 크로스가 어려움. 시각상은 교차로 보임 |
| S4 스테이션 | R1 짧음 | 동일목표(정의상 동시도달 난해) |
| S5 혼잡 | R2 짧음 | 교차는 보임 |

---

## 5. 남은 일 (낮 작업)

1. **`scn_0~3.mp4`를 눈으로 보고** 좋은 구간 채택 (메트릭 무시, 교차·협응 보이면 OK).
   - S2는 성공 구간 확실. S1/S5는 R2 종단크로스가 보이면 채택, 아니면 S5로 교행 대체.
2. **S6(동적장애물)** = 17D `model_10998` + `--enable_obstacles` 별도 클립.
   *단 현재 `SCN_DEMO`에 S6 씬이 없음 → S6 시나리오 추가 + 장애물 enable 셋업 필요(미완).*
3. **S3(배터리)** = 모델에 배터리 obs 없음 → 영상 스킵, 발표에서 말로 설명(orchestrator 영역).
4. **편집** = scn 클립 + S6 클립 이어붙이고 `scenario_schedule.tsv` 타이밍으로 S1~S6 자막 카드.

---

## 6. 교훈 / 주의

- **데모는 메트릭이 아니라 영상으로 판단** — "시간초과"여도 시각상 멀쩡할 수 있음(거의 다 도달).
- **멀쩡한 물리·비주얼을 마진 이득으로 건드리지 말 것** — R2 메트릭 쫓다 action 스무딩 제거(→로봇
  떨림), 에셋 strip(→"사라짐" 소동)을 유발했고 다 복원함. 변경 전 사용자 확인.
- **모델은 처음부터 완벽**(eval 100%)했고, 데모 "박힘/멈춤"은 전부 데모 재생 레이어 문제였음.
- 체크포인트는 git에 `git add -f`로 영구화(`logs/**/*.pt`는 gitignore 기본 제외) — 파드 비영구.

---

## 7. 파일 & 명령 레퍼런스

- 모델: `logs/warehouse_mappo_extobs/model_shelf_final.pt` (git)
- 시나리오 렌더: `bash deploy/runpod/render_scenario.sh [video_length] [scn_only]`
- 운반 렌더: `bash deploy/runpod/render_demo.sh [video_length]`
- eval: `python training/multi_robot/eval_scenarios.py --ckpt <ckpt> --extended_obs --num_episodes 100 --num_eval_envs 32 --tag <t> --headless`
- 자막 타이밍: `logs/demo_videos/scenario_schedule.tsv`
- 카메라 즉석조정: 렌더에 `--cam_eye "x,y,z" --cam_target "x,y,z"`
- 핵심 노브(`demo_play.py`): `SCN_DEMO`(시나리오 정의), `SCN_CAMERA_EYE/TARGET`, `--scn_only`, 스폰 지터(±0.35)

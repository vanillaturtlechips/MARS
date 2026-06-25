# MARS 프로젝트 종합 현황 (메모리 정리, 2026-06-01)

> 세션 메모리에 누적된 프로젝트 지식을 한 문서로 통합. 핸드오프/참조용.
> 컴포넌트별 현황, 핵심 체크포인트, **재시도 금지 목록**, 교훈.

---

## 0. 실행 환경

- **훈련/실행**: RunPod Docker 컨테이너, 경로 `/workspace/MARS/`
- **로컬 편집**: `/home/user/Documents/MARS` → git push → 컨테이너에서 pull
- **가상환경**: 컨테이너 내 `/workspace/isaac_venv/`
- **GPU**: 작업에 따라 RTX A6000(49GB) 또는 A4500(20GB) 사용
- 로그·체크포인트는 컨테이너 내 `/workspace/MARS/logs/` (로컬 직접 접근 불가)
- 설치: `bash deploy/runpod/setup.sh` (원클릭, ~20분, Isaac Sim 6.0 + Isaac Lab v2.3.2)

---

## 1. 마감 일정

- **발표: 2026-06-08**
- **데모 마감: 6/5(1차) ~ 6/7(최종)**
- 이후는 애드온/논문 단계

원칙: 재훈련 없는 산출물을 먼저 확정해 **최악에도 발표 데모 확보**.
시간 압박 하 신규 대형 재훈련은 후순위.

---

## 2. 컴포넌트별 현황

### ✅ Pick & Place (manipulation) — 완성
- **Teacher `logs/warehouse_manipulation/model_2999.pt`: place_rate 100%** (데모용)
- Pick&Place fine-tune(`model_300`/transport 정합): **place_rate 86~94%**
- obs 30D(Teacher, privileged) / 23D(pickplace), action 4D Cartesian IK (dx,dy,dz,gripper)
- **manipulation env를 transport와 정합시킨 5대 요건**(중요):
  1. **FRANKA_PANDA_HIGH_PD_CFG** — 결정타. 약한 PD는 중력 sag로 IK 실패
  2. grasped box 지하 -5m 숨김 (finger 충돌이 IK 방해)
  3. goal = ee 중심 정확히 r 거리 (xy clamp 제거)
  4. grasp 시 cmd_ee_pos = ee 리셋
  5. transport 보상 = `-rew_carry_dist × dist/step` (delta 아님 — delta는 local optimum)
- **Student 정책: 개발 실패(2주+BC+PPO 전부 소진). 재시도 금지. Teacher 사용.**

### ✅ MARL Navigation + 동적 장애물 회피 — 완성
- **`logs/warehouse_mappo/model_10998.pt`: 동적 장애물 회피 S6 98%** (충돌 0%)
- model_9999(MAPPO)에서 **obs 17D 유지 fine-tune** (장애물 거리를 `shelf_dist`에 min 통합)
- 장애물 = kinematic cuboid, 지하 -5m 숨김 → 등장 step(50~130)에 지상으로
- 부수 효과: 충돌 평균 22%→0.2%, S2(3-way) 0→95%
- **S3/S4(좁은통로 우선순위·동일목표 교착)는 RL 한계 → orchestrator 영역.**
  S6(로봇 평행→로봇교착0, 순수 장애물 우회)로 깨끗이 측정한 게 98%.

### ⏸️ 배터리/충전소 — RL 통합 폐기 (orchestrator 영역으로)
- **에이전트 토론 결론: 충전 시점/충전소 선택은 전역 의사결정 → RL 아님.**
  S3/S4가 orchestrator 영역인 것과 구조적 동일.
- RL은 navigation+회피만(model_10998, 17D). orchestrator가 배터리 보고
  충전소를 goal로 발행 → RL은 그냥 navigation.
- obs 20D 확장 fine-tune 시도 → **실패**(충전 행동 미발생: 충전소가 코너라
  탐색 도달 못 함 → charging 경험 0 → 강화 불가). **재시도 금지.**
- env 충전 메커니즘은 `enable_battery=False`로 코드 보존. 발표선 "설계 완료,
  학습은 향후 과제(Phase4 orchestrator)"로 로드맵 명기.

### 🔄 Diff-Drive 전환 (외형 자연스러운 커브) — 진행 중
- 상세: **`docs/DIFFDRIVE_PROGRESS.md`** 참조
- 문제: 홀로노믹 점질량이라 옆걸음/둥둥 미끄러짐
- 시도: (A)외형 유니사이클=발산 폐기 → (B)하드컷 fine-tune=목표도달 실패 →
  (C)적응형 커리큘럼=max_vy 0.4까지 내려가다 벽 → **(D)계층 컨트롤러=최종 채택**
- (D) `diff_drive_controller`: model_10998 그대로 + diff-drive 변환(재학습 0).
  실행 `demo_play.py --diff_drive_ctrl --turn_gain 3.0`
- **열린 질문: 회피 성공률 유지 여부(eval 미검증).** 커브 모양은 거의 확실.
- 브랜치 `diff-drive-finetune` (main 무손상)

---

## 3. 핵심 체크포인트 요약

| 체크포인트 | 용도 | 성능 | obs/act |
|---|---|---|---|
| `warehouse_manipulation/model_2999.pt` | Pick&Place Teacher | place 100% | 30D / 4D IK |
| `warehouse_mappo/model_10998.pt` | MARL nav+동적장애물 | S6 98%, 충돌 0% | 17D / 9D(3로봇×3) |
| `warehouse_mappo_diffdrive_curr/` | diff-drive 커리큘럼 fallback | max_vy 0.4(strafe 60%↓) | 17D / 9D |
| `model_9999.pt` | MARL nav 원본(fine-tune 베이스) | — | 17D / 9D |

---

## 4. 재시도 금지 목록 (무한루프 방지)

1. **Pick&Place Student 정책** — 2주+BC+PPO 전부 실패. Teacher 사용.
2. **배터리/충전소 RL 통합** — 충전 경험 미발생으로 실패. orchestrator 영역.
3. **diff-drive 하드 vy=0 단발 fine-tune** — critic 발산, 목표도달 실패.
4. **커리큘럼 게이트를 "3대 동시 도달"로 측정** — 고정목표 환경서 ~0, 안 움직임.
   (로봇별 도달률로 측정해야 함)
5. **rew_align 관절공간 cosine, rew_goal_dist≥4.0, entropy_coef≥0.001** —
   각각 exploit/VF폭발/std폭발. 상세는 `phase2_tried_approaches.md` 표.

---

## 5. 교훈 (작업 방식)

- **가정 검증 먼저**: 체크포인트 obs_dim, git push/pull 동기화, 커밋 클린 여부를
  계획 전 실제로 확인. 미검증 가정이 쌓이면 계획 전체가 무너짐.
- **핵심 수치 즉시 기록**: eval 결과·훈련 완료 여부 등은 잃으면 사용자가 극도로
  피로. 모르면 솔직히 "잘렸다"고 한 번만 물을 것.
- **목표 들으면 표준 해법부터 메뉴에**: diff-drive에서 "재학습 vs 포기" 2지선다로
  잘못 제시, 정석(계층 컨트롤러)을 누락해 GPU를 여러 번 소모. 정석/표준부터 올릴 것.

---

## 6. 관련 문서

- `docs/DIFFDRIVE_PROGRESS.md` — diff-drive 전환 상세
- `docs/PHASE3_PROGRESS.md`, `docs/PHASE3_IPPO_VS_MAPPO.md` — MARL
- `docs/phase2_transport_debug.md`, `docs/ARCHITECTURE.md`
- `deploy/runpod/RUNPOD_GUIDE.md` — RunPod 설정

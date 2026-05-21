# Phase 2 Transport 학습 불능 — 진단 및 수정 기록 (2026-05-21 최종)

## 최종 환경 설정 (현재 코드 기준)

| 파라미터 | 값 | 이유 |
|---------|-----|------|
| `grasp_dist_threshold` | 999.0 | 즉시 grasp — transport만 학습 |
| `place_dist_threshold` | 0.12m | |
| `rew_transport_dst` | 15.0 | -15×dist/step — 순수 거리 패널티 |
| `rew_align` | 3.0 | cos_sim(action, goal_dir)/step |
| `rew_place` | 100.0 | 소형 터미널 보너스 |
| `rew_time` | -0.02 | |
| 기타 rew_* | 0.0 | 모두 비활성화 |
| Teacher/Student | 동일 | Student 구분 폐기 |

---

## 이번 세션(2026-05-21) 발견·수정된 버그 목록

### 버그 1 — obs[0:3] 부호 오류 (치명적)

**문제**: `transport_rel = box_pos_carried - goal_pos` (반대 방향)
- goal이 오른쪽인데 obs는 왼쪽을 가리킴
- policy가 obs 방향으로 이동 → goal에서 멀어짐

**수정**: `transport_rel = goal_pos - box_pos_carried` (목표까지 잔여 벡터)

---

### 버그 2 — 물리 bounce (치명적)

**문제**: EE home z=0.487m, 테이블 z=0.5m → 박스가 테이블에 박힘
- grasp 순간 PhysX impulse → 박스가 0.345m → 0.746m 순간 도약
- `dist_box_goal` 3 step 만에 0.35m → 0.75m로 폭발

**수정**: `_grasp_ee_offset[:, 2] = 0.06m` — 박스를 EE보다 6cm 위에 운반

---

### 버그 3 — home 로컬 옵티멈 (치명적)

**문제**: `goal_prox = exp(-dist×1) × 5.0` + potential shaping 조합
- EE home(dist=0.296m): exp(-0.296)×5.0 = 3.71/step × 239step = 886
- goal(dist=0.12m): exp(-0.12)×5.0 = 4.44/step × N + place보너스
- home이 "가성비 최고" → transport 안 함

**수정**: `rew_transport = 0`, `rew_goal_prox = 0` 모두 제거
- `-dist × 15.0/step` 패널티만 남김 → home(dist=0.296m, -4.44/step)도 패널티
- goal(dist=0)만이 최적 → 단조로운 gradient

---

### 버그 4 — surrogate_loss ≈ 0 (근본 원인)

**문제**: DLS IK 효율 약 7% (명령 3cm → 실제 이동 ~2mm/step)
- 올바른 action의 보상: 15 × 0.002 = 0.03/step
- noise(std=0.5) 대비 SNR 극히 낮음 → advantage ≈ 0 → gradient 없음
- surrogate_loss 평균 -0.0001로 flatline, 250+ iter 동안 dist_box_goal 불변

**수정**: action-goal 정렬 보상 추가
```python
goal_dir = goal_pos_w - box_pos_carried   # grasped 환경의 목표 방향
cos_sim  = dot(action[:3], goal_dir) / (|action[:3]| × |goal_dir|)
alignment = cos_sim × grasped × 3.0      # +3/step (완전 정렬), -3/step (반대)
```
- IK 실제 이동량과 무관하게 action 방향에 즉각 gradient 제공
- PPO가 "좋은 action vs 나쁜 action"을 즉시 구별 가능

---

### 버그 5 — Joint Control entropy trap

**경위**: transport 수렴 실패 원인을 잘못 진단 → joint space control(act=8) 실험
- `delta_q = action[:7] × 0.05` 적용
- 결과: noise_std 0.54 → 0.90 (100 iter 만에 entropy 폭발)
- credit assignment 불가: 7DOF 관절 조합 → EE 이동 관계 학습 불가

**수정**: Cartesian IK로 복귀, entropy_coef 0.01 → 0.001

---

### 버그 6 — rew_place 과대 설정

**문제**: `rew_place = 800` → VF loss 137,921, advantage 분산 폭발
- 800 보너스 vs 15×dist 패널티(평균 -12/step × 240 = -2880) → 상대 규모 불균형

**수정**: `rew_place = 100.0`

---

### 버그 7 — inference.py act_dim 오류

**문제**: joint control 실험 후 `act_dim=8` 남아있었음
- deploy 단계에서 shape mismatch 유발

**수정**: `act_dim=4` (Cartesian [dx, dy, dz, gripper])

---

### 버그 8 — MARL stationary 패널티 (warehouse_marl_env.py)

**문제**: goal 도달 로봇도 `speed < 0.1` 판정 → -0.3/step 패널티
- 일찍 goal에 도착할수록 더 많이 패널티 → anti-incentive

**수정**: `not_at_goal = (dist > goal_radius).float()` 마스킹

---

## 악순환 메커니즘 (최종 수정 전 상태)

```
IK 효율 2mm/step
  → 올바른 action 보상 0.03/step (noise에 묻힘)
  → advantage ≈ 0
  → surrogate_loss ≈ 0 → PPO 업데이트 없음
  → home에서 goal_prox (제거 전) 수집에 안주
  → dist_box_goal 0.78~0.80m 고착
  → 영원히 반복
```

---

## 수정 후 기대 동작

- iter 20-30: `surrogate_loss` > 0.001 (첫 gradient 발생)
- iter 50+: `dist_box_goal` 0.75m 이하로 감소 시작
- iter 300+: `place_rate` 상승 시작

---

## 훈련 명령

```bash
# RunPod (A5000, 24GB VRAM)
git pull
python training/single_robot/train_manipulation.py \
  --num_envs 5096 --max_iter 3000 --lr 1e-3 --headless
```

---

## 주의사항

- obs dim 변경(30 → 31) 이후 체크포인트 재사용 불가
- Student/Teacher 구분 폐기 — `WarehouseManipulationStudentEnvCfg`는 alias만 유지
- `grasp_dist_threshold=999` → 에피소드 시작 즉시 grasp, approach phase 없음

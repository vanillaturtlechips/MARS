# Phase 2 Transport 학습 불능 — 진단 및 수정 기록

## 증상 (commit 0914d1e 이전)

- `grasp_rate`: 98.5% (정상)
- `dist_box_goal`: 0.72m → **140 iter 동안 전혀 감소하지 않음**
- `surrogate_loss`: ≈ 0 반복 (PPO gradient 없음)
- `place_rate`: 1.5% 수준에서 감소 중

---

## 근본 원인 3가지

### 버그 1 — obs[0:3]이 grasped 후 상수 (치명적)

**문제**: grasped 후 `obs[0:3] = box_pos - ee_pos = grasp_ee_offset = 상수`
- 정책이 600 step 내내 goal이 어느 방향인지 정보를 전혀 받지 못함
- approach phase와 transport phase에서 동일한 입력 → 동일한 출력
- 두 phase의 reward 구조가 완전히 다름 → gradient 서로 상쇄

**수정**: `box_or_goal_rel` 도입
```
not grasped: obs[0:3] = box_pos - ee_pos        (approach 방향)
    grasped: obs[0:3] = box_pos_carried - goal_pos  (transport 방향, goal까지 잔여 벡터)
```
+ `grasped(1)` 플래그를 obs 마지막에 추가 → 정책이 phase를 명시적으로 구분

obs dim: 30 → **31**

---

### 버그 2 — `_prev_dist_box_goal = 999.0` 초기화 (치명적)

**문제**: reset 시 999.0으로 초기화 → 첫 grasp 순간 `delta = 999 - 0.35 = 998.65`
- transport reward spike: `50 × 998.65 = 49,932` (clamp 없을 경우)
- 기존 코드는 `clamp(-0.1, 0.1)`로 우연히 방지했으나, clamp 제거 후 폭발 가능
- VF loss가 400~1600 수준으로 높았던 원인 중 하나

**수정**:
- reset 시: 실제 `(box_pos - goal_pos).norm()` 으로 초기화
- grasp 첫 step: `_prev_dist_box_goal[new_ids] = dist_box_goal[new_ids]` 동기화

---

### 버그 3 — reward scale 불균형 (높음)

| 항목 | 기존 | 수정 | 이유 |
|------|------|------|------|
| `rew_transport` | `10 × delta × 100` | `50 × delta` | 3cm 이동 시 +1.5/step, clamp 제거 |
| `rew_goal_prox` scale | `exp(-5×d)` | `exp(-3×d)` | dist=0.35m에서 0.17→0.35, 먼 거리 gradient 확보 |
| `rew_goal_prox` coef | 1.0 | 5.0 | 절대 규모 5× 증가 |
| `rew_transport_dst` | 0.5 | 3.0 | 6×, dense VF 학습 신호 강화 |

---

## 악순환 메커니즘 (수정 전)

```
transport reward E[A] ≈ 0 (obs에 goal 방향 없음)
    → surrogate_loss ≈ 0 → gradient 없음
    → noise_std 상승 (1.0 → 1.12)
    → EE random walk 심화
    → dist_box_goal 증가
    → goal_prox(exp-5) 더욱 작아짐
    → 반복
```

---

## 수정 후 변경 파일

- `envs/warehouse/warehouse_manipulation_env.py`
  - OBS_DIM: 30 → 31
  - `_get_observations()`: `box_or_goal_rel` + `grasped` 플래그
  - `_get_rewards()`: transport clamp 제거, goal_prox scale 변경, transport_dst 강화
  - `_reset_idx()`: `_prev_dist_box_goal` 실제 거리로 초기화

- `training/single_robot/train_manipulation.py`
  - `empirical_normalization`: False → True
  - `entropy_coef`: 0.001 → 0.01

---

## 주의사항

obs dim 변경(30→31)으로 **이전 checkpoint 재사용 불가**. 새 훈련으로 시작해야 함.

```bash
python training/single_robot/train_manipulation.py \
  --num_envs 5096 --max_iter 1000 --lr 1e-3 --headless
```

**확인 지표 (iter 10~20)**:
- `dist_box_goal` 감소 시작 → transport 학습 중
- `surrogate_loss` 절댓값 > 0.001 → gradient 발생
- `place_rate` 상승 → 수렴 중

# Phase 1 & Phase 1.5 진행 결과 (2026-05-15)

## 환경 스펙

| 항목 | 값 |
|------|-----|
| 시뮬레이터 | Isaac Lab 2.3.2 / Isaac Sim 5.1.0 |
| GPU | RTX 2070 Mobile |
| 알고리즘 | PPO (rsl_rl OnPolicyRunner) |
| 로봇 모델 | Cuboid rigid body (0.5×0.4×0.3 m), 홀로노믹 직접 속도 제어 |
| 환경 수 | 1024 (훈련) / 256 (검증) |

---

## Phase 1 — 장애물 없는 창고 네비게이션

### 구현

- **환경**: `envs/warehouse/warehouse_env.py` — `DirectRLEnv` 서브클래스
- **관측 공간** (6차원): `[goal_x_body, goal_y_body, goal_dist, vx_body, vy_body, omega_z]`
- **행동 공간** (3차원): `[cmd_vx, cmd_vy, cmd_omega]` (body frame, [-1,1] 정규화)
- **보상**:
  - `rew_dist = -0.3 × dist_to_goal` (매 스텝)
  - `rew_goal = +10.0` (골 도달 시)
  - `rew_time = -0.001` (매 스텝 시간 패널티)
- **에피소드**: 20초 (300 스텝 @ 15Hz), 골 반경 0.35m, 골 범위 1~4m

### 훈련 설정

| 항목 | 값 |
|------|-----|
| 네트워크 | MLP [256, 128, 64], ELU |
| lr | 3e-4 (adaptive schedule, desired_kl=0.01) |
| iterations | 1000 |
| 체크포인트 | `logs/warehouse_nav/model_999.pt` |

### 검증 결과

| 지표 | 값 |
|------|-----|
| 골 도달률 | **100%** |
| 평균 에피소드 길이 | 24.9 스텝 |
| 총 검증 에피소드 | 5120 |

---

## Phase 1.5 — 선반 장애물 창고 네비게이션

### 구현

- **환경**: `envs/warehouse/warehouse_obstacle_env.py` — `WarehouseNavEnv` 상속
- **선반 레이아웃**: 4개 (2행×2열), 중앙 십자 통로 확보

```
  [=Shelf 0=]   [=Shelf 1=]   ← y = +2.5
  ←── 메인 통로 ──→
  [=Shelf 2=]   [=Shelf 3=]   ← y = -2.5
    x = -2.0        x = +2.0
```

- **관측 공간** (7차원): Phase 1 6차원 + `min_obstacle_dist` (AABB 최소 거리)
- **추가 보상**:
  - `rew_prox_warn = -0.1` (선반까지 거리 < 0.8m)
  - `rew_prox_crit = -1.0` (선반까지 거리 < 0.4m)
- **골 샘플링**: rejection sampling으로 선반 내부에 골 생성 방지

### 훈련 이슈 및 해결 과정

| 이슈 | 원인 | 해결 |
|------|------|------|
| VF loss 폭발 (385→1952) | `rew_prox_crit=-5.0`이 너무 커서 보상 분포 이동 | 패널티 -0.1/-1.0으로 축소, 처음부터 재훈련 |
| noise std 폭주 (1.4→2.75) | adaptive schedule이 KL 낮을 때 lr을 무한 증가 | `desired_kl=0.05` 시도 → 여전히 폭주 |
| noise std 고착 (1.02) | `fixed schedule + lr=1e-4` 너무 작아 gradient 없음 | 최적 체크포인트(iter=100) 선택으로 우회 |

### 최종 체크포인트 선택 근거

adaptive schedule로 훈련 시 iter 100 근방(noise std ≈ 0.92)에서 정책이 가장 안정적임을 확인.  
이후 lr 폭주로 noise std가 계속 증가하므로 iter=100 체크포인트를 Phase 1.5 기준 모델로 채택.

| 항목 | 값 |
|------|-----|
| 체크포인트 | `logs/warehouse_obstacle_nav/model_100.pt` |
| 훈련 시점 noise std | 0.92 |

### 검증 결과

| 지표 | Phase 1 | Phase 1.5 |
|------|---------|-----------|
| 골 도달률 | 100% | **94.4%** |
| 평균 에피소드 길이 | 24.9 스텝 | 41.2 스텝 |
| 총 검증 에피소드 | 5120 | 13544 |

에피소드 길이 증가(25→41 스텝)는 선반 우회 경로 때문으로 정상.  
실패 5.6%는 선반 모서리에 끼거나 좁은 통로에서 타임아웃.

---

## 주요 파일

```
envs/warehouse/
  warehouse_env.py              # Phase 1 환경
  warehouse_obstacle_env.py     # Phase 1.5 환경 (선반 추가)
  agents/rsl_rl_ppo_cfg.py      # PPO 설정 (Phase 1 / Phase 1.5 분리)

training/single_robot/
  train_navigation.py           # Phase 1 훈련
  train_navigation_obstacle.py  # Phase 1.5 훈련
  test_navigation_headless.py   # Phase 1 헤드리스 검증
  test_obstacle_headless.py     # Phase 1.5 헤드리스 검증
  view_navigation.py            # Phase 1 GUI 뷰어
  view_obstacle_navigation.py   # Phase 1.5 GUI 뷰어

logs/
  warehouse_nav/model_999.pt          # Phase 1 최종 모델
  warehouse_obstacle_nav/model_100.pt # Phase 1.5 기준 모델
```

---

## 다음 단계

- **Phase 2**: Pick & Place (물체 집기/내려놓기) 구현

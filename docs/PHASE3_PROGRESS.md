# Phase 3 MARL 훈련 진행 기록

> 3대 로봇 동시 운용 — 충돌 없는 자율 경로 협상

---

## 목표

창고 환경에서 로봇 3대가 서로 충돌하지 않고 각자의 목표 지점에 도달하는 분산 실행 정책 학습.

- **Actor**: 로봇 1대의 관측만 입력 → 실제 Jetson 배포 가능
- **Critic**: 전체 상태 입력 → 훈련 시 centralized 평가
- **목표 성능**: 충돌률 < 1%, 교착률 < 1%

---

## 아키텍처: True CTDE MAPPO

### Centralized Training, Decentralized Execution

```
훈련 시:  Actor(17-dim) + Critic(51-dim global state)
실행 시:  Actor(17-dim) 단독 — 각 로봇 독립 추론
```

### 관측 공간 진화 (OBS_PER_ROBOT)

| 버전 | 차원 | 내용 | 문제 |
|------|------|------|------|
| v1 | 9-dim | goal(3) + vel(3) + shelf(1) + 다른로봇 거리(2) | 다른 로봇 방향 정보 없음 |
| v2 | 13-dim | v1 + 다른 로봇 방향벡터(body frame) (4) | 상대 속도 없어 접근 인식 불가 |
| v3 | **17-dim** | v2 + 다른 로봇 상대속도(vx_rel, vy_rel) (4) | **현재** |

**v3 (17-dim) 구성 (로봇 1대 기준):**
```
goal_x_body, goal_y_body, goal_dist       : 3
vx_body, vy_body, omega_z                 : 3
min_shelf_dist                            : 1
other_dx, dy, dist, vx_rel, vy_rel × 2   : 10  (N-1=2대)
합계: 17
```

### 네트워크 구조

```
Actor : Linear(17→256) → ELU → Linear(256→128) → ELU → Linear(128→64) → ELU → Linear(64→3) → Tanh
Critic: Linear(51→256) → ELU → Linear(256→128) → ELU → Linear(128→64) → ELU → Linear(64→1)
```

---

## 보상 설계: Markov Potential Game (MPG)

### 이론적 근거

Yan & Liu (arXiv:2503.22867, 2025) Theorem 5:

```
rᵢ = α·rᵢ_self + β·Σⱼ≠ᵢ rᵢⱼ
rᵢⱼ = rⱼᵢ (대칭) → Nash Equilibrium 존재 및 수렴 보장
```

### Self Reward

```python
r_self = -dist_sq_norm + rew_goal * reached + rew_time
# dist_sq_norm = (||pos - goal||² / goal_range²)
# rew_goal = 6.0 (목표 도달)
# rew_time = -0.01 (time penalty)
```

### Pairwise Reward — Delta Repulsion Potential (PBRS)

```python
r_yield = -(1/d_t - 1/d_{t-1}) * W_REP
# 멀어지면(양보): d_t > d_{t-1} → r > 0 (보너스)
# 가까워지면:    d_t < d_{t-1} → r < 0 (패널티)
# SAFE_DIST 밖은 0으로 마스킹 (불필요한 간섭 방지)
```

**핵심 특성**: 왕복 합산 = 0 (제로섬) → reward hacking 불가

### 최종 파라미터

| 파라미터 | 값 | 이유 |
|---------|-----|------|
| `rew_collision` | -200 | 교착 패널티(-90)보다 110점 낮게 — 자살 학습 방지 |
| `rew_goal` | 6.0 | 목표 도달 강한 유인 |
| `SAFE_DIST` | 2.5m | 물리적 회피 가능 거리 확보 (아래 계산 참고) |
| `W_REP` | 1.5 | delta repulsion 스케일 (`potential_reward.py` 하드코딩) |
| `ALPHA` | 1.0 | 자기 목표 가중치 (`cfg.alpha`) |
| `cfg.beta` | **0.5** (model_9999.pt 실제 훈련값) / **1.5** (재훈련 시 의도값) | pairwise 전체 가중치 — effective = cfg.beta × W_REP |

**SAFE_DIST 물리 계산 근거:**
```
충돌 판정 거리: 0.55m
dt = 1/60s × decimation 4 = 0.067s/step
최대 속도 ≈ 1.0 m/s
3스텝 반응 필요 → 최소 반응 거리 = 3 × 0.067 × 1.0 ≈ 0.2m
SAFE_DIST 1.2m → 횡이동 가능: 1.2 × 0.067 × 1.0 = 0.08m < 0.55m (회피 불가!)
SAFE_DIST 2.5m → 충분한 여유 확보
```

---

## 훈련 이력

### Phase 3.0 — IPPO (기준선)

**체크포인트**: `logs/warehouse_ippo/model_400.pt`

| 시나리오 | 충돌 | 교착 | 전원도달 |
|---------|------|------|---------|
| S1 정면 충돌 | 0% | 0% | **100%** |
| S2 3-way 교착 | 100% | 0% | 0% |
| S3 통로 우선 | 100% | 0% | 0% |
| S4 동일 목표 | 100% | 0% | 0% |
| S5 혼합 | 0% | 100% | 0% |

→ S1만 해결. 협력 관계를 전혀 학습하지 못함.

---

### Phase 3.1 — True CTDE MAPPO (9-dim obs)

**체크포인트**: `logs/warehouse_mappo/model_4999.pt`
**변경**: IPPO → CTDE (Centralized Critic 추가), MPG 보상 도입

| 시나리오 | 충돌 | 교착 | 전원도달 |
|---------|------|------|---------|
| S1 정면 충돌 | 0% | 0% | **100%** |
| S2 3-way 교착 | 100% | 0% | 0% |
| S3 통로 우선 | 0% | 0% | **100%** |
| S4 동일 목표 | 100% | 0% | 0% |
| S5 혼합 | 0% | 100% | 0% |

**성과**: S3 해결. S1 유지. Critic이 전체 상태를 보게 되면서 통로 우선순위 학습 성공.

**미해결**: S2(방향 정보 부족), S4(구조적 문제), S5(교착)

---

### Phase 3.2 — obs 13-dim + W_REP 조정 실험

**체크포인트**: `logs/warehouse_mappo/model_5399.pt`
**변경**: obs 9→13dim (방향벡터 추가), W_REP 여러 값 실험

| W_REP | 결과 | 문제 |
|-------|------|------|
| 0.5 | S1 충돌 100% | 반발력 너무 약함 |
| 1.0 | S2 충돌 유지 | 개선 없음 |
| 1.5 | S1 교착 100% | **회귀 발생** |

**교훈**: W_REP fine-tuning만으로는 근본 문제 해결 불가. 구조적 수정 필요.

---

### Phase 3.3 — 구조적 문제 전체 식별 및 수정 (현재)

**식별된 구조적 문제 3가지:**

#### 문제 1: Suicide Learning (자살 학습)
```
rew_collision(-20) < rew_stationary × 300steps(-90)
→ 로봇이 교착보다 충돌을 선택하는 것이 수학적으로 유리
해결: rew_collision = -200 (교착 -90과 110점 차이)
```

#### 문제 2: SAFE_DIST 물리적 한계
```
SAFE_DIST=1.2m → 반응 가능 시간 = 1.2/1.0 = 1.2s = 18스텝
but 실제 회피에 필요한 횡이동 불가 (0.08m < 0.55m)
해결: SAFE_DIST = 2.5m (충분한 조기 경보)
```

#### 문제 3: 상대 속도 관측 누락
```
접근하는 로봇의 속도를 모름 → 정지한 로봇과 돌진하는 로봇 구분 불가
해결: obs에 vx_rel_body, vy_rel_body 추가 → 17-dim
```

**최종 수정 커밋**: `5de8e28`

---

## 훈련 속도 최적화

| num_envs | steps/s | 비고 |
|---------|---------|------|
| 128 | ~14,000 | 초기 |
| 1024 | ~120,000 | 단순 증가 |
| 8192 | **~460,000** | PhysX replicate_physics 활용 |

Isaac Sim PhysX가 병목 (GPU 39% 사용). `num_envs=8192`에서 GPU 봉인 해제, ~33배 속도 향상.

---

## 최종 훈련 결과 — model_9999.pt (Freeze 확정)

**모델**: 17-dim obs + rew_collision=-200 + SAFE_DIST=2.5 + W_REP=1.5 + cfg.beta=0.5, 10000 iter

| 시나리오 | 충돌률 | 교착률 | 전원도달 | 비고 |
|---------|:------:|:------:|:-------:|------|
| S1 정면 충돌 | 0% | 7% | **93%** | |
| S2 3-way 교착 | 0% | 16% | **84%** | |
| S3 통로 우선 | 1% | 4% | **95%** | |
| **S1~S3 평균** | **0.3%** | **9%** | **90.7%** | 목표: 충돌<1% ✅ / 교착<1% ❌ |
| S4 동일 목표 | 0% | 100% | 0% | 구조적 한계 — LLM 레이어로 처리 |
| S5 혼합 | — | — | — | 시나리오 재설계 완료 (아래 참고) |

**참고**: model_11998.pt (계속 훈련) 시 S3 95%→1% 퇴행 → model_9999.pt가 실질적 최고점.

### S4 한계: Qwen 오케스트레이션으로 해결
동일 목표 할당은 MPG 보상 구조상 훈련으로 해결 불가. Jetson 배포 시 목표 할당 레이어에서 사전에 서로 다른 목표 부여.

### S5 시나리오 버그 수정 (2026-05-23)
기존 S5 대각 경로 `(-3,-3)→(3,3)`, `(-3,3)→(3,-3)`는 각각 2개의 선반 내부를 통과:
- t≈0.05-0.10: 출발점 근처 선반 (bottom-left / top-left)
- t≈0.90-0.95: 도착점 근처 선반 (top-right / bottom-right)

→ 물리적으로 해결 불가능한 시나리오. y=0 중앙 통로(선반 없음)를 교차하는 경로로 교체:
```
spawns: [(-4.0, 0.0), (4.0, 0.3), (0.0, 4.0)]
goals:  [( 4.0, 0.0), (-4.0, 0.3), (0.0, -4.0)]
```
새 S5는 RunPod에서 eval 재실행 필요.

---

## 다음 단계

1. **S5 재eval** — RunPod에서
   ```bash
   python training/multi_robot/eval_scenarios.py \
     --ckpt logs/warehouse_mappo/model_9999.pt \
     --num_episodes 100 --num_eval_envs 16 --tag 17dim_s5fix --headless
   ```

2. **Jetson 배포** (Phase 4)
   - Actor TorchScript export
   - ROS2 bridge 연결
   - Qwen 오케스트레이션 레이어 (목표 할당)

---

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `envs/warehouse/warehouse_marl_env.py` | 환경 정의 (obs, reward, termination) |
| `envs/warehouse/ippo_wrapper.py` | CTDE 래퍼 (Actor/Critic 분리) |
| `training/multi_robot/potential_reward.py` | MPG 보상 함수 |
| `training/multi_robot/train_marl.py` | 훈련 스크립트 |
| `training/multi_robot/eval_scenarios.py` | 시나리오별 평가 (병렬 지원) |

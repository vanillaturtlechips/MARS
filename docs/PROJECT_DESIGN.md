# 물류 창고 자율 멀티 로봇 시스템 — 설계도

> 목표: 물류 창고에서 여러 로봇이 명시적 명령 없이 목표만 받아 자율적으로 협력하는 시스템

---

## 1. 전체 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                   LAYER 1 — 두뇌 (PC / Cloud)        │
│                                                     │
│   Claude API ──→ LangChain ──→ Task Planner         │
│        │                           │                │
│      RAG                      Sub-Agents            │
│   (pgvector)                A/B/C/D/E               │
│   - 창고 맵                                          │
│   - 임무 기억                                        │
│   - 행동 라이브러리                                   │
└──────────────────────────┬──────────────────────────┘
                           │ ROS2 / WebSocket
┌──────────────────────────▼──────────────────────────┐
│               LAYER 2 — 현장 두뇌 (Jetson Nano Super) │
│                                                     │
│   소형 LLM (Llama 3.2 3B / Qwen2.5 3B, INT4 양자화) │
│        │                                            │
│   ┌────┴─────────────────────────────────┐          │
│   │  A2A 통신 (ROS2 토픽)                │          │
│   │  [재고] ←→ [경로] ←→ [이상감지]      │          │
│   │     ↕              ↕                │          │
│   │  [Robot1] ←→ [Robot2] ←→ [Robot3]  │          │
│   └──────────────────────────────────────┘          │
└──────────────────────────┬──────────────────────────┘
                           │ 모터 명령 (goal → RL policy)
┌──────────────────────────▼──────────────────────────┐
│               LAYER 3 — 물리 로봇                    │
│                                                     │
│   Isaac Lab 훈련 RL 정책 (TorchScript inference)     │
│   Robot 1 │ Robot 2 │ Robot 3 │ ...                 │
└─────────────────────────────────────────────────────┘
```

---

## 2. 핵심 알고리즘

### 2-1. 단일 로봇 — PPO + Domain Randomization + Teacher-Student

```
목적함수:
  J(π) = 𝔼_τ~π [ Σ γᵗ r(sₜ, aₜ) ]

Sim-to-Real 파이프라인 (3단계):

  [Step 1] System Identification
    배포 전 실제 하드웨어 측정:
    - 바닥 마찰 실측 → DR 범위 기준점 설정
    - 모터 지연/데드밴드 측정
    - 측정값 기반으로 DR 범위 설정 (추정 아닌 실측)

  [Step 2] Domain Randomization (DR)
    훈련 중 매 에피소드마다 랜덤화
    - 바닥 마찰:    μ ~ Uniform(0.3, 1.2)
    - 페이로드:     m ~ Uniform(0, 30kg)
    - 모터 지연:    d ~ Uniform(0, 20ms)
    - 센서 노이즈:  ε ~ N(0, 0.05)
    주의: 실측값 기준 3배 이상 넓은 범위는 과도한 랜덤화 위험

  [Step 3] Teacher-Student Distillation (Phase 2 이상 필수)
    교사 (시뮬 전용):
      특권 정보 접근 — 박스 정확한 위치/무게, 지형 실제값
    학생 (배포용):
      카메라 + LiDAR 센서 관측만 사용
      교사 궤적으로부터 증류(distillation)
    → 학생이 실제 센서로 특권 정보를 추론하도록 학습

Asymmetric Actor-Critic:
  훈련 시  Critic: 전체 상태 (위치, 속도, 힘 등) 사용
  배포 시  Actor:  카메라 + LiDAR 센서만 사용

하드웨어 안전 범위 (배포 필수):
  - 최대 속도 제한 (소프트/하드 이중)
  - 토크 제한
  - 비상 정지 백업 컨트롤러 (비학습 레이어)
```

### 2-2. 멀티 로봇 — IPPO 베이스라인 → MAPPO + Markov Potential Game

```
참고 논문: Yan & Liu, "Markov Potential Game Construction and MARL
           with Applications to Autonomous Driving" (arXiv 2503.22867, 2025)

훈련 순서:
  1단계: IPPO 베이스라인 먼저 실행 (Yu et al. 2022 — 단순 IPPO도 강력)
  2단계: MAPPO와 수렴 속도/성능 비교
  3단계: 개선이 유의미할 때만 MAPPO 채택

Theorem 5 기반 보상 설계 (MPG 보장):
  rᵢ = α * rᵢ^self(sᵢ, aᵢ) + β * Σⱼ≠ᵢ rᵢⱼ(sᵢ, sⱼ, aᵢ, aⱼ)
                                         ↑ 이 구조면 자동으로 MPG

  자기 보상 rᵢ^self (목표 추적):
    -||posᵢ - goalᵢ||²      거리 페널티 (연속)
    +10                     목표 도달 보너스
    -0.01 * t               시간 페널티

  쌍별 보상 rᵢⱼ (충돌 회피):
    rᵢⱼ = -1 / sqrt((xᵢ-xⱼ)² + (yᵢ-yⱼ)² + ε)
    ε = 1e-5  (분모 0 방지)
    → 가까울수록 큰 페널티, 멀면 0에 수렴

  포텐셜 함수 (자동 유도):
    φ^self  = Σᵢ rᵢ^self
    φ^joint = Σᵢ Σⱼ<ᵢ rᵢⱼ
    Φ = α * φ^self + β * φ^joint

  효과:
    - Nash Equilibrium 존재 및 수렴 수학적 보장
    - 교착 상태가 NE가 아님 → 자연 해소
    - 예상치 못한 로봇 행동에도 robust (논문: 0/100 충돌률)

  α, β 튜닝:
    α = 1.0  (효율 가중치)
    β = 1.5  (안전 가중치, 창고는 안전 우선)

탐색 중복 방지:
  - 에이전트별 엔트로피 보너스 적용
  - 역할 조건화(role-conditioning) 검토

고정 시나리오 평가 (필수):
  - 셀프플레이 수치 단독 신뢰 금지
  - 5가지 고정 시나리오로 별도 평가 (좁은 통로, 교착, 배터리 등)
  - rᵢⱼ 쌍 수 = N(N-1)/2 → 로봇 수 증가 시 스케일링 모니터링

CTDE (Centralized Training, Decentralized Execution):
  훈련: 전체 로봇 상태 + Φ 함께 학습
  실행: 각 로봇은 자기 센서만 보고 독립 실행
```

### 2-3. LLM 양자화 (Jetson 엣지 배포)

```
제약: Jetson Orin Nano Super 8GB (CPU/GPU 공유 메모리)
      RL 정책 + ROS2 + LLM 동시 실행 → LLM에 가용 메모리 ~4GB

양자화 방식 (ollama / llama.cpp GGUF):
  FP16 3B 모델: ~6.0 GB  → Jetson 불가
  INT8 3B 모델: ~3.5 GB  → 가능하지만 빠듯
  INT4 Q4_K_M:  ~2.0 GB  → 권장 (품질 손실 최소화)

ollama 모델 태그:
  llama3.2:3b-instruct-q4_K_M
  qwen2.5:3b-instruct-q4_K_M

벤치마킹 지표:
  - tokens/sec (목표: > 15 tok/s — 실시간 명령 생성 기준)
  - 구조화 JSON 출력 성공률 (로봇 명령 파싱 필수)
  - 첫 토큰 레이턴시 (< 500ms 목표)
  - 메모리 점유 (jtop으로 측정)
```

### 2-4. World Model (Phase 5 이후 선택)

```
DreamerV3 방식:
  ŝ_{t+1} = f_θ(sₜ, aₜ)   상태 전이 예측
  r̂_t     = g_φ(sₜ, aₜ)   보상 예측

  훈련 가속:
    PhysX 시뮬 대신 f_θ로 "상상" 롤아웃
    → 실제 시뮬 대비 ~100x 샘플 효율
```

---

## 2.5 Operational State Abstraction Layer

### 목적

ROS2 raw telemetry를 AI Agent가 reasoning 가능한 operational semantics로 변환한다.

AI Agent는 `/odom`, `/scan` 같은 low-level robotics telemetry를 직접 처리하지 않고,
운영 의미(Operational Meaning)를 기반으로 reasoning을 수행한다.

---

### 데이터 흐름

```text
ROS2 Topics
    ↓
Telemetry Processor / State Aggregator
    ↓
Operational State JSON
    ↓
Claude Orchestrator + Sub Agents
```

---

### 주요 ROS2

```text
/odom           → nav_msgs/Odometry
/battery_state  → sensor_msgs/BatteryState
/scan           → sensor_msgs/LaserScan
/diagnostics    → diagnostic_msgs/DiagnosticArray
/task_status    → custom message
/robot_status   → custom message
```

---

### Operational State JSON 구조

```json
{
  "timestamp": "2026-05-17T11:32:00Z",

  "warehouse_state": {
    "traffic_level": "HIGH",
    "active_robot_count": 12,
    "blocked_zone_count": 2,
    "system_load": "PEAK"
  },

  "robots": [
    {
      "robot_id": "R1",
      "zone": "A",
      "position": {
        "x": 12.4,
        "y": 7.8
      },
      "status": "BUSY",
      "battery_pct": 18,
      "current_task_id": "TASK_42",
      "traffic_state": "BLOCKED",
      "health_state": "NORMAL",
      "estimated_idle_eta_sec": 120
    }
  ],

  "tasks": [
    {
      "task_id": "TASK_42",
      "type": "PICKUP_DELIVERY",
      "priority": "HIGH",
      "status": "IN_PROGRESS",
      "assigned_robot": "R1",
      "source_zone": "A",
      "destination_zone": "C",
      "queue_wait_sec": 45
    }
  ],

  "events": [
    {
      "event_type": "LOW_BATTERY",
      "robot_id": "R2",
      "severity": "HIGH",
      "timestamp": "2026-05-17T11:31:20Z"
    }
  ],

  "alerts": [
    {
      "type": "BOTTLENECK_RISK",
      "zone": "B",
      "confidence": 0.82,
      "recommended_action": "reroute nearby robots"
    }
  ]
}
```

---

### Field Domain 정의

#### warehouse_state

##### traffic_level

```text
LOW
MEDIUM
HIGH
CRITICAL
```

##### active_robot_count

```text
Integer >= 0
```

##### blocked_zone_count

```text
Integer >= 0
```

##### system_load

```text
LOW
NORMAL
HIGH
PEAK
```

---

#### robots

##### robot_id

```text
String
```

##### zone

```text
A-Zone
B-Zone
Loading
Charging
etc.
```

##### position

###### x

```text
Float
```

###### y

```text
Float
```

##### status

```text
IDLE
BUSY
MOVING
BLOCKED
CHARGING
ERROR
OFFLINE
```

##### battery_pct

```text
0 ~ 100
```

##### current_task_id

```text
TASK_xxx | null
```

##### traffic_state

```text
NORMAL
CONGESTED
BLOCKED
```

##### health_state

```text
NORMAL
WARNING
CRITICAL
```

##### estimated_idle_eta_sec

```text
Integer >= 0
```

---

#### tasks

##### task_id

```text
String
```

##### type

```text
PICKUP
DELIVERY
PICKUP_DELIVERY
CHARGING
INSPECTION
```

##### priority

```text
LOW
MEDIUM
HIGH
CRITICAL
```

##### status

```text
PENDING
ASSIGNED
IN_PROGRESS
BLOCKED
COMPLETED
FAILED
```

##### assigned_robot

```text
Robot ID | null
```

##### source_zone

```text
Warehouse Zone
```

##### destination_zone

```text
Warehouse Zone
```

##### queue_wait_sec

```text
Integer >= 0
```

---

#### events

##### event_type

```text
LOW_BATTERY
CONGESTION_ALERT
ROBOT_FAILURE
TASK_TIMEOUT
COLLISION_RISK
SENSOR_ERROR
TRAFFIC_SPIKE
```

##### robot_id

```text
Robot ID | null
```

##### severity

```text
LOW
MEDIUM
HIGH
CRITICAL
```

##### timestamp

```text
ISO8601 datetime
```

---

#### alerts

##### type

```text
BOTTLENECK_RISK
OVERLOAD_RISK
CHARGING_SHORTAGE
CONGESTION_WARNING
```

##### zone

```text
Warehouse Zone
```

##### confidence

```text
0.0 ~ 1.0
```

##### recommended_action

```text
Free Text
```

---

### Middle Layer 역할

* ROS2 telemetry aggregation
* low-level robotics state → operational semantics 변환
* event generation
* AI reasoning용 compact state 생성

---

### 구현 파일 구조

```text
agents/state/
  ├── telemetry_collector.py
  ├── state_aggregator.py
  ├── event_engine.py
  └── operational_state_schema.py
```

---

### 설계 핵심

Operational State Layer는 robotics telemetry와 AI reasoning 사이의 abstraction layer 역할을 수행한다.

이 계층을 통해:

* raw robotics telemetry를 operational semantics로 변환
* event-driven reasoning 지원
* supervisory AI orchestration 가능
* multi-agent coordination 지원
* bottleneck/anomaly detection 수행

구조를 구현할 수 있다.

---

## 3. 서브 에이전트 설계

| 에이전트 | 역할 | 핵심 도구 |
|---------|------|---------|
| A. 재고 관리자 | 상품 위치 검색, 선반 현황 | pgvector RAG |
| B. 경로 계획자 | 충돌 없는 경로 생성, 임무 배정 | pathfinder, task_allocator |
| C/D/E. 로봇 담당 | goal → RL 정책 트리거, 상태 모니터링 | robot.set_goal(), ROS2 |
| F. 이상 감지자 | 타임아웃, 센서 이상, 안전 구역 침입 | 브로드캐스트 긴급 정지 |

### A2A 통신 규칙
```
직접 통신 (오케스트레이터 거치지 않음):
  - 충돌 협상      (실시간, < 100ms)
  - 임무 재배정    (로봇 장애 시)
  - 긴급 정지      (이상 감지 즉시 브로드캐스트)

오케스트레이터 경유:
  - 새 임무 할당
  - 전체 계획 변경
  - 로그 / 보고
```

### 자율 시스템 안전 패턴
```
명령 파이프라인:
  오케스트레이터 제안 → 검증 → 실행 → 체크포인트
        ↑ Propose-then-Commit          ↑ Rollback 지점

Cost Governor:
  - Claude API 호출당 최대 토큰 제한
  - 세션당 비용 상한 (무한 루프 방지)
  - 예산 초과 시 자동 중단

Kill Switch & Canary:
  - F 에이전트(이상 감지자)가 전체 정지 브로드캐스트
  - 카나리 토큰: 무음 오작동 감지용 주기적 헬스체크
  - 레이턴시 임계값 초과 시 자동 알림

Propose-then-Commit:
  - 로봇 이동 명령은 경로 검증 후 실행
  - 충돌 위험 경로는 재계획 요청
```

---

## 4. 개발 단계 (Phases)

---

### Phase 0 — 기반 환경 구축 ✅ 완료

**내용:** Isaac Lab 설치 및 기본 RL 동작 확인

- [x] Isaac Sim 5.1.0 + Isaac Lab 2.3.2 설치
- [x] NVIDIA Driver 580.142, CUDA 13.0
- [x] Cartpole PPO 훈련 (보상 0.10 → 4.38)
- [x] GUI 뷰어 동작 확인

---

### Phase 1 — 단일 로봇 이동 ✅ 완료

**목표:** 창고 환경에서 로봇 한 대가 지정 좌표로 자율 이동

**결과:**
```
Phase 1   (장애물 없음): 골 도달률 100%, 평균 24.9 스텝
Phase 1.5 (선반 장애물): 골 도달률 94.4%, 평균 41.2 스텝
           실패 5.6% — 선반 모서리 끼임, 좁은 통로 타임아웃
```

**구현 상세:**

```
환경: DirectRLEnv 서브클래스 (warehouse_env.py / warehouse_obstacle_env.py)

관측 공간:
  Phase 1   6차원: [goal_x_body, goal_y_body, goal_dist, vx_body, vy_body, omega_z]
  Phase 1.5 7차원: 위 6개 + min_obstacle_dist (AABB 최소 거리)

행동 공간: 3차원 [cmd_vx, cmd_vy, cmd_omega], body frame, [-1,1] 정규화
  → max_vx=1.5 m/s, max_vy=1.0 m/s, max_omega=2.0 rad/s 스케일링

보상:
  rew_dist = -0.3 × dist_to_goal       (매 스텝, 연속)
  rew_goal = +10.0                      (골 도달 시)
  rew_time = -0.001                     (매 스텝 시간 패널티)
  Phase 1.5 추가:
    rew_prox_warn = -0.1  (선반까지 < 0.8m)
    rew_prox_crit = -1.0  (선반까지 < 0.4m)

에피소드: 20초 (300 스텝 @ 15Hz), 골 반경 0.35m, 골 범위 1~4m

네트워크: MLP [256, 128, 64], ELU, lr=3e-4 adaptive (desired_kl=0.01)
env 수: 1024 (훈련) / 256 (검증)
iterations: Phase 1 — 1000, Phase 1.5 — 100 (lr 폭주로 조기 선택)

훈련 이슈:
  VF loss 폭발: rew_prox_crit=-5.0 → -1.0으로 축소
  noise std 폭주: adaptive schedule이 KL 낮을 때 lr 무한 증가
  → iter=100 체크포인트 (noise std ≈ 0.92)를 Phase 1.5 기준 모델로 채택
```

모델 위치:
- `logs/warehouse_nav/model_999.pt`          (Phase 1)
- `logs/warehouse_obstacle_nav/model_100.pt` (Phase 1.5)

잔여 과제: 5.6% 실패율 → Phase 3 멀티로봇 적용 시 증폭 가능. Phase 2 전 원인 분석 권장.

- [x] 창고 환경 USD 파일 제작
- [x] 단일 로봇 PPO 훈련 (1024 병렬 env)
- [x] 헤드리스 검증 → GUI 시각 확인

---

### Phase 2 — 단일 로봇 조작 (Pick & Place)

**목표:** 로봇이 박스를 집어서 지정 위치에 내려놓기

**구현 상세:**

```
1단계 — 그리퍼 로봇 환경 구성
  파일: envs/warehouse/warehouse_manipulation_env.py
  로봇: 6-DOF 암 + 평행 그리퍼 (Isaac Lab articulation)
  박스: 다양한 크기(0.2~0.4m) / 무게(0.5~5kg) DR 적용
  목표 선반: 고정 4위치 → 후반부 랜덤 확장

2단계 — 관측 공간 (Teacher / Student 분리)
  Teacher 관측 (시뮬 전용, 특권 정보):
    [박스_xyz, 박스_quat, 박스_무게, end_effector_xyz, gripper_width,
     goal_xyz, joint_pos×6, joint_vel×6]  → ~28차원

  Student 관측 (배포용, 실제 센서):
    [rgb_d_feature×64(CNN 출력), end_effector_xyz, gripper_width,
     goal_xyz_approx, joint_pos×6, joint_vel×6]  → ~82차원

3단계 — 보상 설계 (4단계 커리큘럼)
  Phase A (접근): rew = 0.1 × (1 / dist_to_box)          박스까지 접근
  Phase B (파지): rew = +5.0 × grasp_success              파지 성공
  Phase C (이송): rew = 0.1 × (1 / dist_box_to_goal)     목표로 이동
  Phase D (거치): rew = +20.0 × place_success             최종 거치 성공
                  rew = -10.0 × box_drop                  낙하 패널티

  → 커리큘럼: Phase A 수렴 → B 추가 → C 추가 → D 추가
    각 Phase에서 성공률 > 80% 되면 다음 단계 추가

4단계 — Teacher-Student 증류
  (a) Teacher 정책 먼저 훈련 (PPO, 특권 정보 사용)
      목표: place_success_rate > 90%
  (b) Teacher 궤적 데이터 수집 (10만+ 에피소드)
  (c) Student가 Teacher 행동을 모방학습 (DAgger 또는 BC+fine-tuning)
  (d) Student 단독 평가: 시뮬 내 seen/unseen 박스 크기

5단계 — Asymmetric Actor-Critic
  훈련 Critic: Teacher 관측 (박스 정확한 위치, 무게)
  배포 Actor:  Student 관측 (RGB-D 추정)

6단계 — TorchScript export → Jetson 배포 테스트
  deploy/export_model.py 확장 (manipulation actor)
  Jetson에서 deploy/jetson/inference.py로 단독 실행 확인

네트워크: MLP [512, 256, 128], ELU (Teacher) / CNN+MLP (Student)
env 수: 256 (Phase 2는 접촉 시뮬로 FPS 감소)
예상 훈련 시간: ~4.5시간 (RTX 2070)
```

체크포인트:
- [ ] 그리퍼 로봇 Isaac Lab 환경 (`warehouse_manipulation_env.py`)
- [ ] Teacher PPO 훈련 (place_success_rate > 90%)
- [ ] Teacher 궤적 데이터 수집 (10만 에피소드)
- [ ] Student 모방학습 + 단독 평가
- [ ] Unseen 박스 크기 zero-shot 평가
- [ ] TorchScript export → Jetson inference 확인

---

### Phase 3 — 멀티 로봇 협력

**목표:** 로봇 3대가 동시에 서로 방해 없이 임무 수행

**구현 상세:**

```
1단계 — 멀티 로봇 Isaac Lab 환경
  파일: training/multi_robot/train_ippo.py
  로봇 3대를 동일 창고 환경에 동시 스폰
  각 로봇 독립 관측 (Phase 1.5 기준 7차원 × 3 → 분리 실행)
  충돌 감지: 로봇 간 거리 < 0.5m → episode terminated

2단계 — IPPO 베이스라인
  각 로봇이 완전 독립 PPO로 학습 (공유 없음)
  동일 네트워크 구조, Parameter Sharing (가중치 공유)
  → VRAM 절감: 로봇 3대를 모델 1개로

  보상 (IPPO 단계):
    rᵢ^self만 사용 (-||pos-goal||² + 10·reached - 0.01·t)
    충돌 패널티: -5.0 (단순 이진)

  수렴 확인 기준 (100 iter 조기 진단):
    30 iter 안에 평균 보상 상승세 확인
    → 없으면 보상 재설계

3단계 — MPG 보상 구현 (potential_reward.py)
  rᵢ = α * rᵢ^self + β * Σⱼ≠ᵢ rᵢⱼ
  α = 1.0, β = 1.5

  rᵢⱼ = -1 / sqrt((xᵢ-xⱼ)² + (yᵢ-yⱼ)² + 1e-5)
  → 이진 페널티 → 연속 반발력으로 교체

4단계 — MAPPO 전환 (train_marl.py)
  공유 Critic: 전체 로봇 상태 concatenation 입력
  독립 Actor:  각 로봇 자기 관측만 (CTDE)
  IPPO 체크포인트에서 fine-tuning (학습 안정화)

5단계 — IPPO vs MAPPO 비교
  동일 환경, 동일 시나리오 5종에서 측정:
  - 교착 발생률 (목표: < 1%)
  - 평균 임무 완료 시간
  - 충돌 발생률 (목표: < 1%)
  → 유의미한 차이 없으면 IPPO 유지 (단순한 게 낫다)

6단계 — 고정 시나리오 5종 평가
  시나리오 1: 정면 충돌 (좁은 통로 양방향 진입)
  시나리오 2: 3-way 교착 (세 로봇 삼각 대치)
  시나리오 3: 배터리 우선순위 (배터리 낮은 로봇 우선 통과)
  시나리오 4: 동일 목표 경쟁 (두 로봇이 같은 선반 목표)
  시나리오 5: 장애물 + 다중 로봇 혼합

VRAM 관리 (RTX 2070 8GB 기준):
  Parameter Sharing + FP16 → 3대 × 128 env ≈ 6.5GB
```

체크포인트:
- [x] 멀티 로봇 Isaac Lab 환경 구성 (3대 동시 스폰, 충돌 감지)
- [x] IPPO 베이스라인 수렴 확인 — 400 iter, noise_std 1.00→0.56, ep_len 220-280
- [x] MPG 보상 구현 (`potential_reward.py`) — Danger Zone 마스킹, rew_collision=-150
- [x] True CTDE MAPPO 구현 — Critic 27-dim global, per-robot credit assignment
- [x] IPPO vs MAPPO 5종 시나리오 비교 완료 (PHASE3_IPPO_VS_MAPPO.md)
  - True CTDE MAPPO: 전원도달 40%, 교착 20% (IPPO 대비 전원도달 2배)
- [x] 17-dim obs + rew_collision=-200 + SAFE_DIST=2.5 from scratch 훈련 (model_9999.pt)
- [x] 충돌률 0.2% 달성 (목표 <1% ✅)
- [x] S1~S3 전원도달률 90.7% 달성 — Low-level Controller 확정 (Freeze)
- [ ] A2A 충돌 협상 프로토콜 구현 (다른 팀원 담당)
- [ ] S4/S5 → Phase 4 오케스트레이터로 해결 예정

---

### Phase 4 — 에이전트 레이어 구축

**목표:** Claude API + 서브 에이전트가 임무를 자율 분해하고 로봇에게 전달

**구현 상세:**

```
1단계 — pgvector RAG 구축
  DB: PostgreSQL + pgvector extension
  테이블:
    warehouse_map: shelf_id, location_xyz, product_id, embedding
    task_history:  task_id, success, robot_id, duration, embedding
    action_library: action_name, description, embedding, json_template

  임베딩: text-embedding-3-small (OpenAI) or nomic-embed-text (로컬)
  검색: 코사인 유사도, top-k=5

2단계 — LLM 양자화 및 벤치마킹
  ollama로 두 모델 Q4_K_M 로드:
    ollama pull llama3.2:3b-instruct-q4_K_M
    ollama pull qwen2.5:3b-instruct-q4_K_M

  벤치마킹 스크립트 (benchmark_llm.py):
    프롬프트: "로봇 A에게 선반 B3 → 게이트 G1 이동 명령을 JSON으로 생성"
    측정: tokens/sec, 첫 토큰 레이턴시, JSON 파싱 성공률 (100회)
    jtop으로 VRAM 점유 측정

  선택 기준:
    tokens/sec > 15 AND JSON 성공률 > 95% AND 메모리 < 3.5GB

3단계 — Claude API 오케스트레이터
  파일: agents/orchestrator/orchestrator.py
  claude-sonnet-4-6 사용
  임무 입력 → 서브 에이전트 A~F 호출 → 결과 집계

  Cost Governor (cost_governor.py):
    max_tokens_per_call = 4096
    max_cost_per_session = $2.0  (무한 루프 방지)
    예산 초과 시 → 임무 중단 + 알림

4단계 — Propose-then-Commit 패턴
  모든 로봇 이동 명령은 2단계:
    (1) B에이전트(경로 계획자)가 경로 검증
    (2) 충돌 없음 확인 후 C/D/E 에이전트에 실행 전달
  검증 실패 시 → 재계획 요청 (최대 3회 재시도)

5단계 — 서브 에이전트 A~F 구현
  A. inventory_agent.py: pgvector 검색으로 상품 위치 반환
  B. path_agent.py: A* + 로봇 3대 경로 충돌 사전 검사
  C/D/E. robot_agent.py: ROS2 /goal_pose 퍼블리시
  F. safety_agent.py: 타임아웃 감시 + Kill Switch 브로드캐스트

6단계 — Jetson LLM 연동
  ros2_bridge.py가 /task_command 토픽 수신
  → ollama API (localhost:11434) 호출
  → JSON 파싱 → /goal_pose 퍼블리시
  레이턴시 목표: /task_command 수신 → /goal_pose 발행 < 1초
```

체크포인트:
- [ ] PostgreSQL + pgvector 구축 (`knowledge_base.py`)
- [ ] 창고 맵 / 임무 기록 / 행동 라이브러리 임베딩 적재
- [ ] LLM 벤치마킹 스크립트 작성 및 Jetson 실행
- [ ] 모델 선택 확정 (llama3.2 vs qwen2.5)
- [ ] Claude API 오케스트레이터 + Cost Governor
- [ ] Propose-then-Commit 경로 검증 구현
- [ ] 서브 에이전트 A~F 구현
- [ ] Jetson LLM → ROS2 브릿지 end-to-end 확인

---

### Phase 5 — 통합 및 시뮬 검증

**목표:** 전체 파이프라인이 Isaac Sim 안에서 end-to-end 동작

**구현 상세:**

```
1단계 — 시나리오 통합 테스트 5종
  시나리오 1 입고:
    "박스 10개를 3구역 선반에 배치"
    Claude → 경로 계획 → 로봇 3대 동시 작업
    측정: 완료 시간, 충돌 없음 여부

  시나리오 2 출고:
    "상품 A 5박스를 2번 게이트로"
    재고 에이전트 RAG 검색 → 위치 파악 → 로봇 배정

  시나리오 3 장애:
    "Robot 2 배터리 부족 → 자동 재배정"
    F 에이전트 감지 → B 에이전트 재계획 → 나머지 로봇 인수

  시나리오 4 안전:
    "5구역 사람 감지 → 전체 정지"
    F 에이전트 Kill Switch → 전 로봇 즉시 정지
    레이턴시 측정: 감지 → 정지 < 100ms

  시나리오 5 교착:
    "좁은 통로 로봇 2대 동시 진입 → 자동 해소"
    MPG 보상으로 NE 수렴 확인

2단계 — Sim-to-Real 갭 측정
  Jetson 실배포 전 시뮬 vs 실제 비교:
  - DR 범위 내 랜덤 물리값 시뮬: 성능 저하 없음 확인
  - 미관측 변형(unseen DR)에서 zero-shot 평가

3단계 — 레이턴시 프로파일링
  Claude API → Jetson LLM: ≤ 500ms
  Jetson LLM → /goal_pose 발행: ≤ 500ms
  /goal_pose → 로봇 도달: 환경 의존 (측정 값으로 기록)
  /odom → /cmd_vel (RL policy): ≤ 10ms (15Hz 기준)
```

체크포인트:
- [ ] 시나리오 5종 전부 통과
- [ ] 임무 완료율 > 95%
- [ ] 교착 발생률 < 1%
- [ ] Kill Switch 레이턴시 < 100ms 확인
- [ ] Unseen DR zero-shot 평가
- [ ] 레이턴시 전 구간 측정 기록
- [ ] 최종 시뮬 영상 기록

---

## 5. 기술 스택 요약

| 영역 | 기술 | 버전 |
|------|------|------|
| 시뮬레이터 | Isaac Sim | 5.1.0 |
| RL 프레임워크 | Isaac Lab | 2.3.2 |
| 단일 로봇 RL | rsl_rl (PPO) | 3.0.1 |
| 멀티 로봇 RL | IPPO → MAPPO (커스텀) | - |
| 딥러닝 (훈련 PC) | PyTorch | 2.7.0+cu128 |
| 딥러닝 (Jetson) | PyTorch | 2.8.0 (JetPack 6.2) |
| 오케스트레이터 | Claude API | claude-sonnet-4-6 |
| 에이전트 프레임워크 | LangChain | 프로토타입 |
| RAG | pgvector (PostgreSQL) | - |
| Jetson LLM | Qwen2.5 3B (채택) | INT4 Q4_K_M (ollama), 21.5 tok/s |
| 통신 | ROS2 Humble | - |
| 모델 배포 | TorchScript | - |
| 엣지 디바이스 | Jetson Orin Nano Super | 8GB |
| 훈련 하드웨어 | RTX 2070 Mobile | 8GB VRAM |

---

## 6. 프로젝트 디렉토리 구조

```
~/MARS/
│
├── envs/
│   └── warehouse/
│       ├── warehouse_env.py              # Phase 1 환경 ✅
│       ├── warehouse_obstacle_env.py     # Phase 1.5 환경 ✅
│       ├── warehouse_manipulation_env.py # Phase 2 환경
│       └── assets/                       # USD 파일 (선반, 박스, 로봇)
│
├── training/
│   ├── single_robot/
│   │   ├── train_navigation.py           # Phase 1 ✅
│   │   ├── train_navigation_obstacle.py  # Phase 1.5 ✅
│   │   └── train_manipulation.py         # Phase 2
│   └── multi_robot/
│       ├── train_ippo.py                 # Phase 3 베이스라인
│       ├── train_marl.py                 # Phase 3 MAPPO
│       └── potential_reward.py           # Markov Potential Game 보상
│
├── agents/
│   ├── orchestrator/
│   │   ├── orchestrator.py               # Claude API 오케스트레이터
│   │   ├── task_planner.py               # LangChain 임무 분해
│   │   └── cost_governor.py              # API 비용 제한
│   ├── sub_agents/
│   │   ├── inventory_agent.py            # 재고 관리 (pgvector 검색)
│   │   ├── path_agent.py                 # 경로 계획 + 충돌 사전 검사
│   │   ├── robot_agent.py                # 로봇 담당 (ROS2 goal 발행)
│   │   └── safety_agent.py               # 이상 감지 + Kill Switch
│   └── rag/
│       ├── knowledge_base.py             # pgvector DB 구축 및 검색
│       └── warehouse_docs/               # 창고 지식 원본
│
├── deploy/
│   ├── export_model.py                   # TorchScript export (훈련 PC)
│   └── jetson/
│       ├── inference.py                  # Jetson 추론 엔진 + latency 측정
│       └── ros2_bridge.py               # ROS2 /odom + /goal_pose → /cmd_vel
│
├── logs/
│   ├── warehouse_nav/model_999.pt        # Phase 1 최종 모델 ✅
│   └── warehouse_obstacle_nav/model_100.pt # Phase 1.5 기준 모델 ✅
│
└── docs/
    ├── PROJECT_DESIGN.md
    ├── PHASE1_PROGRESS.md
    ├── MARL_MPG_DECISION.md
    └── TRAINING_TIME_ESTIMATION.md
```

---

## 7. Jetson 환경 현황

```
하드웨어: Jetson Orin Nano Super Developer Kit (8GB)
OS: Ubuntu 22.04 (JetPack 6.2, L4T R36.4.7)

설치 완료:
  ✅ PyTorch 2.8.0 + CUDA (cuSPARSELt 0.7.0, cuDSS 0.7.1.4)
  ✅ torchvision 0.23.0
  ✅ ROS2 Humble (ros-humble-ros-base)
  ✅ Python 3.10.12

벤치마킹 결과 (2026-05-16):
  ┌──────────────────────────────┬────────────┬────────────┐
  │ 항목                         │ llama3.2:3b│ qwen2.5:3b │
  ├──────────────────────────────┼────────────┼────────────┤
  │ tokens/sec                   │ 21.1       │ 21.5 ✅    │
  │ 첫 토큰 레이턴시 (ms)        │ 663        │ 640 ✅     │
  │ JSON 성공률 (%)              │ 100        │ 100        │
  └──────────────────────────────┴────────────┴────────────┘
  → qwen2.5:3b-instruct-q4_K_M 채택

  ✅ ollama 설치 완료
  ✅ qwen2.5:3b-instruct-q4_K_M 설치 완료
  ✅ actor_phase15.pt inference latency: 0.33ms = 3034 Hz (목표 100Hz의 30배)
  ✅ ros2_bridge.py: /goal_pose → RL actor → /cmd_vel 파이프라인 동작 확인

접속:
  ssh nvidia@192.168.55.1  (USB-C 연결 시)
  또는 공유기 IP로 SSH
```

---

## 8. 다음 세션 시작점

```bash
# RunPod 재생성 후 — Phase 3 Fine-tuning
git clone https://github.com/vanillaturtlechips/MARS.git /workspace/MARS
bash /workspace/MARS/deploy/runpod/setup.sh

source /workspace/isaac_venv/bin/activate
cd /workspace/MARS

# Fine-tuning: True CTDE MAPPO + 개선된 보상 (rew_collision=-80, rew_stationary=-0.5)
python training/multi_robot/train_marl.py \
  --ippo_ckpt logs/warehouse_mappo/model_4999.pt \
  --num_envs 128 --max_iter 5000

# 훈련 완료 후 평가
python training/multi_robot/eval_scenarios.py \
  --ckpt logs/warehouse_mappo/model_9999.pt \
  --num_episodes 100 --tag mappo_finetuned
```

**현재 우선순위:**
1. RunPod 신규 Pod 생성 → `model_4999.pt` 체크포인트 복구 확인
2. Fine-tuning 실행 (rew_collision=-80 적용)
3. 5종 시나리오 재평가 → 전원도달률 60~70% 목표

---

*최종 업데이트: 2026-05-18*

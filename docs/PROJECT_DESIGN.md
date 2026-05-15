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
│   (벡터DB)                  A/B/C/D/E               │
│   - 창고 맵                                          │
│   - 임무 기억                                        │
│   - 행동 라이브러리                                   │
└──────────────────────────┬──────────────────────────┘
                           │ ROS2 / WebSocket
┌──────────────────────────▼──────────────────────────┐
│               LAYER 2 — 현장 두뇌 (Jetson Nano Super) │
│                                                     │
│   소형 LLM (Llama 3.2 3B)                           │
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

### 2-1. 단일 로봇 — PPO + Domain Randomization

```
목적함수:
  J(π) = 𝔼_τ~π [ Σ γᵗ r(sₜ, aₜ) ]

Sim-to-Real 해결:
  훈련 중 매 에피소드마다 랜덤화
  - 바닥 마찰:    μ ~ Uniform(0.3, 1.2)
  - 페이로드:     m ~ Uniform(0, 30kg)
  - 모터 지연:    d ~ Uniform(0, 20ms)
  - 센서 노이즈:  ε ~ N(0, 0.05)

Asymmetric Actor-Critic:
  훈련 시  Critic: 전체 상태 (위치, 속도, 힘 등) 사용
  배포 시  Actor:  카메라 + LiDAR 센서만 사용
```

### 2-2. 멀티 로봇 — MAPPO + Markov Potential Game

```
참고 논문: Yan & Liu, "Markov Potential Game Construction and MARL
           with Applications to Autonomous Driving" (arXiv 2503.22867, 2025)

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
    β = 0.5  (안전 가중치, 창고는 안전 우선이므로 높게 설정)

CTDE (Centralized Training, Decentralized Execution):
  훈련: 전체 로봇 상태 + Φ 함께 학습
  실행: 각 로봇은 자기 센서만 보고 독립 실행
```

### 2-3. World Model (Phase 4 이후 선택)

```
DreamerV3 방식:
  ŝ_{t+1} = f_θ(sₜ, aₜ)   상태 전이 예측
  r̂_t     = g_φ(sₜ, aₜ)   보상 예측

  훈련 가속:
    PhysX 시뮬 대신 f_θ로 "상상" 롤아웃
    → 실제 시뮬 대비 ~100x 샘플 효율
```

---

## 3. 서브 에이전트 설계

| 에이전트 | 역할 | 핵심 도구 |
|---------|------|---------|
| A. 재고 관리자 | 상품 위치 검색, 선반 현황 | RAG 벡터DB |
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

---

## 4. 개발 단계 (Phases)

### Phase 0 — 기반 환경 구축 ✅ 완료
- [x] Isaac Sim 5.1.0 + Isaac Lab 2.3.2 설치
- [x] NVIDIA Driver 580.142, CUDA 13.0
- [x] Cartpole PPO 훈련 (보상 0.10 → 4.38)
- [x] GUI 뷰어 동작 확인

---

### Phase 1 — 단일 로봇 이동
**목표:** 창고 환경에서 로봇 한 대가 목표 지점까지 자율 이동

```
Isaac Lab 커스텀 환경:
  - 선반 배치된 창고 맵
  - 휠 로봇 (4륜, Anymal 또는 커스텀)
  - 장애물 회피 포함

보상 설계:
  r = +10 (도착) - 0.01t (시간) - 5 (충돌)

Domain Randomization:
  바닥 마찰, 로봇 무게, 센서 노이즈
```

체크포인트:
- [ ] 창고 환경 USD 파일 제작
- [ ] 단일 로봇 PPO 훈련 (1024 병렬 env)
- [ ] 헤드리스 검증 → GUI 시각 확인

---

### Phase 2 — 단일 로봇 조작 (Pick & Place)
**목표:** 로봇이 박스를 집어서 지정 위치에 내려놓기

```
Isaac Lab 환경 확장:
  - 그리퍼 달린 로봇 암
  - 다양한 크기/무게 박스
  - 목표 선반 위치

보상 설계:
  r = +20 (박스 올바른 위치에 놓음)
    + 0.1 (박스 접근 중)
    - 10  (박스 낙하)

Asymmetric Actor-Critic:
  훈련 Critic: 박스 정확한 위치, 무게 알고 있음
  배포 Actor:  카메라로 추정만 가능
```

체크포인트:
- [ ] 그리퍼 로봇 Isaac Lab 환경
- [ ] Pick & Place PPO 훈련
- [ ] TorchScript export → Jetson 배포 테스트

---

### Phase 3 — 멀티 로봇 협력
**목표:** 로봇 3대가 동시에 서로 방해 없이 임무 수행

```
MARL 알고리즘: MAPPO (Multi-Agent PPO)
  - 공유 Critic (전체 상태)
  - 독립 Actor (각 로봇 자기 관측만)

Markov Potential Game 보상:
  위 2-2항 참조

훈련 환경:
  - 로봇 3대 동시 운용
  - 같은 통로 진입 시나리오 포함
  - 배터리 부족 시나리오 포함
```

체크포인트:
- [ ] MAPPO 환경 구성 (multi-agent Isaac Lab)
- [ ] 포텐셜 함수 보상 설계 및 검증
- [ ] 교착 상태 발생률 측정 (목표: < 1%)
- [ ] A2A 충돌 협상 프로토콜 구현

---

### Phase 4 — 에이전트 레이어 구축
**목표:** Claude API + 서브 에이전트가 임무를 자율 분해하고 로봇에게 전달

```
스택:
  오케스트레이터:  Claude API (claude-sonnet-4-6)
  프레임워크:      LangChain (프로토타입)
  RAG:            ChromaDB + 창고 지식베이스
  Jetson LLM:     Llama 3.2 3B (GGUF, 4-bit 양자화)
  통신:            ROS2 토픽

RAG 지식베이스:
  - 창고 선반 맵 (좌표 → 상품)
  - 임무 기록 (성공/실패 패턴)
  - 로봇 매뉴얼 (능력, 제한)
  - 행동 라이브러리 (검증된 시퀀스)
```

체크포인트:
- [ ] ChromaDB 창고 맵 구축
- [ ] LangChain 임무 분해 파이프라인
- [ ] 서브 에이전트 A~F 구현
- [ ] Jetson에서 Llama 3.2 3B 실행 확인

---

### Phase 5 — 통합 및 시뮬 검증
**목표:** 전체 파이프라인이 Isaac Sim 안에서 end-to-end 동작

```
시나리오 테스트:
  1. 입고: "박스 10개 3구역에 배치"
  2. 출고: "상품 A 5박스 → 2번 게이트"
  3. 장애: "Robot 2 배터리 부족 → 재배정"
  4. 안전: "5구역 사람 감지 → 전체 정지"
  5. 교착: "좁은 통로 동시 진입 → 자동 해소"

측정 지표:
  - 임무 완료율 (목표: > 95%)
  - 교착 발생률 (목표: < 1%)
  - 평균 임무 시간
  - Sim-to-Real 성능 갭 (Jetson 배포 후)
```

체크포인트:
- [ ] 5가지 시나리오 전부 통과
- [ ] 오케스트레이터 → Jetson → 로봇 레이턴시 측정
- [ ] 최종 시뮬 영상 기록

---

## 5. 기술 스택 요약

| 영역 | 기술 | 버전 |
|------|------|------|
| 시뮬레이터 | Isaac Sim | 5.1.0 |
| RL 프레임워크 | Isaac Lab | 2.3.2 |
| 단일 로봇 RL | rsl_rl (PPO) | 3.0.1 |
| 멀티 로봇 RL | MAPPO (커스텀) | - |
| 딥러닝 | PyTorch | 2.7.0+cu128 |
| 오케스트레이터 | Claude API | claude-sonnet-4-6 |
| 에이전트 프레임워크 | LangChain | 프로토타입 |
| RAG | ChromaDB | - |
| Jetson LLM | Llama 3.2 3B (GGUF) | - |
| 통신 | ROS2 Humble | - |
| 모델 배포 | TorchScript / ONNX | - |
| 엣지 디바이스 | Jetson Nano Super | 8GB |
| 훈련 하드웨어 | RTX 2070 Mobile | 8GB VRAM |

---

## 6. 프로젝트 디렉토리 구조

```
~/MARS/
│
├── envs/
│   └── warehouse/
│       ├── warehouse_env.py       # Isaac Lab 창고 환경
│       ├── warehouse_env_cfg.py   # 환경 설정
│       └── assets/                # USD 파일 (선반, 박스, 로봇)
│
├── training/
│   ├── single_robot/
│   │   ├── train_navigation.py
│   │   └── train_manipulation.py
│   └── multi_robot/
│       ├── train_marl.py          # MAPPO 훈련
│       └── potential_reward.py    # Markov Potential Game 보상
│
├── agents/
│   ├── orchestrator/
│   │   ├── orchestrator.py        # Claude API 오케스트레이터
│   │   └── task_planner.py        # LangChain 임무 분해
│   ├── sub_agents/
│   │   ├── inventory_agent.py     # 재고 관리
│   │   ├── path_agent.py          # 경로 계획
│   │   ├── robot_agent.py         # 로봇 담당
│   │   └── safety_agent.py        # 이상 감지
│   └── rag/
│       ├── knowledge_base.py      # ChromaDB 구축
│       └── warehouse_docs/        # 창고 지식 원본
│
├── deploy/
│   ├── export_model.py            # TorchScript/ONNX export
│   └── jetson/
│       ├── inference.py           # Jetson 추론 엔진
│       └── ros2_bridge.py         # ROS2 연결
│
├── logs/
│   └── cartpole_ppo/              # ✅ 완료 (~/ai-engineering-from-scratch/logs/)
│
├── test_cartpole_headless.py      # ✅ 완료
├── train_cartpole_ppo.py          # ✅ 완료
├── view_cartpole.py               # ✅ 완료
└── ~/docs/PROJECT_DESIGN.md       # 이 파일
```

---

## 7. 다음 세션 시작점

```bash
# 환경 활성화
export PATH="$HOME/.local/bin:$PATH"
source ~/ai-engineering-from-scratch/.venv-isaac/bin/activate

# 현재 완료 확인
python -c "import isaaclab; print('OK')"

# Phase 1 시작 — 창고 환경 제작
cd ~/MARS
# → envs/warehouse/warehouse_env.py 부터
```

**Jetson 조립 후 먼저 할 것:**
1. `deploy/export_model.py` — Cartpole 모델 ONNX export
2. Jetson에서 inference 동작 확인
3. Phase 1 병렬 진행

---

*최종 업데이트: 2026-05-15*

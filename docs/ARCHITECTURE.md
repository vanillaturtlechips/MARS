# MARS 시스템 아키텍처

**MARS (Multi-Agent Robot System)** — 물류 창고 자율 멀티 로봇 관제 AI Agent.

자연어 명령을 받아 여러 로봇에 작업을 분배하고, 각 로봇이 학습된 정책으로
자율 주행·조작하며, ROS2를 통해 실제 하드웨어로 연결되는 3-레이어 시스템.

---

## 3-레이어 아키텍처

```
┌─ ① Orchestrator 레이어 (Phase 4, LLM) ───────────────────────┐
│   전역 조율: 자연어 → task 분배, 우선순위, 교착 해소           │
│   - Claude API: 자연어 명령 해석 → 로봇별 목표 할당            │
│   - fleet 전역 상태 기반 우선순위 결정 (누가 먼저/대기)        │
│        │ /robot_i/goal_pose,  /robot_i/priority               │
│        ▼                          ▲ /fleet_status             │
├─ ② RL 정책 레이어 (Phase 1~3, per-robot) ────────────────────┤
│   로컬 반응: obs → action (학습된 정책, 반응적 회피·이동)     │
│   - 자기 obs만 관측 (goal·라이다·이웃 거리) → 전역 정보 없음  │
│        │ /robot_i/cmd_vel        ▲ obs (/odom, /scan, 이웃)   │
│        ▼                          │                           │
├─ ③ ROS2 제어/통신 레이어 ────────────────────────────────────┤
│   센서 수집 · 명령 실행 · 로봇 간 통신 · sim2real 연결점       │
│        │ 하드웨어 드라이버        ▲ 센서                       │
│        ▼                          │                           │
└─ 로봇 (Jetson 추론 + 구동부 + 라이다/odom) ──────────────────┘
```

### 레이어별 책임

| 레이어 | 역할 | 관측 범위 | 구현 |
|--------|------|-----------|------|
| ① Orchestrator | 자연어 해석, task 분배, **전역 우선순위·교착 조율** | fleet 전체 | Claude API (Phase 4) |
| ② RL 정책 | obs→action, **반응적 회피·이동** | 로봇 로컬 | PPO/MAPPO (Phase 1~3) |
| ③ ROS2 제어 | 센서·명령·통신, 하드웨어 연결 | 토픽 I/O | ROS2 Humble |

---

## 핵심 설계 원칙: RL(로컬) vs LLM(전역)의 책임 분리

**RL 정책(②)은 로컬 관측만 본다.** 자기 goal, 라이다 최근접 거리, 이웃 로봇
상대 위치만 입력받으므로 *반응적 회피*는 잘하지만, "여러 로봇 중 누가
양보할지" 같은 *전역 조율*은 구조적으로 불가능하다 — 그건 ① Orchestrator의 책임.

이 분리는 평가 결과로 명확히 드러난다 (eval_scenarios, 동적 장애물 ON):

| 시나리오 | 전원도달 | 성격 | 담당 레이어 |
|----------|---------|------|-------------|
| **S6 동적 장애물 회피** | **97%** | 로컬 반응 (장애물 우회) | ② RL ✅ |
| S2 3-way 교착 | 95% | 로컬 회피로 해소 가능 | ② RL ✅ |
| S5 혼잡 통로 | 80% | 로컬 회피 | ② RL ✅ |
| S1 정면 | 71% | 로컬 회피 | ② RL |
| **S3 좁은통로 우선순위** | **8%** | **전역 양보 결정 필요** | ① Orchestrator |
| **S4 동일목표 경쟁** | **0%** | **전역 task 재할당 필요** | ① Orchestrator |

→ S3/S4가 낮은 것은 RL의 *실패*가 아니라 **레이어 책임 밖**이다.
"좁은 통로에서 누가 먼저 갈지(S3)", "같은 목표에 둘이 갈 때 재배정(S4)"은
전역 상태를 보는 Orchestrator가 `/priority`·`/goal_pose`로 조율할 문제.

---

## ROS2 토픽 맵 (레이어 경계 인터페이스)

| 경계 | 토픽 | 메시지 | 설명 |
|------|------|--------|------|
| ①→② | `/robot_i/goal_pose` | PoseStamped | 목표 위치 할당 |
| ①→② | `/robot_i/priority` | (hold/go) | 우선순위·교착 조율 |
| 센서→② | `/robot_i/odom` | Odometry | 자기 pose·velocity |
| 센서→② | `/robot_i/scan` | LaserScan | **장애물 최근접 거리 (정적+동적 통합)** |
| ②통신 | `/robot_j/odom` | Odometry | 이웃 로봇 상대 위치 |
| ②→③ | `/robot_i/cmd_vel` | Twist | 속도 명령 (vx, vy, ω) |
| ③→① | `/robot_i/status` | (도달/교착/배터리) | fleet 피드백 |

### sim2real 정합 포인트: `/scan` ↔ obs의 `shelf_dist`

RL obs의 장애물 채널(`shelf_dist`)은 시뮬에서 **정적 선반 + 동적 장애물의
min 거리**로 계산된다. 실로봇에서는 **라이다 `/scan`의 최근접 거리**가 정확히
같은 의미 — 라이다는 정적/동적을 구분 없이 "가장 가까운 장애물"을 잡으므로,
시뮬 설계가 자연스럽게 실로봇으로 이어진다 (sim2real 갭 최소화).

---

## 학습 산출물 현황

| Phase | 환경 | 모델 | 성능 |
|-------|------|------|------|
| 1 / 1.5 | 단일 로봇 nav + 장애물 회피 | `actor_phase15.pt` | 배포됨 |
| 2 | Pick & Place (Franka, IK) | `warehouse_pickplace/model_300.pt` | **place 99.5%** |
| 3 | MARL 3로봇 (MAPPO) | `warehouse_mappo/model_9999.pt` | S1·S3 회피 |
| 3+ | + 동적 장애물 회피 (fine-tune) | `warehouse_mappo/model_10998.pt` | **S6 97%** |
| 4 | Orchestrator (LLM) | `agents/` (stub) | 설계 단계 |

### 향후 과제

- **배터리/충전소** — env(obs 20D·충전소·충전/방전 보상)는 구현 완료(`enable_battery`),
  학습은 향후 과제. obs 차원 확장(17→20D)으로 actor 입력층·critic이 새로 초기화되어
  fine-tune이 scratch 수준이 되는 게 난점. 본 프로젝트 fine-tune 성공 사례(동적 장애물)는
  모두 입력층 보존(obs 차원 유지)이 전제였음. 해결엔 충분한 scratch 재학습 또는
  critic transfer 구조가 필요.

배포: `deploy/jetson/` (actor_phase15/phase2_final/phase3_marl.pt) + `ros2_bridge.py`

---

## 데이터 흐름 (end-to-end)

```
자연어 명령 ("A구역 박스를 B로 옮겨")
   │
   ▼ ① Orchestrator (Claude API)
task 분해 → 로봇별 goal + 우선순위
   │  /robot_i/goal_pose, /robot_i/priority
   ▼ ② RL 정책 (per-robot, Jetson 추론)
obs(/odom + /scan + 이웃) → action
   │  /robot_i/cmd_vel
   ▼ ③ ROS2 제어 → 구동부
로봇 이동 (회피·조작)
   │  /robot_i/status (도달/교착/배터리)
   └─▶ ① Orchestrator 피드백 (재조율)
```

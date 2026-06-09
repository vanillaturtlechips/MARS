# MARS 시스템 아키텍처

> ⚠️ **2026-06-09 경고:** 이 문서의 **S1~S6 성능표(S6 97·S3 8·S4 0 등)는 오염됨 —
> 어떤 커밋된 eval json과도 안 맞고 S6는 obstacles OFF로 잰 무효 숫자다. 인용 금지.**
> 검증된 숫자·현황은 [`docs/STATUS.md`](STATUS.md)를 봐라. 아래는 **레이어 설계 *개념***
> 참고용으로만.

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

### 배터리/충전 — Orchestrator 영역 (RL 통합 폐기)

충전 의사결정("언제/어느 충전소로")은 **전역 상태 기반 고수준 결정**이라 ① Orchestrator
영역이다 — 충전소(2개) < 로봇(3개) 공유자원 경합, fleet 배터리·충전소 점유가 필요하며,
이는 S3(좁은통로 우선순위)·S4(동일목표)가 orchestrator 영역인 것과 **구조적으로 동일**하다.

RL에 충전을 통합 시도했으나(obs 20D + 충전 보상), 충전은 학습됐지만(평균배터리 0.7+)
navigation/회피가 붕괴(S1/S2/S4 충돌 100%)하고 std가 발산했다. 원인은 **로컬 obs인 RL에
전역 의사결정을 떠넘긴 것** — reward 함수가 충전 vs 회피 충돌을 막는 땜질 코드로 뒤덮였다.

**올바른 설계:** RL(②)은 navigation+회피만 한다(`model_10998`, S6 97%, obs 17D — 검증됨).
Orchestrator(①)가 배터리(`/robot_i/status`)를 보고 충전소를 `/robot_i/goal_pose`로 발행하면,
RL은 충전소인지도 모른 채 그냥 navigation한다. env의 충전 메커니즘(충전소 도달 시 회복)은
유지하되 RL 학습엔 넣지 않는다(`enable_battery=False`). 이로써 멀티task·obs 확장 문제가
원천 소멸하고, "RL이 못 하는 전역 결정을 LLM이 한다"는 레이어 분리 원칙이 일관되게 적용된다.

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

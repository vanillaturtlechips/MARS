# 세션 진행 현황 (2026-05-18)

## 완료된 작업

### Deploy 스크립트
- `deploy/export_model.py` — rsl_rl 체크포인트 → TorchScript export
  - 버그 수정: actor 키 remapping (`actor.0.weight` → `net.0.weight`)
- `deploy/jetson/inference.py` — TorchScript 추론 엔진 + latency 측정
- `deploy/jetson/ros2_bridge.py` — `/odom` + `/goal_pose` → `/cmd_vel` (15Hz)
- `deploy/jetson/benchmark_llm.py` — ollama REST API 벤치마킹
- `deploy/runpod/setup.sh` — RunPod 원클릭 설치 스크립트
- `deploy/runpod/RUNPOD_GUIDE.md` — 포트/livestream/TensorBoard 설정 문서화

### Phase 3 최종 완료 ✅

**model_9999.pt — Low-level Controller 확정 (Freeze)**

| 시나리오 | 충돌률 | 교착률 | 전원도달 |
|---------|:------:|:------:|:-------:|
| S1 정면 충돌 | 0% | 7% | **93%** |
| S2 3-way 교착 | 0% | 16% | **84%** |
| S3 통로 우선 | 1% | 4% | **95%** |
| S4 동일 목표 | 0% | 100% | 0% |
| S5 혼합 장애물 | 0% | 100% | 0% |
| **S1~S3 평균** | **0.3%** | **9%** | **90.7%** |
| **전체 평균** | **0.2%** | **45.4%** | **54.4%** |

**핵심 성과:**
- 충돌률 **0.2%** — 목표 <1% ✅ 달성
- S1~S3 전원도달 **90.7%** — 실무 배포 가능 수준
- S4: Phase 4 목표 배정(LLM 오케스트레이터)으로 해결 예정
- S5: Phase 4 교착 감지 + 재라우팅으로 해결 예정

**훈련 구성 (최종 확정):**
- obs: 17-dim (상대속도 vx_rel/vy_rel 추가)
- rew_collision: -200
- SAFE_DIST: 2.5m
- W_REP: 1.5
- 8192 envs × 3 robots, 10000 iter
- 훈련 시간: 약 3.5시간 (RunPod A6000)

**보상 엔지니어링 이력:**
| 문제 | 원인 | 수정 |
|------|------|------|
| VF loss 폭발 (1329) | rew_time*t 2차 누적 | 상수 -0.01/step |
| death exploitation | rew_collision 미적용 | -150 penalty (IPPO) |
| camping local optimum | rew_goal=10 연속 | 10→3 축소 |
| noise_std 발산 | entropy_coef 지배 | 0.01→0.001 |
| S2/S4 즉시 충돌 | obs에 상대속도 없음 | 9→17dim + SAFE_DIST 2.5m |
| 자살 학습 | rew_collision 너무 약함 | -200 (교착 -90보다 110점 낮게) |

**SAFE_DIST 1.8m 실험 결과 (실패, model_11998.pt):**
- S3 95% → 1% 붕괴 — 좁은 통로에서 역효과
- model_9999.pt(2.5m)가 최종 확정

### Phase 2 코드 (⚠️ 재훈련 필요)
- `envs/warehouse/warehouse_manipulation_env.py` — Franka Panda Pick & Place
  - Teacher 관측 33차원 (특권 정보), Student 관측 25차원 (실제 센서)
  - 박스 크기/질량 DR 적용
- `training/single_robot/train_manipulation.py` — Teacher PPO
- `training/single_robot/eval_manipulation.py` — place_success_rate 평가 스크립트

**Phase 2 훈련 결과 (model_2999.pt) — 무효**
- 3000 iter 완료, 256 envs, RunPod A6000
- eval 결과: place 100%, avg_len 2.0 → **trivial success 버그**

**Trivial Success 버그 원인 (2가지)**
1. **Env 설계 버그**: EE ready pose ≈ (0.4, 0, 0.5)m가 box 스폰 범위(x∈[0.3,0.6]) 및 PLACE_GOALS 4개 모두의 grasp/place threshold 이내
   - `grasp_dist_threshold=0.25m` → EE가 reset 시점에 이미 grasp 성공
   - `place_dist_threshold=0.35m` → 4개 goal 모두 EE 시작점에서 0.35m 이내
2. **Obs 정규화 누락**: 훈련 시 `empirical_normalization=True`, eval 시 normalization stats 미포함 → actor 입력 스케일 불일치 → near-zero actions

**Trivial Success 버그 수정 완료 (2026-05-19)**
- `grasp_dist_threshold: 0.25 → 0.06m` ✅
- `place_dist_threshold: 0.35 → 0.08m` ✅
- 박스 스폰 범위: x∈[0.3,0.6] → x∈[0.55,0.75] (EE 시작점에서 멀리) ✅
- `empirical_normalization=False` ✅

### Jetson 완료
- PyTorch 2.8.0 + CUDA 설치 (cuSPARSELt 0.7.0, cuDSS 0.7.1.4)
- ROS2 Humble 설치
- ollama + qwen2.5:3b-instruct-q4_K_M 설치 (채택)
- `actor_phase15.pt` export 및 복사 완료

### Jetson 벤치마킹 결과
| 항목 | llama3.2:3b | qwen2.5:3b |
|------|------------|------------|
| tokens/sec | 21.1 | **21.5** ✅ |
| 첫 토큰 레이턴시 | 663ms | **640ms** ✅ |
| JSON 성공률 | 100% | **100%** ✅ |

→ **qwen2.5:3b-instruct-q4_K_M 채택**

- RL policy inference: **0.33ms = 3034 Hz** (목표 100Hz의 30배)
- ros2_bridge.py: `/goal_pose` → `/cmd_vel` 파이프라인 동작 확인

---

## RunPod 재생성 절차

Pod 삭제/GPU 회수 후 재생성 시:

```bash
# 새 Pod 생성 시 필수: 포트 8211(livestream), 6006(TensorBoard) 사전 추가
# → RUNPOD_GUIDE.md 참고

git clone https://github.com/vanillaturtlechips/MARS.git /workspace/MARS
bash /workspace/MARS/deploy/runpod/setup.sh
```

---

## 전체 남은 작업

| 항목 | 상태 |
|------|------|
| Phase 3 Low-level Controller 확정 (model_9999.pt) | ✅ 완료 |
| Phase 4 에이전트 레이어 (오케스트레이터) | 다음 단계 |
| Phase 4 S4 목표 충돌 → LLM 목표 배정으로 해결 | Phase 4 |
| Phase 4 S5 교착 → 교착 감지 + 재라우팅으로 해결 | Phase 4 |
| Phase 2 Env 버그 수정 + 재훈련 | 다음 세션 |
| Phase 2 Teacher-Student 증류 | 재훈련 이후 |
| Phase 5 통합 테스트 | Phase 3/4 완료 후 |

---

## 파일 위치 요약

```
MARS/
├── logs/
│   ├── warehouse_nav/model_999.pt              # Phase 1 ✅
│   ├── warehouse_obstacle_nav/model_100.pt     # Phase 1.5 ✅
│   ├── warehouse_ippo/model_400.pt             # Phase 3 IPPO ✅
│   └── warehouse_mappo/model_9999.pt           # Phase 3 최종 ✅ (Freeze)
├── deploy/
│   ├── export_model.py                         # ✅
│   ├── jetson/
│   │   ├── actor_phase15.pt                   # ✅ Jetson에 복사됨
│   │   ├── inference.py                       # ✅ 0.33ms 확인
│   │   ├── ros2_bridge.py                     # ✅ 동작 확인
│   │   └── benchmark_llm.py                  # ✅
│   └── runpod/
│       ├── setup.sh                           # ✅ 원클릭 설치
│       └── RUNPOD_GUIDE.md                   # ✅ 포트/livestream 가이드
├── envs/warehouse/
│   ├── warehouse_env.py                       # Phase 1 ✅
│   ├── warehouse_obstacle_env.py              # Phase 1.5 ✅
│   ├── warehouse_manipulation_env.py          # Phase 2 코드 ✅ (미훈련)
│   ├── warehouse_marl_env.py                 # Phase 3 ✅
│   └── ippo_wrapper.py                       # Phase 3 ✅ rsl_rl 3.x 호환
└── training/
    ├── single_robot/train_manipulation.py      # Phase 2
    └── multi_robot/
        ├── potential_reward.py                # MPG 보상 ✅ (SAFE_DIST=2.5)
        ├── train_ippo.py                     # Phase 3 IPPO ✅
        ├── train_marl.py                    # Phase 3 True CTDE MAPPO ✅
        ├── eval_scenarios.py                # 5종 시나리오 평가 ✅
        └── demo_play.py                     # USD 에셋 데모 ✅

GitHub: github.com/vanillaturtlechips/MARS (main)
Jetson: ssh nvidia@192.168.55.1 (USB-C)
RunPod: A6000, /workspace/isaac_venv
```

---

*최종 업데이트: 2026-05-18 — Phase 2 trivial success 버그 발견, 다음 세션 재훈련 예정*

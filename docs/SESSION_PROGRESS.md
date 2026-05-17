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

### Phase 3 True CTDE MAPPO 완료

**구현 완료:**
- `envs/warehouse/warehouse_marl_env.py` — 3대 멀티로봇 환경
  - 선반 장애물 + 로봇 간 충돌 감지 (0.55m)
  - **rew_collision=-80** (greedy rush 억제, S2/S4 즉시 충돌 방지)
  - **rew_stationary=-0.5** (S5 교착 완화)
  - alpha=1.0, beta=0.5 (MPG 튜닝 완료)
- `training/multi_robot/potential_reward.py` — MPG 보상 (arXiv 2503.22867)
  - Non-Markovian time penalty 수정: `rew_time*t` → 상수 -0.01/step
  - Danger Zone 마스킹: SAFE_DIST=1.2m 밖에서는 pairwise 보상 0
- `training/multi_robot/train_ippo.py` — entropy_coef 0.01→0.001
- `training/multi_robot/train_marl.py` — MAPPO (True CTDE)
- `training/multi_robot/eval_scenarios.py` — 5종 시나리오 평가 스크립트
- `envs/warehouse/ippo_wrapper.py` — `get_observations()` TensorDict 단독 반환 (rsl_rl 3.x)

**Phase 3 훈련 및 평가 결과 (PHASE3_IPPO_VS_MAPPO.md 참고):**

| 모델 | 전원도달률 | 교착률 | 충돌률 |
|------|:-------:|:------:|:------:|
| IPPO (model_400.pt) | 20% | 20% | 60% |
| 가짜 CTDE MAPPO (model_5399.pt) | 0% | 80% | 20% |
| **True CTDE MAPPO (model_4999.pt)** | **40%** | **20%** | **40%** |

**보상 엔지니어링 이력:**
| 문제 | 원인 | 수정 |
|------|------|------|
| VF loss 폭발 (1329) | rew_time*t 2차 누적 | 상수 -0.01/step |
| death exploitation | rew_collision 미적용 | -150 penalty (IPPO) |
| camping local optimum | rew_goal=10 연속 | 10→3 축소 |
| noise_std 발산 | entropy_coef 지배 | 0.01→0.001 |
| S2/S4 즉시 충돌 | rew_collision=-25 너무 약함 | **-25 → -80** |
| S5 교착 | 정지 패널티 부족 | **-0.3 → -0.5** |

### Phase 2 코드
- `envs/warehouse/warehouse_manipulation_env.py` — Franka Panda Pick & Place
  - Teacher 관측 33차원 (특권 정보), Student 관측 25차원 (실제 센서)
  - 박스 크기/질량 DR 적용
- `training/single_robot/train_manipulation.py` — Teacher PPO + Student fine-tuning

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

설치 후 Fine-tuning (True CTDE MAPPO → 개선된 보상):
```bash
source /workspace/isaac_venv/bin/activate
cd /workspace/MARS

# Phase 3 Fine-tuning (model_4999.pt에서 시작, 개선된 보상 파라미터 적용)
# rew_collision=-80, rew_stationary=-0.5 (warehouse_marl_env.py에 반영됨)
python training/multi_robot/train_marl.py \
  --ippo_ckpt logs/warehouse_mappo/model_4999.pt \
  --num_envs 128 --max_iter 5000

# 훈련 완료 후 평가
python training/multi_robot/eval_scenarios.py \
  --ckpt logs/warehouse_mappo/model_9999.pt \
  --num_episodes 100 --tag mappo_finetuned
```

---

## 전체 남은 작업

| 항목 | 상태 |
|------|------|
| Phase 3 Fine-tuning (rew_collision=-80) | RunPod 생성 후 즉시 실행 |
| Phase 3 전원도달률 40%→60~70% 목표 | Fine-tuning 완료 후 |
| Phase 3 A2A 충돌 협상 프로토콜 | 다른 팀원 담당 |
| Phase 2 Teacher PPO 훈련 | 추후 진행 |
| Phase 2 Teacher-Student 증류 | 위 이후 |
| Phase 4 에이전트 레이어 | 다른 팀원 담당 |
| Phase 5 통합 테스트 | Phase 3/4 완료 후 |

---

## 파일 위치 요약

```
MARS/
├── logs/
│   ├── warehouse_nav/model_999.pt              # Phase 1 ✅
│   ├── warehouse_obstacle_nav/model_100.pt     # Phase 1.5 ✅
│   ├── warehouse_ippo/model_400.pt             # Phase 3 IPPO ✅ (RunPod)
│   └── warehouse_mappo/model_4999.pt           # Phase 3 True CTDE MAPPO ✅ (RunPod)
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
│   ├── warehouse_marl_env.py                 # Phase 3 ✅ (rew_collision=-80)
│   └── ippo_wrapper.py                       # Phase 3 ✅ rsl_rl 3.x 호환
└── training/
    ├── single_robot/train_manipulation.py      # Phase 2
    └── multi_robot/
        ├── potential_reward.py                # MPG 보상 ✅
        ├── train_ippo.py                     # Phase 3 IPPO ✅
        ├── train_marl.py                    # Phase 3 True CTDE MAPPO ✅
        ├── eval_scenarios.py                # 5종 시나리오 평가 ✅
        └── demo_play.py                     # USD 에셋 데모 ✅

GitHub: github.com/vanillaturtlechips/MARS (main)
Jetson: ssh nvidia@192.168.55.1 (USB-C)
RunPod: RTX 3090, /workspace/isaac_venv
```

---

*최종 업데이트: 2026-05-18*

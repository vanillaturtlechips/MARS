# 세션 진행 현황 (2026-05-16)

## 완료된 작업

### Deploy 스크립트
- `deploy/export_model.py` — rsl_rl 체크포인트 → TorchScript export
  - 버그 수정: actor 키 remapping (`actor.0.weight` → `net.0.weight`)
- `deploy/jetson/inference.py` — TorchScript 추론 엔진 + latency 측정
- `deploy/jetson/ros2_bridge.py` — `/odom` + `/goal_pose` → `/cmd_vel` (15Hz)
- `deploy/jetson/benchmark_llm.py` — ollama REST API 벤치마킹
- `deploy/runpod/setup.sh` — RunPod 원클릭 설치 스크립트
- `deploy/runpod/RUNPOD_GUIDE.md` — 포트/livestream/TensorBoard 설정 문서화

### Phase 3 코드 + 보상 엔지니어링 완료
- `envs/warehouse/warehouse_marl_env.py` — 3대 멀티로봇 환경
  - 선반 장애물 + 로봇 간 충돌 감지 (0.55m)
  - rew_collision=-150 적용 (death exploitation 차단)
  - alpha=1.0, beta=0.5 (MPG 튜닝 완료)
- `training/multi_robot/potential_reward.py` — MPG 보상 (arXiv 2503.22867)
  - Non-Markovian time penalty 수정: `rew_time*t` → 상수 -0.01/step
  - Danger Zone 마스킹: SAFE_DIST=1.2m 밖에서는 pairwise 보상 0
  - rew_goal: 10.0 → 3.0 (camping local optimum 해소)
- `training/multi_robot/train_ippo.py` — entropy_coef 0.01→0.001 (noise std 발산 억제)
- `training/multi_robot/train_marl.py` — MAPPO (Asymmetric Actor-Critic, 미훈련)
- `training/multi_robot/demo_play.py` — USD 에셋 시각화 데모
  - iw_hub 로봇 + full_warehouse 환경 (ISAAC_NUCLEUS_DIR 자동 캐시)
  - try/except fallback (cuboid)
  - num_envs=1, noise_std=0.01 (결정론적)

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

## Phase 3 IPPO 훈련 결과 (RunPod RTX 3090)

**400 iter 기준:**
- noise_std: 1.00 → 0.56 (수렴 중, 감소세)
- episode length: 80-110 → 220-280 (생존 학습됨)
- mean reward: 50-115 (plateau — IPPO 구조적 한계, MAPPO로 돌파 예정)
- 체크포인트: `logs/warehouse_ippo/model_400.pt`

**보상 엔지니어링 이력:**
| 문제 | 원인 | 수정 |
|------|------|------|
| VF loss 폭발 (1329) | rew_time*t 2차 누적 | 상수 -0.01/step |
| death exploitation | rew_collision 미적용 | -150 penalty 추가 |
| camping local optimum | rew_goal=10 연속 | 10→3 축소 |
| noise_std 발산 | entropy_coef 지배 | 0.01→0.001 |

---

## RunPod 재생성 절차

Pod 삭제 후 재생성 시 `/workspace` 볼륨은 유지되면 재사용, 없으면 아래 실행:

```bash
# 새 Pod 생성 시 필수: 포트 8211(livestream), 6006(TensorBoard) 사전 추가
# → RUNPOD_GUIDE.md 참고

git clone https://github.com/vanillaturtlechips/MARS.git /workspace/MARS
bash /workspace/MARS/deploy/runpod/setup.sh
```

설치 후 훈련:
```bash
source /workspace/isaac_venv/bin/activate
cd /workspace/MARS

# Phase 3 MAPPO (IPPO 체크포인트에서 이어받기)
python training/multi_robot/train_marl.py \
  --headless --num_envs 256 \
  --checkpoint logs/warehouse_ippo/model_400.pt

# Phase 2 Teacher PPO
python training/single_robot/train_manipulation.py \
  --headless --num_envs 512
```

---

## 전체 남은 작업

| 항목 | 상태 |
|------|------|
| Phase 2 Teacher PPO 훈련 | 내일 진행 |
| Phase 2 Teacher-Student 증류 | 위 이후 |
| Phase 3 MAPPO 훈련 | 내일 진행 (IPPO model_400.pt 이어받기) |
| Phase 3 IPPO vs MAPPO 5종 비교 | MAPPO 완료 후 |
| Phase 3 demo_play.py 테스트 | 나중에 |
| Phase 4 에이전트 레이어 | 다른 분 담당 |
| Phase 5 통합 테스트 | Phase 3/4 완료 후 |

---

## 파일 위치 요약

```
MARS/
├── logs/
│   ├── warehouse_nav/model_999.pt          # Phase 1 ✅
│   ├── warehouse_obstacle_nav/model_100.pt # Phase 1.5 ✅
│   └── warehouse_ippo/model_400.pt         # Phase 3 IPPO ✅ (RunPod에 있음)
├── deploy/
│   ├── export_model.py                     # ✅
│   ├── jetson/
│   │   ├── actor_phase15.pt               # ✅ Jetson에 복사됨
│   │   ├── inference.py                   # ✅ 0.33ms 확인
│   │   ├── ros2_bridge.py                 # ✅ 동작 확인
│   │   └── benchmark_llm.py              # ✅
│   └── runpod/
│       ├── setup.sh                       # ✅ 원클릭 설치
│       └── RUNPOD_GUIDE.md               # ✅ 포트/livestream 가이드
├── envs/warehouse/
│   ├── warehouse_env.py                   # Phase 1 ✅
│   ├── warehouse_obstacle_env.py          # Phase 1.5 ✅
│   ├── warehouse_manipulation_env.py      # Phase 2 코드 ✅ (미훈련)
│   └── warehouse_marl_env.py             # Phase 3 코드 ✅
└── training/
    ├── single_robot/train_manipulation.py  # Phase 2
    └── multi_robot/
        ├── potential_reward.py            # MPG 보상 ✅
        ├── train_ippo.py                 # Phase 3 IPPO ✅ (400 iter 완료)
        ├── train_marl.py                # Phase 3 MAPPO (미훈련)
        └── demo_play.py                 # USD 에셋 데모 ✅

GitHub: github.com/vanillaturtlechips/MARS (main)
Jetson: ssh nvidia@192.168.55.1 (USB-C)
RunPod: RTX 3090, /workspace/isaac_venv
```

---

*최종 업데이트: 2026-05-16*

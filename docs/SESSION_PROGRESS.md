# 세션 진행 현황 (2026-05-16)

## 완료된 작업

### Deploy 스크립트
- `deploy/export_model.py` — rsl_rl 체크포인트 → TorchScript export
  - 버그 수정: actor 키 remapping (`actor.0.weight` → `net.0.weight`)
- `deploy/jetson/inference.py` — TorchScript 추론 엔진 + latency 측정
- `deploy/jetson/ros2_bridge.py` — `/odom` + `/goal_pose` → `/cmd_vel` (15Hz)
- `deploy/jetson/benchmark_llm.py` — ollama REST API 벤치마킹
- `deploy/runpod/setup.sh` — RunPod 원클릭 설치 스크립트

### Phase 2/3 코드
- `envs/warehouse/warehouse_manipulation_env.py` — Franka Panda Pick & Place
  - Teacher 관측 33차원 (특권 정보), Student 관측 25차원 (실제 센서)
  - 박스 크기/질량 DR 적용
- `envs/warehouse/warehouse_marl_env.py` — 3대 멀티로봇 환경
  - 선반 장애물 + 로봇 간 충돌 감지 (0.55m)
- `training/single_robot/train_manipulation.py` — Teacher PPO + Student fine-tuning
- `training/multi_robot/potential_reward.py` — MPG 보상 (arXiv 2503.22867)
- `training/multi_robot/train_ippo.py` — IPPO 베이스라인
- `training/multi_robot/train_marl.py` — MAPPO (Asymmetric Actor-Critic)

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

## 현재 진행 중 — RunPod RTX 3090

### 상황
setup.sh 실행 완료 (Isaac Sim 5.1.0 + Isaac Lab 2.3.2 + venv 설치됨)

### 남은 오류 (train_ippo.py 실행 시)
```
1. libXt.so.6 없음 → CUDA/Vulkan 초기화 실패
2. rsl_rl 모듈 없음
```

### 다음에 할 것 (RunPod)
```bash
# 1. 시스템 라이브러리 설치
apt-get update -q && apt-get install -y \
    libxt6 libxrandr2 libgl1-mesa-glx libglu1-mesa \
    libvulkan1 libegl1 libgles2

# 2. rsl_rl 설치
pip install rsl-rl

# 3. 훈련 실행
export CUDA_VISIBLE_DEVICES=0
cd /workspace/MARS
python training/multi_robot/train_ippo.py --headless --num_envs 256
```

### setup.sh에 추가 필요
위 apt-get + rsl-rl 설치를 setup.sh에 넣어야 함 (다음 세션에서).

---

## 전체 남은 작업

| 항목 | 상태 |
|------|------|
| Phase 2 Teacher PPO 훈련 | RunPod에서 진행 예정 |
| Phase 2 Teacher-Student 증류 | 위 이후 |
| Phase 3 IPPO 베이스라인 | RunPod에서 진행 예정 |
| Phase 3 MAPPO 비교 | 위 이후 |
| Phase 4 에이전트 레이어 | 다른 분 담당 |
| Phase 5 통합 테스트 | Phase 3/4 완료 후 |

---

## 파일 위치 요약

```
MARS/
├── logs/
│   ├── warehouse_nav/model_999.pt          # Phase 1 ✅
│   └── warehouse_obstacle_nav/model_100.pt # Phase 1.5 ✅
├── deploy/
│   ├── export_model.py                     # ✅
│   ├── jetson/
│   │   ├── actor_phase15.pt               # ✅ Jetson에 복사됨
│   │   ├── inference.py                   # ✅ 0.33ms 확인
│   │   ├── ros2_bridge.py                 # ✅ 동작 확인
│   │   └── benchmark_llm.py              # ✅
│   └── runpod/
│       └── setup.sh                       # ✅ (rsl-rl 추가 필요)
├── envs/warehouse/
│   ├── warehouse_env.py                   # Phase 1 ✅
│   ├── warehouse_obstacle_env.py          # Phase 1.5 ✅
│   ├── warehouse_manipulation_env.py      # Phase 2 (미훈련)
│   └── warehouse_marl_env.py             # Phase 3 (미훈련)
└── training/
    ├── single_robot/train_manipulation.py  # Phase 2
    └── multi_robot/
        ├── potential_reward.py            # MPG 보상
        ├── train_ippo.py                 # Phase 3 베이스라인
        └── train_marl.py                # Phase 3 MAPPO

GitHub: github.com/vanillaturtlechips/MARS (main)
Jetson: ssh nvidia@192.168.55.1 (USB-C)
RunPod: RTX 3090, /workspace/isaac_venv
```

---

*작성: 2026-05-16*

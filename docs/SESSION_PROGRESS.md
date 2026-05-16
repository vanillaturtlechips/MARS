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

## RunPod RTX 3090 — 현재 상태

### setup.sh 수정 완료 (2026-05-16)
- [0/6] apt-get: libxt6, libvulkan1, libgl1-mesa-glx 등 추가
- [3b/6] pxr 경로 설정 3단계 추가:
  1. sitecustomize.py → Python 시작 시 `import isaacsim` 자동 실행
  2. find로 pxr 폴더 탐색 → pxr_path.pth 생성
  3. 미발견 시 `usd-core` pip 설치 (fallback)
- isaaclab_rl 의존성 추가: tensorboard, gymnasium, onnx
- 검증 단계에 pxr / isaaclab_rl import 확인 추가

### 훈련 스크립트 import 수정 완료
```python
# 이전 (잘못된 경로)
from rsl_rl.runners import OnPolicyRunner
from envs.warehouse.agents.rsl_rl_ppo_cfg import RslRlPpoActorCriticCfg, RslRlOnPolicyRunnerCfg

# 수정 후
from isaaclab_rl.rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
```
- train_ippo.py ✅
- train_marl.py ✅
- train_manipulation.py ✅

### 다음 RunPod 접속 시 할 것
```bash
# venv가 이미 설치됐으면:
source /workspace/isaac_venv/bin/activate
cd /workspace/MARS
git pull origin main   # 수정된 스크립트 가져오기
export CUDA_VISIBLE_DEVICES=0
python training/multi_robot/train_ippo.py --headless --num_envs 256

# 새 RunPod이면:
bash deploy/runpod/setup.sh
```

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

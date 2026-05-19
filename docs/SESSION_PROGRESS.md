# 세션 진행 현황 (2026-05-19)

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

### Phase 2 Teacher 완료 ✅ (2026-05-19 세션)

**model_2999.pt — Teacher Policy 확정**

| 체크포인트 | place rate | avg_len |
|-----------|-----------|---------|
| model_2400 | 1.4% | 898 |
| model_2700 | 100% | 352 |
| **model_2999** | **100%** | **183** |

- 100% place rate, avg 183 스텝 (≈3초) 만에 pick & place 완료

---

#### 이번 세션 버그 수정 이력 (2026-05-19 오전 세션 이어서)

| # | 커밋 | 문제 | 원인 | 수정 |
|---|------|------|------|------|
| 1 | `ea987c1` | eval 전 체크포인트 전부 0% place | box spawn x=[0.60,0.75] → Franka 최대 도달거리(0.855m) 초과 | threshold 0.40m, box x=[0.50,0.65] |
| 2 | `9b3d73f` | place 100%, avg_len=2 (trivial success) | box spawn x=[0.50,0.65]이 PLACE_GOALS x=[0.40,0.50]과 겹침 | PLACE_GOALS를 y=±0.32-0.35 측면으로 이동, box x=[0.45,0.55] |
| 3 | `fb3c5de` | iter 1085 ep_len=895 고착, noise_std=0.61 | approach≈transport(그라스프 경계) → hover 로컬옵티멈 | rew_approach 5→3, rew_grasp 10→30, rew_transport 5→10 |
| 4 | `7dc4522` | iter 2999 ep_len=880, noise_std=1.34 발산 | transport=10×exp(-0.35×0.5)×799=6,711 >> place=20 → grasp 후 무한 hover | rew_approach 3→0.5, rew_transport 10→1, rew_place 20→800 |
| 5 | `af45f1b` | iter 898 ep_len=770 고착, 학습 zero | "늦은 place(1,430pt) > 빠른 place(980pt)" — 절대거리 Exp가 시간당 보상 누적 | transport: Exp(-dist) → Progress Delta clamp(-0.1,0.1)×100 (성과급 방식) |

**Progress Delta 수정 후 즉시 수렴:**
- iter 312: ep_len=575, noise_std=0.77 (이전 run iter 899에서도 못 본 수치)
- iter 726: ep_len=520, noise_std=0.53
- iter 1811: ep_len=565, noise_std=0.41 (수렴)
- **iter 2700~2999: place rate 100%**

---

#### 최종 확정 환경 설정

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| action_space | 4 | [dx, dy, dz, gripper] Cartesian delta |
| max step | 3cm/step | DLS IK λ=0.01 |
| stiffness | 400 N·m/rad | |
| damping | 40 | |
| box 스폰 | x∈[0.45,0.55], y∈[-0.15,0.15] | Franka 최대 도달거리 89% |
| PLACE_GOALS | y=±0.32-0.35 (측면 4곳) | box spawn과 겹침 없음 |
| `grasp_dist_threshold` | 0.25m | |
| `place_dist_threshold` | 0.12m | |
| `rew_approach` | 0.5 × exp(-dist×5.0) | decay=5.0 중거리 hover 억제 |
| `rew_grasp` | 30.0 | 단발 보너스 |
| `rew_transport` | 10.0 × delta.clamp(-0.1,0.1) × 100 | Progress Delta (성과급) |
| `rew_place` | 800.0 | 대형 터미널 보상 |
| `rew_time` | -0.02/step | |
| Teacher obs | 30-dim | box_rel+quat+mass+gripper+goal_rel+jpos+jvel |

---

### Phase 2 Student 진행 중 (2026-05-19 세션)

#### Student 훈련 1차 실패 및 수정

| # | 커밋 | 문제 | 원인 | 수정 |
|---|------|------|------|------|
| 1 | — | iter 724까지 ep_len=899 고착 | Student obs(25dim)에 box 위치 없음 → approach gradient 있어도 policy가 활용 불가 | `cd3394a` box_rel + N(0,0.03m) 추가 → 28dim |

**반복 실수 기록**: Phase 3에서도 상대속도 누락(9→17dim)으로 동일 패턴 발생.
**교훈**: obs 설계 전 "policy가 reward를 받으려면 어떤 정보가 필요한가?" 체크리스트 필수.

#### Student obs 확정 (28-dim)

```
ee_pos(3) + gripper_w(1) + goal_rel(3) + noisy_box_rel(3, σ=0.03m) + jpos(9) + jvel(9)
```

- `noisy_box_rel`: 카메라 감지 시뮬레이션 (σ=0.03m)
- 실제 Jetson 배포 시 RGB-D 카메라 출력으로 대체

#### 현재 상태 (2026-05-19 세션 종료 시점)

- Student 2차 훈련 **재시작** (from Teacher model_2999.pt)
- 명령: `python training/single_robot/train_manipulation.py --student --teacher_ckpt logs/warehouse_manipulation_teacher/model_2999.pt --num_envs 5096 --max_iter 3000 --headless`

**조기 진단 기준**:
1. iter ~300: ep_len < 800 (grasp 시작)
2. iter ~600: reward 양수 전환
3. iter ~1500: ep_len < 400
4. 최종: place_rate > 80% → Student 완료

---

#### 다음 세션 재시작 절차

```bash
# RunPod 접속 후
cd /workspace/MARS && git pull

# Student 훈련 재개 (체크포인트 있으면 --resume_ckpt 추가)
python training/single_robot/train_manipulation.py \
  --student \
  --teacher_ckpt logs/warehouse_manipulation_teacher/model_2999.pt \
  --num_envs 5096 --max_iter 3000 --headless

# TensorBoard
tensorboard --logdir logs/warehouse_manipulation_student --port 6006

# 평가 (Student 훈련 완료 후)
python training/single_robot/eval_manipulation.py \
  --ckpt logs/warehouse_manipulation_student/model_2999.pt \
  --num_episodes 200 --num_envs 512
```

---

#### 이후 단계

1. **Student eval**: place_rate > 80% 확인
2. **Phase 4**: LLM 오케스트레이터 (Claude API + pgvector RAG)
3. **Phase 5**: 통합 테스트 (Jetson 배포, Student 정책 교체)

---

### Jetson 완료
- PyTorch 2.8.0 + CUDA 설치
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

- RL policy inference: **0.33ms = 3034 Hz**
- ros2_bridge.py: `/goal_pose` → `/cmd_vel` 파이프라인 동작 확인

---

## RunPod 재생성 절차

```bash
git clone https://github.com/vanillaturtlechips/MARS.git /workspace/MARS
bash /workspace/MARS/deploy/runpod/setup.sh
```

---

## 전체 남은 작업

| 항목 | 상태 |
|------|------|
| Phase 3 Low-level Controller (model_9999.pt) | ✅ 완료 |
| Phase 2 Teacher (model_2999.pt, 100% place) | ✅ 완료 |
| Phase 2 Student (28dim obs, 재훈련 중) | 🔄 진행 중 |
| Phase 4 에이전트 레이어 (LLM 오케스트레이터) | 다음 단계 |
| Phase 5 통합 테스트 | Phase 4 완료 후 |

---

## 파일 위치 요약

```
MARS/
├── logs/
│   ├── warehouse_nav/model_999.pt              # Phase 1 ✅
│   ├── warehouse_obstacle_nav/model_100.pt     # Phase 1.5 ✅
│   ├── warehouse_ippo/model_400.pt             # Phase 3 IPPO ✅
│   ├── warehouse_mappo/model_9999.pt           # Phase 3 최종 ✅ (Freeze)
│   └── warehouse_manipulation_teacher/
│       └── model_2999.pt                       # Phase 2 Teacher ✅ (100% place)
├── deploy/
│   ├── export_model.py                         # ✅
│   ├── jetson/
│   │   ├── actor_phase15.pt                   # ✅ Jetson에 복사됨
│   │   ├── inference.py                       # ✅
│   │   ├── ros2_bridge.py                     # ✅
│   │   └── benchmark_llm.py                  # ✅
│   └── runpod/
│       ├── setup.sh                           # ✅
│       └── RUNPOD_GUIDE.md                   # ✅
├── envs/warehouse/
│   ├── warehouse_env.py                       # Phase 1 ✅
│   ├── warehouse_obstacle_env.py              # Phase 1.5 ✅
│   ├── warehouse_manipulation_env.py          # Phase 2 ✅
│   ├── warehouse_marl_env.py                 # Phase 3 ✅
│   └── ippo_wrapper.py                       # Phase 3 ✅
└── training/
    ├── single_robot/
    │   ├── train_manipulation.py              # Phase 2
    │   └── eval_manipulation.py              # Phase 2 평가
    └── multi_robot/
        ├── train_ippo.py                     # Phase 3 IPPO ✅
        ├── train_marl.py                    # Phase 3 MAPPO ✅
        └── eval_scenarios.py                # Phase 3 평가 ✅
```

GitHub: github.com/vanillaturtlechips/MARS (main)
Jetson: ssh nvidia@192.168.55.1 (USB-C)
RunPod: A6000, /workspace/isaac_venv

---

*최종 업데이트: 2026-05-19 — Phase 2 Teacher 100% place 달성, Student 2차 훈련 시작 (28dim obs)*

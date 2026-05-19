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

### Phase 2 (⚠️ 훈련 진행 중 — 2026-05-19 세션)

- `envs/warehouse/warehouse_manipulation_env.py` — Franka Panda Pick & Place
- `training/single_robot/train_manipulation.py` — Teacher PPO
- `training/single_robot/eval_manipulation.py` — place_success_rate 평가 스크립트

---

#### 이번 세션 버그 수정 이력 (2026-05-19)

| # | 커밋 | 문제 | 원인 | 수정 |
|---|------|------|------|------|
| 1 | trivial success (이전 세션) | place 100% 즉시 성공 | EE 시작점≈grasp zone | box 스폰 x∈[0.60,0.75], threshold 축소 |
| 2 | `476af16` | IK가 EE를 거의 안 움직임 (0.002m/step) | Franka stiffness=80 → 명령 12%만 추종 | stiffness=400, damping=40 |
| 3 | `24f3e5a` → `b9b518d` | joint space 제어 시도 후 실패 | entropy=11.41(최대), reward=-89로 고착 | Cartesian IK로 revert |
| 4 | `92d9ed2` → revert | delta(PBRS) reward 폭발 | EE 방황 시 누적 음수 (-532), VF loss=30000 | exp(-dist) 방식으로 복원 |
| 5 | `6e464f9` | 원거리에서 학습 신호 소실 | decay=3 → exp(-3m)≈0.05 (gradient 거의 0) | decay=1.0으로 완화 |
| 6 | **`542f41d`** | **모든 훈련 실패 근본 원인** | **box에 `disable_gravity=False`** → 리셋마다 바닥으로 추락 | **`disable_gravity=True`** (collision도 disable) |
| 7 | `7ff6388` | iter 1534에서도 grasp 미발생 | EE-box 평균거리 0.36m > threshold 0.20m | `grasp_dist_threshold: 0.20→0.30m` |

**Box gravity 버그 상세**:
- `collision_enabled=False` 상태에서 gravity만 켜져 있으면 → 매 에피소드 리셋 후 박스가 테이블 통과해 즉시 추락
- 이전 모든 훈련(model_2999.pt 포함)이 이 버그로 무효화
- 수정 후 즉시 reward 70 → 2026 (보상 신호 정상화 확인)

---

#### 현재 확정 환경 설정 (최종)

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| action_space | 4 | [dx, dy, dz, gripper] Cartesian delta |
| max step | 3cm/step | DLS IK λ=0.01 |
| stiffness | 400 N·m/rad | (기본 80에서 상향) |
| damping | 40 | |
| box 스폰 | x∈[0.60,0.75] | EE 시작점(x≈0.4)에서 최소 0.2m 이격 |
| `grasp_dist_threshold` | 0.30m | EE-박스 거리 |
| `place_dist_threshold` | 0.12m | 박스-goal 거리 |
| `disable_gravity` | True | **필수 — 없으면 박스 추락** |
| `collision_enabled` | False | proximity grasp 방식 |
| `rew_approach` | 5.0 × exp(-dist×1.0) | |
| `rew_transport` | 5.0 × exp(-dist×1.0) | grasped 시에만 |
| `rew_grasp` | 10.0 | 파지 성공 순간 |
| `rew_place` | 20.0 | 거치 성공 |
| `rew_time` | -0.02/step | |
| `empirical_normalization` | False | |
| PLACE_GOALS | 4개 (0.4~0.5, ±0.1~0.2) | 테이블 위 목표 선반 |
| Teacher obs | 30-dim | box_rel+quat+mass+gripper+goal_rel+jpos+jvel |

---

#### 현재 훈련 상태 (2026-05-19 세션 종료 시점)

- **Run**: gravity 버그 수정 + grasp_dist=0.30m 반영 후 **재시작** (from scratch)
- **명령**: `python training/single_robot/train_manipulation.py --num_envs 5096 --max_iter 3000 --headless`
- **iter ~210 기준 지표**:
  - Mean reward: ~2020 (approach exp 보상만 쌓이는 중)
  - Episode length: 899 (아직 grasp 미발생 — 정상, 초반 탐색 단계)
  - Entropy: ~5.07 (탐색 중, 이전 좋은 run과 동일 궤도)
  - Action noise std: 0.87
- **확인 포인트**: iter 400~500에서 episode_length < 899 → grasp 발생 신호

---

#### 다음 세션 재시작 절차

```bash
# RunPod 접속 후
cd /workspace/MARS && git pull

# 훈련 재개 (체크포인트 있으면 --resume_ckpt 추가)
python training/single_robot/train_manipulation.py \
  --num_envs 5096 --max_iter 3000 --headless

# TensorBoard
tensorboard --logdir logs/warehouse_manipulation_teacher --port 6006

# 평가 (훈련 완료 후)
python training/single_robot/eval_manipulation.py \
  --ckpt logs/warehouse_manipulation_teacher/model_XXXX.pt
```

**체크 지표 우선순위**:
1. `episode_length` 감소 → grasp 발생 확인 (가장 중요)
2. `rew_grasp` 상승 → 파지 학습 진행
3. `rew_transport` 상승 → 운반 학습 진행
4. 최종 `place_success_rate > 90%` → Teacher 완료

---

#### 이후 단계 (Phase 2 Teacher 완료 후)

1. **eval**: `python training/single_robot/eval_manipulation.py --ckpt logs/.../model_2999.pt`
2. **Student 훈련**: `--student --teacher_ckpt logs/.../model_2999.pt`
3. **Teacher-Student 증류**: 가중치 필터링 (입력층 30→25dim 제외, 나머지 공유)

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

*최종 업데이트: 2026-05-19 — Phase 2 box gravity 버그(근본 원인) 수정 완료, grasp_dist=0.30m, 재훈련 중 (iter ~210)*

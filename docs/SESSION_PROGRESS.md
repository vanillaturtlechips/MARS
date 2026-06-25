# 세션 진행 현황 (최종 업데이트: 2026-05-30)

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

---

## 2026-05-20 세션

### Phase 2 시각 씬 정비 ✅

**문제**: Isaac Sim GUI에서 선반이 로봇을 덮치는 "엉망진창" 상태

**원인 및 수정**:

| 문제 | 원인 | 수정 |
|------|------|------|
| 선반이 로봇 위치(x=0)에 겹침 | 커스텀 ShelfB0/B1 prim 배치 오류 | 선반 제거 후 warehouse USD 전체 사용 |
| 로봇이 창고 바닥 타일 밖에 위치 | warehouse USD 좌표계 미확인 | SM_floor47 prim Properties 확인 (X=2.9519, Y=3.0, Z=0.0) → offset `(-2.95, -3.0, 0.0)` |
| 카메라 `lookat` 파라미터 오류 | IsaacLab API는 `target` 사용 | `set_camera_view(eye, target)` |
| `_setup_scene`에서 카메라 설정 무효 | 뷰포트가 아직 준비 안 됨 | `__init__`에서 `super().__init__()` 호출 후 설정 |

**최종 씬 설정**:
```python
# warehouse_multiple_shelves.usd — SM_floor47 타일이 로봇 원점(0,0,0)에 정렬
warehouse_cfg.func("/World/Warehouse", ..., translation=(-2.95, -3.0, 0.0))

# 조명
DomeLightCfg(intensity=1000)  # 전체 배경
SphereLightCfg(intensity=15000) × 3  # 작업대 상단

# 카메라 (로봇 + 테이블 + 창고 한 프레임)
set_camera_view(eye=[-1.5, -2.0, 1.5], target=[1.5, 0.5, 0.3])
```

---

### Phase 2 Student 3차 훈련 결과 (RunPod A5000, 3000 iter)

| iter | reward | ep_len | 비고 |
|------|--------|--------|------|
| ~518 | **691** | **520** | 훈련 중 최고점 |
| 2999 | 578 | 581 | 수렴 후 소폭 하락 |

**eval (model_2999.pt, 500에피소드 × 128 env)**:
```
place 39.2%  drop 0.0%  timeout 60.8%  avg_len 669.4
```
→ 이전 54%보다 하락. 3000 iter 학습이 오히려 최고점을 지나쳐 성능 저하.

**저장된 체크포인트**:
```
model_0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 2999
```
최고 성능 구간(iter ~518)은 model_300~model_600 사이 → 중간 체크포인트 eval 예정.

---

### 버그 수정

| 커밋 | 문제 | 수정 |
|------|------|------|
| `a625976` | eval GUI 강제 headless | `--headless` 명시 시만 headless 처리 |
| `a625976` | Teacher 체크포인트 obs/act 불일치 (33dim/9act) | 체크포인트에서 아키텍처 자동 감지 (`load_actor`), obs 불일치 시 zero-action fallback |
| `2cd14eb` | YCB 박스 USD 로컬 하드코딩 경로 → RunPod에서 FileNotFoundError | `importlib + glob`으로 isaacsim extscache 동적 탐색 |

---

### 중간 체크포인트 eval 결과 (부분)

eval 도중 RunPod 종료로 중단. 확인된 결과:

| 체크포인트 | place% | timeout% | avg_len | 비고 |
|-----------|--------|----------|---------|------|
| model_300 | 2.3% | 97.7% | 882.4 | 학습 초반, 예상된 낮은 성능 |
| model_600 | (중단) | — | — | — |

**반성**: model_300은 iter 518 정점 이전 구간이라 처음부터 제외했어야 함.

### 다음 RunPod 재시작 시 eval 명령어

```bash
git pull origin main

# model_300 제외, 정점 구간 집중
python training/single_robot/eval_manipulation.py \
    --ckpt logs/warehouse_manipulation_student/model_600.pt \
           logs/warehouse_manipulation_student/model_900.pt \
           logs/warehouse_manipulation_student/model_1200.pt \
           logs/warehouse_manipulation_student/model_1500.pt \
    --student --num_envs 128 --num_episodes 200 --headless
```

- **목표**: place_rate > 80% 체크포인트 확정
- **실패 시**: lr 낮추거나 (1e-4 → 3e-5) early stop 적용 후 4차 훈련
- **통과 시**: Phase 4 (LLM 오케스트레이터) 진입

> **체크포인트 위치**: RunPod `/workspace/MARS/logs/warehouse_manipulation_student/` → GitHub push로 보존 (model_0~2700 포함)

---

### Phase 2 Student 진행 중 (2026-05-19 세션)

#### Student 훈련 실패 이력 및 수정

| # | 커밋 | 문제 | 원인 | 수정 |
|---|------|------|------|------|
| 1 | `cd3394a` | iter 724까지 ep_len=899 고착 | Student obs(25dim)에 box 위치 없음 → approach gradient 있어도 policy가 활용 불가 | noisy_box_rel(3) 추가 → 28dim |
| 2 | `1d1ef17` | σ=0.03m 고정은 per-step 샘플 → 3cm 이동 vs 최대 노이즈(SNR<1) → gradient 파괴 위험 | per-step 고정 노이즈는 현실과 다름 (카메라는 같은 환경에서 일정한 노이즈 레벨 유지) | per-episode [1cm, 6cm] 균일 샘플로 변경 |

**반복 실수 기록**: Phase 3 상대속도 누락(9→17dim)과 동일한 패턴.
**교훈**: obs 설계 전 "policy가 reward를 받으려면 어떤 정보가 필요한가?" 체크리스트 필수. "나중에 추가" 없음.

#### Student obs 확정 (28-dim)

```
ee_pos(3) + gripper_w(1) + goal_rel(3) + noisy_box_rel(3) + jpos(9) + jvel(9)
```

- `noisy_box_rel`: 에피소드마다 σ ∈ [1cm, 6cm] 균일 샘플 (카메라 DR)
- 에피소드 내 노이즈 레벨 고정 → gradient 안정 + 에피소드마다 카메라 품질 변동 → Sim2Real 강건성
- 실제 Jetson 배포 시 RGB-D 카메라 출력으로 대체

#### 현재 상태 (2026-05-19 세션 종료 시점)

- Student 3차 훈련 **재시작** (from Teacher model_2999.pt, per-episode DR 적용)
- 명령: `python training/single_robot/train_manipulation.py --student --teacher_ckpt logs/warehouse_manipulation_teacher/model_2999.pt --num_envs 5096 --max_iter 3000 --headless`

**조기 진단 기준**:
1. iter ~300: ep_len < 800, reward 양수 전환
2. iter ~800: ep_len < 600
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
- ROS2 Jazzy 설치
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
| Phase 2 Student (28dim obs, 3차 훈련 완료 — best ckpt 탐색 중) | 🔄 진행 중 |
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

---

## 2026-05-21 세션

### Phase 2 근본 원인 진단 및 전면 재설계

이전 Teacher(model_2999.pt) 기반 Student 훈련 전략 폐기.
`dist_box_goal`이 0.78~0.80m에서 250+ iter 고착 → 근본 원인 집중 디버깅.

#### 발견된 버그 및 수정 (상세: `docs/phase2_transport_debug.md`)

| # | 커밋 | 문제 | 수정 |
|---|------|------|------|
| 1 | `3f8a4a5` | `transport_rel` 부호 반전 — goal 반대 방향 이동 | `goal - box_carried` (양수 방향) |
| 2 | `3f8a4a5` | 물리 bounce — EE home z < table z → 박스 폭발 | `grasp_ee_offset[2] = 0.06m` |
| 3 | `5d9c86f` | home 로컬 옵티멈 — `goal_prox + shaping`으로 home이 최고 reward 지점 | `rew_goal_prox = 0`, `rew_transport = 0` |
| 4 | `7b50e8f` | surrogate_loss ≈ 0 — IK 효율 2mm/step → 정책 gradient 없음 | alignment reward `cos_sim(action, goal_dir) × 3.0/step` |
| 5 | (동일) | Joint control entropy trap — noise_std 0.54→0.90 폭발 | Cartesian IK 복귀, entropy_coef 0.001 |
| 6 | (동일) | `rew_place=800` → VF loss 137,921 | `rew_place=100` |
| 7 | `7b40faf` | `inference.py` act_dim=8 (joint control 잔재) | act_dim=4 수정 |
| 8 | `dcd3672` | MARL stationary 패널티 — goal 도달 로봇도 -0.3/step | `not_at_goal` 마스킹 |

#### 현재 상태 (2026-05-21 세션 종료)

- Phase 2 재훈련 대기 중 (RunPod에서 `git pull` 후 시작 필요)
- **근본 원인 확정**: IK 효율 저하로 인한 `surrogate_loss ≈ 0` deadlock
- **수정 완료**: alignment reward 추가 (commit 7b50e8f, push 완료)

```bash
# RunPod 재시작 명령
git pull
pkill -f train_manipulation
nohup python training/single_robot/train_manipulation.py \
  --num_envs 5096 --max_iter 3000 --lr 1e-3 --headless \
  > train.log 2>&1 &
tail -f train.log
```

**iter 20-30 확인 기준**:
- `surrogate_loss` > 0.001 → gradient 발생 확인
- `dist_box_goal` 0.75m 이하로 감소 → transport 학습 시작
- `place_rate` 상승 → 수렴 중

#### 전체 남은 작업 (업데이트)

| 항목 | 상태 |
|------|------|
| Phase 3 Low-level Controller (model_9999.pt) | ✅ 완료 |
| Phase 2 환경 재설계 + 버그 수정 | ✅ 완료 |
| Phase 2 훈련 수렴 | 🔄 재훈련 대기 (alignment reward 적용) |
| Phase 4 에이전트 레이어 (LLM 오케스트레이터) | Phase 2 완료 후 |
| Phase 5 통합 테스트 | Phase 4 완료 후 |

---

*최종 업데이트: 2026-05-21 — Phase 2 transport 근본 원인(surrogate_loss≈0) 진단 완료, alignment reward 수정 push. 재훈련 대기 중.*

---

## 2026-05-23 세션

### Phase 3 전면 재검증 및 버그 수정

**커밋**: `8b3b236`

#### 발견 및 수정된 버그 3종

| # | 파일 | 문제 | 수정 |
|---|------|------|------|
| 1 | `eval_scenarios.py` | **S5 시나리오 물리적 불가능** — 대각 경로 `(-3,-3)→(3,3)`, `(-3,3)→(3,-3)`이 각각 2개의 선반 내부를 통과 (t≈0.05-0.10, 0.90-0.95). 훈련 실패가 아니라 설계 버그. | y=0 중앙 통로(선반 없음) 교차 경로로 교체: `spawns [(-4,0),(4,0.3),(0,4)] goals [(4,0),(-4,0.3),(0,-4)]` |
| 2 | `eval_scenarios.py` | **eval 분류 버그** — 마지막 스텝에서 goal 도달 시 `terminated AND timed_out` 동시 True → `if terminated and not timed_out` 조건에서 deadlock으로 오분류 | `if terminated[idx]` 로 변경 후 reached 별도 계산 |
| 3 | `warehouse_marl_env.py` | **cfg.beta 불일치** — 코드는 `0.5`, docs는 `1.5` 의도. model_9999.pt는 0.5로 학습됨 (pairwise 반발력 의도보다 약함) | `cfg.beta = 1.5` 로 수정 + 주석에 "model_9999.pt는 0.5 훈련" 명기 |

#### Phase 3 최종 확정 — model_9999.pt Freeze

```
17-dim obs + rew_collision=-200 + SAFE_DIST=2.5 + W_REP=1.5 + cfg.beta=0.5 (실제 훈련값)
```

| 시나리오 | 충돌률 | 교착률 | 전원도달 |
|---------|:------:|:------:|:-------:|
| S1 정면 충돌 | 0% | 7% | **93%** |
| S2 3-way 교착 | 0% | 16% | **84%** |
| S3 통로 우선 | 1% | 4% | **95%** |
| **S1~S3 평균** | **0.3%** | **9%** | **90.7%** |
| S4 동일 목표 | 0% | 100% | 0% — 구조적 한계, LLM 레이어 해결 |
| S5 혼합 | 0% | 100% | 0% — 시나리오 버그 수정됨, RunPod 재eval 필요 |

**목표 달성**: 충돌 < 1% ✅ / 교착 < 1% ❌ (9%, 구조적 한계)

---

### Phase 2 시각 에셋 복원

**커밋**: `415bca5`

5월 20일 세션에서 추가했던 에셋 코드가 5월 21일 환경 전면 재설계 시 소실됨. 동일 내용 복원:

| 변경 | 내용 |
|------|------|
| 박스 `CuboidCfg` → YCB `003_cracker_box` | `importlib + glob`으로 isaacsim extscache 동적 탐색, `scale=0.7×`, `collision_enabled=False` (훈련 물리 동일) |
| 테이블 `CuboidCfg` → `PackingTable` USD | kinematic, 상면 z≈0.50m (기존 Cuboid 상면과 동일) |
| 창고 배경 | `enable_background=True` 시에만 `warehouse_multiple_shelves.usd` + 구체 조명 3개 로드 |
| 기본값 `enable_background=False` | headless 훈련 속도 영향 없음 (warehouse USD가 PhysX 4× 병목 유발하므로) |

---

### 남은 작업 (업데이트)

| 항목 | 상태 |
|------|------|
| Phase 3 Low-level Controller (model_9999.pt) | ✅ Freeze 확정 |
| Phase 3 S5 재eval (수정된 시나리오) | 🔄 RunPod 필요 |
| Phase 2 환경 재설계 + 에셋 복원 | ✅ 완료 |
| Phase 2 훈련 재시작 (alignment reward) | 🔄 RunPod 필요 |
| Phase 2 Student 중간 체크포인트 eval (model_600~1500) | 🔄 RunPod 필요 |
| Phase 4 LLM 오케스트레이터 | Phase 2 완료 후 |

---

*최종 업데이트: 2026-05-23 — Phase 3 버그 수정 완료, Phase 2 Teacher 100% place rate 확정 및 GitHub 저장 완료.*

---

## 2026-05-25 세션 — Phase 2 Student 포기, 코드베이스 정리

### Phase 2 최종 결론

**Student 정책 개발 포기. Teacher(model_2999.pt, 100% place_rate) 단일 정책으로 확정.**

시도한 모든 방법:

| 접근법 | 결과 |
|--------|------|
| 순수 Student PPO | 7% plateau (2주 이상) |
| BC(Behavioral Cloning) + PPO, lr=1e-3 | 43%→7% (iter 67 붕괴) |
| BC + PPO, lr=1e-4 | iter 22 붕괴 |
| BC + PPO, lr=1e-5, noise_std=0.1 | 7% (분포 불일치) |
| BC + critic warmup (actor frozen) | 1~2% |

실패 근본 원인: Student obs(30D)로는 Teacher action을 충분히 예측 불가. 랜덤 critic이 초기에 BC를 오염. 학습 분포 불일치.

### 코드 정리 (커밋 2dd3d5a)

- `WarehouseManipulationStudentEnvCfg`, `WarehouseManipulationStudentTransportEnvCfg` 삭제
- `collect_bc_data.py`, `train_bc.py`, `diag_transport.py` 삭제
- train_manipulation.py: `--student`, `--transport_only`, `--bc_ckpt` 등 제거
- `actor_phase2_student.pt` 삭제 → `actor_phase2_final.pt` 통일
- `logs/warehouse_manipulation_student/` 삭제

### 데모 실행

```bash
python training/single_robot/demo_manipulation.py \
  --ckpt logs/warehouse_manipulation/model_2999.pt \
  --num_envs 4 --livestream 1
```

---

## 최종 상태 (2026-05-25)

| 항목 | 상태 |
|------|------|
| Phase 3 Low-level Controller (model_9999.pt) | ✅ 완료 |
| Phase 2 Pick & Place (model_2999.pt, 100% place) | ✅ 완료 |
| Phase 2 Student 정책 | ❌ 포기 — Teacher로 대체 |
| Phase 4 LLM 오케스트레이터 | 🔄 진행 중 (다른 팀원) |

## 다음 세션 할 일

| 순서 | 작업 | 명령어 |
|------|------|--------|
| 1 | Phase 2 데모 녹화 | `python training/single_robot/demo_manipulation.py --ckpt logs/warehouse_manipulation/model_2999.pt --num_envs 4 --livestream 1` |
| 2 | Phase 3 S5 재eval | `python training/multi_robot/eval_scenarios.py --ckpt logs/warehouse_mappo/model_9999.pt --num_episodes 100 --num_eval_envs 16 --tag 17dim_s5fix --headless` |
| 3 | Phase 4 LLM 오케스트레이터 | 다른 팀원 담당 |

---

## 2026-05-29~30 세션 — Transport Phase 2 완성 + Pick & Place 설계

### 핵심 성과

| 항목 | 결과 |
|------|------|
| Transport Phase 1 (carry-only) | ✅ model_200.pt, 100% reach rate |
| Transport Phase 2 (carry+release+place) | ✅ model_600.pt, **99% place_rate** (Stage 3, goal=0.50m) |
| Pick & Place 완성 설계 | ✅ manipulation env 실제 release 추가, train_pickplace.py 생성 |

---

### Transport Phase 1 & 2 훈련 성공 (`warehouse_transport_env.py`)

**구조**: force_grasp(박스 텔레포트) → carry → release → 물리 착지

**Phase 2 진행 중 발생한 버그 3종 및 수정**:

| # | 문제 | 원인 | 수정 | 커밋 |
|---|------|------|------|------|
| 1 | stat 이중집계 — place_rate=65% 허위 표시 | `_get_rewards`에서 `_stat_placed`, `_stat_episodes`를 `placed.sum()`으로 동시 증가 → `_get_dones`와 이중 계산 | `_get_rewards`의 두 줄 제거, `_get_dones`에서만 집계 | `6ae8403` |
| 2 | 조기 release — 무작위 위치(goal 0.40m+)에서 release | `wants_release`에 near_goal 게이팅 없음. grip_mean=+0.10, std=0.27 → P(action<-0.3)=7%/step 상시 발동 | `near_xy < near_release_dist(0.20m)` 조건 추가 | `6ae8403` |
| 3 | 즉시 release local optimum | bias=-0.8 오버라이드 → 에피소드 시작 즉시 release → carry 없이 허위 place 성공 | bias override 제거 (bootstrap+penalty로만 탐색) | `47f092c` |

**결과**: iter 200 이후 99% place_rate, Stage 3(goal=0.50m) 도달, force_rel=0.0%

**체크포인트**:
```
logs/warehouse_transport/model_200.pt        # Phase 1 carry-only
logs/warehouse_transport_phase2/model_600.pt # Phase 2 transport+place (99%)
```

---

### Pick & Place 완성 — Teacher 한계 발견 및 수정

**Teacher (model_2999.pt) 한계**:
```python
# 기존 (가짜 place) — 박스를 절대 안 놓음
placed = self._grasped & (dist_box_goal < threshold)
```
→ approach+grasp+transport 완료 후 박스를 쥔 채 goal 근처에서 에피소드 종료. 물리적 내려놓기 없음.

**Transport Phase 2 구조 이식** (`warehouse_manipulation_env.py`):
```python
# 실제 물리 착지
settled = self._has_released & (self._steps_after_rel >= settle_steps)
placed  = ~self._grasped & settled & (dist_box_goal < threshold)
```

추가된 요소:
- `near_release_dist=0.20m` XY 게이팅: goal 근처서만 release 허용
- `bootstrap`: 초기 N iter 강제 release로 place 경험 주입
- `settle_steps=10`: release 후 box 안착 대기
- `fell_off` 감지: box z < 0.9m → 종료
- `rew_release_near=50.0`: goal 근처 release one-time bonus
- `rew_grip_penalty=1.5`: goal 근처 gripper 닫힘 soft 패널티
- box release 시 EE 아래 0.08m 배치 (finger collision 방지, Bug 10과 동일)

**전체 파이프라인**:
```
approach+grasp (Teacher 기존 skill)
       ↓
transport (Teacher 기존 skill)
       ↓
near_goal gating → release (신규)
       ↓
물리 착지 확인 (신규)
```

---

### 신규 훈련 스크립트: `train_pickplace.py`

Teacher `model_2999.pt`에서 fine-tuning:

```bash
# RunPod에서 실행
python training/single_robot/train_pickplace.py \
    --headless --num_envs 2048 \
    --teacher_ckpt logs/warehouse_manipulation_teacher/model_2999.pt
```

커리큘럼 (place_rate 기반):
- Stage 1 (box_spawn=0.20m): 10iter 평균 place_rate ≥ 20%
- Stage 2 (box_spawn=0.30m): 15iter 평균 place_rate ≥ 20%
- Stage 3 (box_spawn=0.45m): 최종

모니터링 지표:
- `grasp=X%`: approach+grasp 성공률 (Teacher skill 유지 여부)
- `place_rate`: 실제 물리 착지율
- `force=X%(p=Y)`: bootstrap decay (40iter 후 0)

---

### 전체 체크포인트 현황

| 정책 | 파일 | 역할 | place 방식 |
|------|------|------|-----------|
| Navigation | `warehouse_nav/model_999.pt` | 이동 로봇 | — |
| Obstacle Nav | `warehouse_obstacle_nav/model_100.pt` | 장애물 회피 | — |
| IPPO | `warehouse_ippo/model_400.pt` | 3대 독립 PPO | — |
| MAPPO | `warehouse_mappo/model_9999.pt` | 3대 협력 최종 | — |
| Teacher | `warehouse_manipulation_teacher/model_2999.pt` | approach+grasp+carry | 가짜 (미해결) |
| Transport P1 | `warehouse_transport/model_200.pt` | carry-only | — |
| Transport P2 | `warehouse_transport_phase2/model_600.pt` | carry+release | **실제 물리 착지** |
| Pick & Place | `warehouse_pickplace/` (훈련 예정) | 완전한 pick & place | **실제 물리 착지** |

---

### 다음 할 일

| 순서 | 작업 | 명령어 |
|------|------|--------|
| 1 | Pick & Place fine-tuning | `python training/single_robot/train_pickplace.py --headless --num_envs 2048 --teacher_ckpt logs/warehouse_manipulation_teacher/model_2999.pt` |
| 2 | 결과 확인 후 model_pickplace.pt 저장 | iter 600, place_rate ≥ 80% 목표 |
| 3 | Phase 4 demo 연결 | `demo_phase4.py` — Claude API → Isaac Sim 실제 연결 |

---

*최종 업데이트: 2026-05-30 — Transport Phase 2 완성(99%), manipulation env 실제 release 이식, train_pickplace.py 생성.*

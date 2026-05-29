# TransportEnv 디버깅 기록 — carry_dist 0.55m 고착 원인 추적

**기간**: 2026-05-29  
**환경**: `envs/warehouse/warehouse_transport_env.py`  
**학습 스크립트**: `training/single_robot/train_transport.py`

---

## 환경 설계 개요

### 목표
Franka Panda 로봇이 박스를 집어 goal 위치까지 운반하는 Transport Policy 학습.  
Phase 1: carry-only (reach 성공), Phase 2: release + place.

### 관측 (23D)
```
goal_rel(3) + dist_to_goal(1) + ee_vel(3) + jpos(7) + jvel(7) + gripper_w(1) + is_grasped(1)
```
- `goal_rel = goal_pos - ee_pos` (grasped 시 항상 의미 있는 방향 신호)

### 액션 (4D)
`[dx, dy, dz, gripper]` — Cartesian IK via DLS, ±3cm/step

### 에피소드 흐름
1. `_reset_idx`: 홈 포즈 복귀, `_pending=True`
2. Step 1 `_get_rewards`: force_grasp 발동 → box snap to EE, goal 설정
3. 이후: IK로 EE 이동, kinematic lock으로 box를 EE 위치에 추적
4. 종료: EE가 goal 0.05m 이내(term_reached) 또는 시간 초과

### 설정 (최종)
```python
rew_carry_dist = 1.0    # -dist/step
rew_dir        = 0.0    # 제거됨 (advantage SNR 파괴)
rew_place      = 0.0    # Phase 1에서 0
rew_time       = -0.02
goal_spawn_dist = 0.10  # 커리큘럼 시작: EE에서 10cm
init_noise_std  = 0.10
```

---

## 커리큘럼

```python
CURRICULUM_STAGES = [
    (0.10, 50.0, 15),   # 1단계: place_rate 50% → 진급
    (0.20, 40.0, 15),   # 2단계
    (0.35, 35.0, 15),   # 3단계
    (0.50, None, None), # 최종
]
```

place_rate = (placed + reached) / episodes. Phase 1에서 release 비활성화이므로 term_reached만으로 진급.

---

## 발견·수정된 버그 목록 (시간순)

### 버그 1 — `reached` 누락 (done 미반환)
**커밋**: `b58526f`  
**현상**: VF loss 스파이크 9367, 학습 불능  
**원인**:
```python
# 수정 전 (버그)
return placed | fell_off, timed_out   # reached 빠짐!
```
EE가 goal 0.05m 이내에 도달할 때마다 `rew_place=300` 보상이 **에피소드 종료 없이 매 step** 지급됨.

**수정**:
```python
return placed | reached | fell_off, timed_out
```

---

### 버그 2 — goal z 고정값 1.15m
**커밋**: `b58526f`  
**원인**: EE home z ≈ 1.30m인데 goal z = 1.15로 하드코딩 → 초기 3D 거리 = √(0.10²+0.15²) ≈ 0.18m.  
`reached` 임계값 0.05m는 물리적으로 거의 도달 불가.

**수정**:
```python
self._goal_pos_w[act_ids, 2] = ee_now[:, 2]  # EE z에 맞춤
```

---

### 버그 3 — `rew_dir` 고분산 노이즈로 advantage SNR 파괴
**커밋**: `134ed8a`  
**현상**: surrogate_loss ≈ 0 → 50+ iter 동안 policy 업데이트 없음  
**원인**: `rew_dir = dot(ee_vel, goal_dir)`. ee_vel ≈ ±1.8 m/s → advantage 분산 폭발. carry 신호 묻힘.

**수정**: `rew_dir = 0.0` 제거, `rew_carry_dist` 0.3 → 1.0

---

### 버그 4 — `init_noise_std=1.0` → VF cold-start
**커밋**: `fcf3f36`  
**현상**: carry_dist = 0.62m 고착  
**원인**: std=0.51 → EE 매 step 0.05m 이동 → 모든 state가 동일 return(-93) → V(s)=-93 everywhere → advantage=0

**수정**: `init_noise_std 1.0→0.10`, std clamp max 0.5→0.30

---

### 버그 5 — Actor 출력 레이어 Xavier 초기화
**커밋**: `964f476`  
**원인**: `init_noise_std=0.10`으로 noise를 줄여도 Xavier init mean ≈ N(0, 0.5)이 지배.  
action = mean + 0.10×noise ≈ mean (random, 크다) → EE 랜덤 이동 → VF cold-start 여전

**수정**:
```python
runner.alg.policy.actor[-1].weight.data.zero_()
runner.alg.policy.actor[-1].bias.data.zero_()
```

---

### 버그 6 — IK 누적 중력 드리프트 (핵심 원인)
**커밋**: `1d51847`  
**현상**: zero-init 확인 후에도 `[Diag iter1] carry_dist=0.5858` — policy 문제 아님  
**원인**: 기존 IK `new_q = current_q + delta_q` 구조.
- zero action → delta_q=0 → new_q = current_q
- 중력으로 관절 처짐 → 처진 위치가 다음 step의 current_q → 누적 드리프트
- 125 step 동안 EE 0.5m 드리프트

**수정**: `_cmd_ee_pos` 절대 목표 버퍼 도입
```python
self._cmd_ee_pos += actions[:, :3] * 0.03          # 절대 목표 누적
delta_to_cmd = self._cmd_ee_pos - current_ee_pos   # IK가 절대 목표를 추적
delta_q = J_dls @ delta_to_cmd
new_q = current_q + delta_q
```
zero action → `_cmd_ee_pos` 고정 → IK가 중력에 맞서 능동 보상.

---

### 버그 7 — pending 상태 IK: `_cmd_ee_pos=0.0`이 월드 원점으로 날아감
**커밋**: `403fa25`  
**현상**: carry_dist iter1=0.5551 (이전 수정 후에도 고착)  
**원인**: `_reset_idx`에서 `_cmd_ee_pos=0.0`. `_apply_action`이 `_get_rewards`(force_grasp)보다 먼저 실행:
```
delta_to_cmd = (0,0,0) - ee_pos_home = [-0.55, 0, -1.30]  (magnitude 1.41m)
→ 로봇이 매 에피소드 첫 step마다 월드 원점으로 최대 속도 이동
```

**수정**:
```python
# pending 환경은 delta=0 마스킹
delta_to_cmd = torch.where(
    self._pending.unsqueeze(1).expand(-1, 3),
    torch.zeros_like(ee_pos),
    self._cmd_ee_pos - ee_pos,
)
# cmd_ee 누적도 pending 중에는 건너뜀
not_pending = (~self._pending).float().unsqueeze(1)
self._cmd_ee_pos += actions[:, :3] * 0.03 * not_pending
```

---

### 버그 8 — goal lx 클램프가 커리큘럼 0.10m 무력화
**커밋**: `403fa25`  
**원인**: EE local x ≈ 0.35m인데 `lx.clamp(0.55, 1.45)` 적용.
- raw lx best = 0.35 + 0.13 = 0.48m → clamp 후 0.55m
- 실제 최소 EE-goal 거리 = 0.55 - 0.35 = **0.20m** (goal_spawn_dist=0.10 설정해도 불가)

**수정**: lx/ly 클램프 제거, EE 기준 직접 offset으로 goal 배치
```python
self._goal_pos_w[act_ids, 0] = ee_now[:, 0] + r * torch.cos(th)
self._goal_pos_w[act_ids, 1] = ee_now[:, 1] + r * torch.sin(th)
self._goal_pos_w[act_ids, 2] = ee_now[:, 2]
```

---

### 버그 9 — `_reset_idx` 후 stale `robot.data.joint_pos`
**커밋**: `fcfc510`  
**현상**: carry_dist iter1 = 0.451m (여전히 0.10m 아님)  
**원인**: Isaac Lab `step()` 실행 순서:
```
_reset_idx:
    write_joint_state_to_sim(home_pose)   ← physics = home_pose ✓
    set_joint_position_target(home_pose)  ← PD target = home_pose ✓
    (scene.update() 없음)

다음 step() _apply_action:
    robot.data.joint_pos = STALE(이전 ep 끝 위치)   ← 문제
    target_q = stale_q + delta_q(=0) = stale_q
    → PD controller가 arm을 home에서 stale 위치로 당김
    → force_grasp ee_now = 중간 어딘가 (home 아님)
    → goal = wrong 위치 + r → carry_dist 폭등
```
`init_at_random_ep_len=True`로 첫 rollout에서 ~50% env 조기 리셋 → 과반수가 stale target 문제.

**수정**: `_home_q` 버퍼 도입. pending 환경은 stale `robot.data.joint_pos` 대신 `_home_q` 사용
```python
# __init__
self._home_q = torch.tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], device=d)
              .unsqueeze(0).expand(n, -1).clone()

# _apply_action
base_q = torch.where(
    self._pending.unsqueeze(1).expand(-1, 7),
    self._home_q,                          # stale 대신 home 사용
    self.robot.data.joint_pos[:, :7],
)
new_q = (base_q + delta_q).clamp(-2.8, 2.8)

# pending 중 그리퍼도 닫힘 유지
gripper_action = torch.where(self._pending, torch.full((n,), -1.0, ...), self._actions[:, 3])
```

---

### 버그 10 — PhysX 충돌 폭발 (box-finger geometry 중첩)
**커밋**: `49eee80`  
**원인**: box center를 `panda_leftfinger` tip에 스냅하면 finger collision geometry와 완전히 겹침.  
rigid body box가 kinematic override되므로 PhysX 척력이 전부 arm joints로 전달.  
→ 매 physics step 수백 N 방향 힘 → arm joints 편차 → carry_dist 지속 상승.

**수정**: box center = EE tip - 0.08m (finger 아래 매달림으로 중첩 해소)
```python
# force_grasp
state[:, :3] = ee_now
state[:, 2] -= 0.08

# kinematic lock (매 step)
frozen[:, :3] = ee_pos[grasped_ids]
frozen[:, 2] -= 0.08

# release
state[:, :3] = ee_pos[rel_ids]
state[:, 2] -= 0.08
```

---

## 버그 발생 원인 패턴

| 카테고리 | 버그 수 |
|---------|--------|
| RL reward/done 설계 오류 | 3 (버그 1, 2, 3) |
| Policy 초기화 문제 | 2 (버그 4, 5) |
| Physics/IK 제어 로직 | 3 (버그 6, 7, 10) |
| 환경 초기화 타이밍 | 2 (버그 8, 9) |

---

## Isaac Lab 실행 순서 (중요)

```python
# step() 내부 실행 순서
_pre_physics_step(action)
for _ in range(decimation):          # decimation=2
    _apply_action()
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt=physics_dt)      # ← 루프 내부에서 robot.data 갱신

episode_length_buf += 1

_get_dones()                         # ← rewards 전에 실행!
_get_rewards()                       # ← force_grasp 여기서 발동
_reset_idx(done_envs)               # ← rewards 후, observations 전
_get_observations()
```

**핵심**: `_reset_idx` → 다음 `_apply_action` 사이에 `scene.update()` 없음.  
→ `robot.data`는 리셋 전 물리 상태 그대로 (stale).  
→ pending 환경은 반드시 stale data를 우회해야 함.

---

## 진단 로그 (현재)

| 로그 | 의미 |
|------|------|
| `carry_dist` | grasped 환경 평균 EE-goal 거리 |
| `ik_err` | `cmd_ee - ee_pos` 거리 — IK solver 품질 |
| `grip_action_mean/std` | 그리퍼 액션 분포 — -0.3 탐색 여부 |
| `term_reached` | EE가 goal 0.05m 이내 도달률 |
| `grasp_rate` | 파지 중인 env 비율 (항상 100%) |

---

## 현재 상태 및 기대치

### iter 1 기대
- `carry_dist ≈ 0.10m` (goal_spawn_dist)
- `ik_err ≈ 0.02m` 이하 (IK 정상 추적)
- `grip_action_mean ≈ 0.0` (초기화 직후)

### 학습 성공 기준 (Phase 1)
- `term_reached` 증가 → `carry_dist` 감소
- `place_rate` 50% 유지 15 iter → Stage 2 진급 (`goal_spawn_dist 0.10→0.20m`)

### 다음 단계 (Phase 2)
- release 활성화: `wants_release` 조건에서 `& (self.episode_length_buf > self.max_episode_length)` 제거
- goal z를 box 착지 위치(table top ≈ 0.9m)로 변경
- `place_dist_threshold` 재조정

---

## 훈련 명령

```bash
# 신규 학습
python training/single_robot/train_transport.py --headless --num_envs 8092 --max_iter 200

# 체크포인트 이어서
python training/single_robot/train_transport.py --headless --num_envs 8092 \
    --resume_ckpt logs/warehouse_transport/model_100.pt
```

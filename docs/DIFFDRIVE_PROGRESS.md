# Diff-Drive 전환 작업 정리 (2026-06-01)

> 목표: 데모에서 로봇이 **실제 창고 AMR처럼 커브를 그리며** 이동하고, 옆으로
> 미끄러지는(strafe) 부자연스러운 모션을 없앤다. **충돌은 계속 회피**해야 한다.

브랜치: `diff-drive-finetune` (main 데모는 무손상으로 분리 보관)

---

## 1. 문제의 뿌리

학습된 정책(`model_10998.pt`, 동적 장애물 회피 S6 98%)의 물리 본체는
**홀로노믹 점질량 큐브**다.

- 액션 = 로봇당 `[vx, vy, omega]` (body frame 속도)
- 적용 = `write_root_velocity_to_sim` — body 속도를 **직접** sim에 씀
- `max_vx=1.5, max_vy=1.0, max_omega=2.0`

즉 로봇이 **몸을 안 돌리고 옆으로/뒤로 미끄러져** 회피하는 게 학습된 정상
동역학이다. 이게 "스케이트 타듯 둥둥 떠다니는" 모션의 근본 원인.
바퀴·차동구동 제약이 애초에 없다.

**홀로노믹 점질량은 MARL 내비게이션 연구의 표준 모델링**(ORCA/RVO 등)이라
98% 결과 자체는 정당하다. diff-drive는 *정확성* 요건이 아니라 *비주얼 충실도*
업그레이드다.

---

## 2. 시도한 접근과 결과 (시간순)

### (A) 외형 유니사이클 그림자 — **실패(폐기)**
물리는 그대로 두고 외형 메시만 큐브를 쫓는 유니사이클로 굴림.
→ 외형 위치를 따로 적분하다 오버슈트→180° 회전→**발산(순간이동/휙휙/뚫기)**.
→ 폐기. 외형을 큐브 실제 pose에 그대로 붙이는 정직한 버전으로 되돌림.

### (B) 하드 컷 fine-tune (`vy=0` 즉시 차단) — **실패**
`disable_strafe`로 `_apply_action`에서 `vy_b=0` 하드 차단 후 model_10998에서
이어학습. obs/act 차원 불변이라 로드는 깨끗.
- 증상: 충돌은 회피하나 **목표 도달 실패**(episode length가 max에 고정,
  reward 음수 정체, value loss 폭발).
- 원인: strafe 의존 정책에 절벽 같은 동역학 변화. critic이 새 동역학을
  못 따라가 발산. **재시도 금지.**

### (C) 적응형 커리큘럼 (`max_vy` 1.0→0 점진 감소) — **부분 성공, 0.4서 정체**
문헌 검증(Rudin 2021 terrain curriculum, accuracy-based curriculum,
PBRS proxy-trap, offline-to-online critic) 반영.
- `strafe_curriculum`: env가 **성공률 자가측정**(로봇별 도달률) → 성공률이
  문턱 이상이면 `max_vy`를 한 단계 내림, 붕괴하면 되돌림(revert).
- `rew_heading`: **속도투영형** `v_forward × cos(bearing)`
  (순수 `cos(bearing)`은 "목표 보고 멈추는" 리워드해킹 함정이라 회피).
- 해결한 버그들:
  - 게이트 지표를 "3대 동시 도달"→**로봇별 도달률**로 교정(전자는 고정목표
    환경에서 거의 0이라 커리큘럼이 안 움직였음).
  - 노이즈 폭발(`entropy_coef` 보너스가 std를 0.5→1.42로 부풀려 도달률
    0.64에 가둠) → `--entropy_coef 0.0`으로 억제.
- 결과: `max_vy` **1.0→0.4까지 내려감(strafe 60% 제거)**. 그러나 **0.4 벽에서
  정체**(도달률 0.51→0.48 하락, reward 급감). 완전 diff-drive(0) 실패.
  중간 단계(0.4)에서 strafe 없는 회피 재학습을 못 함(리서치가 경고한 구간).

### (D) 계층 컨트롤러 (ROS nav stack 패턴) — **최종 채택, 검증 대기**
RL 재학습을 포기하고 **업계 표준 계층 구조**로 전환.
- 상위(플래너) = `model_10998` (98%, **손도 안 댐**) — desired velocity 출력
- 하위(컨트롤러) = diff-drive 변환:
  - `heading_err = atan2(vy_des, vx_des)` (현재 헤딩 기준 원하는 속도 방향)
  - `omega = clamp(turn_gain · heading_err, ±max_omega)` (그쪽으로 돌아라)
  - `v = |desired_vel| · relu(cos(heading_err))` (정렬된 만큼만 전진)
  - 옆 성분 = 0
- 정책의 회피 **의도**는 보존(왼쪽 피하려 strafe→왼쪽 돌아서 전진).
- **물리 큐브 자체가 nonholonomic**으로 움직임 → 외형은 실제 pose 추종 →
  어긋남/폭주 원천 불가.
- **재학습 0, GPU 0, model_10998 그대로 재생.**

---

## 3. 실행 방법

### 데모 (계층 컨트롤러, 권장)
```bash
git pull
python training/multi_robot/demo_play.py \
  --checkpoint logs/warehouse_mappo/model_10998.pt \
  --diff_drive_ctrl --turn_gain 3.0 --num_envs 1 --max_steps 1500
```
튜닝 노브(영상 보고 한 줄):
- 너무 굼뜨게 돌면 → `--turn_gain 5.0`
- 너무 홱홱 돌면 → `--turn_gain 1.5` 또는 `--max_omega 1.2`

### 정직한 검증 (RunPod headless, 회피 성공률 측정)
```bash
python training/multi_robot/eval_scenarios.py \
  --ckpt logs/warehouse_mappo/model_10998.pt \
  --diff_drive_ctrl --enable_obstacles --headless --num_episodes 100 --tag ctrl
```

### (참고) 커리큘럼 체크포인트 — fallback
`logs/warehouse_mappo_diffdrive_curr/` 에 max_vy 0.4까지 내려간 모델 보관.
"절반의 diff-drive"가 필요하면 `--diff_drive`로 재생.

---

## 4. 현황과 열린 질문

| 항목 | 상태 |
|---|---|
| 커브/주행 모양 자연스러움 | 거의 확실(컨트롤러 수식상 옆걸음 불가) |
| 충돌 회피 성공률 유지 | **미검증** — strafe 의도가 "돌아서 가기"로 변환되며 손실 가능. eval 필요 |
| 재학습 비용 | 0 (model_10998 그대로) |
| main 데모 fallback | 홀로노믹 98% 모델, 항상 살아있음 |

**다음 액션:**
1. 데스크탑에서 데모 영상 — 커브/옆걸음 확인, `turn_gain` 맞춤
2. RunPod headless eval — 회피 성공률 측정(98% 대비 얼마나 지키나)
3. 성공률이 낮으면: `turn_gain`/`max_omega` 상향, 또는 정면충돌 상황만
   보조 로직 검토

---

## 5. 코드 위치

- `envs/warehouse/warehouse_marl_env.py`
  - `_apply_action`: `diff_drive_controller` 분기(계층 컨트롤러), strafe 상한
    로직(커리큘럼/하드차단/기본)
  - `_get_rewards`: `rew_heading`(속도투영), `rew_spin`
  - `_get_dones`: 커리큘럼 성공률 자가측정(로봇별 도달률)
  - cfg 필드: `diff_drive_controller`, `diff_drive_turn_gain`,
    `strafe_curriculum`, `disable_strafe`, `rew_heading` 등
- `training/multi_robot/train_marl.py`: `--strafe_curriculum`, `--diff_drive`,
  `--rew_heading`, `--max_omega`, `--entropy_coef`, `--curriculum_*`
- `training/multi_robot/demo_play.py`: `--diff_drive_ctrl`, `--turn_gain`,
  `--diff_drive`, `--max_omega`
- `training/multi_robot/eval_scenarios.py`: 동일 플래그

## 6. 교훈

사용자의 원래 목표는 "외형이 자연스럽게 커브를 그린다"였는데, 이를
"동역학 재학습"으로 확대 해석해 GPU를 여러 번 소모했다. 정석인 **계층 구조
(RL 플래너 + 고전 로컬 컨트롤러)** 가 처음부터 메뉴에 있어야 했다. 목표를
들으면 표준 해법부터 선택지에 올릴 것.

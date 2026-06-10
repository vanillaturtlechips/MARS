# RL × 멀티에이전트 통합 — 방향 및 재학습 붕괴 연구 계획

> 작성: 2026-06-10. STATUS.md(검증된 현황)를 전제로 한 **전방 계획 문서**.
> 결론을 단정하지 않는다 — 데이터, 선택지, 트레이드오프, 열린 항목만 기록한다.
> 검증된 숫자는 커밋된 eval json이 기준이다(프로젝트 §9).

---

## 0. 핵심 토픽 (프레임)

실제 공장 물류창고를 바탕으로 한 다수 시나리오를 **정량 평가**하고, **RL(로봇 항법) ×
멀티에이전트(함대 오케스트레이션) 콜라보**로 문제를 해결한다. 통합은 두 층을 **분리**하되
**공유 메모리(블랙보드)** 로 잇는 아키텍처.

확정된 방향(2026-06-10):
- **깊은(obs-level) 통합으로 간다** — 에이전트 결정(keepout zone)을 RL 정책이 obs로 소비.
- 그 전제인 **재학습 붕괴를 정면으로 푼다**(아래 §3).
- **충전 가지는 초점 아님** — 통합 서사의 중심은 항법/keepout.

---

## 1. RL 현황 — 검증된 eval (단일 시드, 6 시나리오 × 100ep)

| 모델 | 학습 | S1 | S2 | S3 | S4 | S5 | S6 | 평균도달 | 평균충돌 |
|---|---|---|---|---|---|---|---|---|---|
| 10998 | 광역 랜덤골 MARL (scenario 안 함, 17D) | 78 | 94 | 13 | 1 | 91 | 98 | 62.5 | 0.0 |
| shelf_final | scenario, 동적장애물 없이 (19D) | 93 | 100 | 0 | 1 | 78 | 6 | 46.3 | 21.7 |
| cf_dynobs | scenario+장애물, **std 발산**(freeze_std 누락) | 79 | 96 | — | 0 | 90 | 0 | 56.0 | 19.5 |
| cf_frozen | scenario+장애물, **std 통제**(freeze_std) | 9 | 0 | 35 | 0 | 0 | 99 | 23.8 | 60.7 |

(전원도달률 %. eval json: `eval_cf_dynobs_model_4999.json`, `eval_cf_frozen_model_4999.json` 등.)

### 관측된 사실
- **cf_dynobs의 S6=0은 confound였다**: `--freeze_std` 누락 → rsl_rl의 std(nn.Parameter)가
  0.75→6.18로 단조 발산. reward 평평, eplen은 max(~300)에 고정(타임아웃). 정책이 학습 안 되고
  warm-start 근처에 정체(그래서 S1/S2/S5는 10998과 비슷하게 살아남음). model_4999는 붕괴 후 산물.
- **cf_frozen(std 통제)**: 정책이 실제로 scenario reward를 최적화 → **S6=99로 회복**했으나
  S1/S2/S4/S5가 충돌 100%로 붕괴(S2는 5스텝 즉시 충돌). **단일 동적장애물 회피에 과특화하며
  다중로봇 협응을 trade off.**
- 함의: scenario/커리큘럼 학습은 gradient를 지배하는 한 가지에 특화되고 나머지를 잃는다.
  std 발산이면 warm-start에 정체, std 통제면 한 시나리오 specialist가 됨 — **양쪽 다 못 얻음.**
- 6개를 전반 균형(S6 98 + S1/S2/S5 양호 + 충돌 0)으로 하는 건 **10998(광역 랜덤골)** 뿐.

### 한계 (검증 강도)
- **시드 1개** — S6 99 vs 0이 신호인지 분산인지 미구분.
- 베이스라인이 자기 변형끼리만. 외부(ORCA/RVO/vanilla MAPPO/Nav2) 없음.
- 6 시나리오를 저자 설계. privileged obs(센서 없이 해석적 좌표). 3로봇.

### 배포/데모용
- 현재 최선 = **10998**. shelf_final/cf_frozen은 배포에 쓰지 않음.

---

## 2. 학습 안정성 함정 (재발 방지)

`train_marl.py`:
- **`--freeze_std`** 없으면 scenario/커리큘럼 학습에서 std가 발산(policy.std가 nn.Parameter라
  entropy_coef=0이어도 surrogate gradient가 키움). 코드 주석 245-249에 경고 있음.
- ippo_ckpt warm-start는 **제로패딩 + strict=False**로 17D→19D 차원 불일치를 처리(line 211-235).
  → warm-start 차원 불일치는 붕괴 원인이 아님.
- scenario 학습 권장: `--freeze_std --reset_noise_std <val> --entropy_coef 0`,
  학습 인자는 `--enable_obstacles`(STATUS의 `--enable_dynamic_obstacles`는 잘못된 플래그명).

---

## 3. 재학습 붕괴를 푸는 연구 프로그램

### "정답"의 정의 (성공 판정 기준)
재학습 후 **멀티시드** eval에서 동시 충족:
- (a) S1~S6가 10998의 노이즈 밴드 안 (옛 능력 보존)
- (b) zone-respect ≥ 임계 (새 능력 획득)
- (c) 충돌 ~0

하나라도 미달이면 아직 붕괴.

### 측정 먼저 (지금 없음)
- 시드 1개 + zone-respect 시나리오 부재 → 붕괴를 탐지할 척도가 없다.
- 빌드: 기존 S1~S6 + **새 시나리오**(에이전트 zone 선언 → 정책이 피하나 / 도달 유지하나 / 충돌),
  **다수 시드**. 이게 곧 논문의 정량 eval 인프라.

### obs 설계 (선결정)
- 최소·구조 일관: 기존 "가장 가까운 장애물 거리+방향" 미러링한 **zone 경계 거리+방향(2채널)**.
- 대안: occupancy 패치(풍부, 고위험).

### 레시피 탐색 (척도 위에서)
진단(분포 협소화 + warm-start 드리프트)에 직접 대응하는 두 개가 1순위:
- **rehearsal / 혼합배치**: scenario+zone만 학습하지 말고 **랜덤골 에피소드를 배치에 같이** 섞음
  (10998 능력 보존). 비율이 노브. → 키워드: experience replay, rehearsal, mixed-task training.
- **KL-to-reference**: zone 없는 상태에선 10998에서 못 벗어나게 묶어 zone 있는 곳에서만 거동 변경.
  → 키워드: continual/lifelong RL, EWC, behavioral constraint, trust-region fine-tuning,
  policy distillation(teacher=10998).
- 통제: std 통제(freeze_std), reward 균형(충돌 페널티 vs goal). 배타적 아님 — 더해가며 측정.
- 측정 키워드: forward/backward transfer, plasticity-stability dilemma.

### 어디서 도나
- 붕괴-풀기는 **학습 env = MARL env(Y)** 에서. 멀티시드 헤드리스 eval이 싸므로 학습·정량 평가
  무게중심이 Y. **X(ROS2/Isaac)는 데모 영상용으로 분리** 가능 — sim 단일화는 지금 안 막아도 됨.

### 통합 sim 선택지 (나중)
- **X. RL을 ROS2/Isaac 데모로 이식**: 현실성·데모 영상 ↑, but obs 재구성 + sim-to-sim transfer
  위험 + eval 느림.
- **Y. 오케를 MARL env로**: native RL(transfer gap 없음) + 싼 멀티시드 eval, but ROS2/Nav2
  현실성 상실 + 에이전트 actuation 재배선(LLM·블랙보드 추론은 재사용).
- 블랙보드 철학상 에이전트↔RL은 **항상 블랙보드를 거치는 번역 서비스**로 결합(RL은 분리 모듈
  유지) — KeepoutService가 블랙보드→Nav2 하던 패턴과 동일. 철학은 X/Y를 강제하지 않음.

---

## 4. 충전 (de-scoped, 참고)
- 메모리 `project_battery_attempt`: RL이 배터리/충전을 **학습**하는 건 폐기(navigation 붕괴).
- 블랙보드 철학에선 폐기를 뒤집지 않고도 충전이 통합에 들어갈 수 있음: 에이전트가 순서 결정
  (블랙보드) → ChargingService 디스패치 → 로봇이 충전소로 **항법(RL이 수행)**. 배터리-in-RL은
  폐기된 채로, 충전은 "goal 할당" 레벨에서만. 단 이번 토픽의 초점은 아님.

---

## 5. 리스크 (열림)
깊은 obs-level은 **continual-RL fine-tuning 안정성**이라는 열린 연구문제. 결과 두 갈래:
- 레시피가 (a)(b)(c) 동시 충족 → 깊은 통합 성립, 그 방법이 논문 기여.
- 어떤 조합도 옛·새 능력 동시 확보 실패 → "깊은 통합엔 내재적 안정성 비용" → 다른 결과(얕은
  goal-level 후퇴 근거). 타임라인·컴퓨트 열림.

---

## 6. 즉시 다음 작업
- [ ] **rehearsal 가능 여부 확인**: `scenario_train`이 랜덤골을 대체하는지 vs 혼합 가능한지
      (`warehouse_marl_env.py`의 goal/scenario 샘플링 코드). 대체뿐이면 혼합배치가 env 수정 작업.
- [ ] zone obs 채널 추가가 얼마짜리인지 (`_get_observations`, `_nearest_obstacle_body` 구조).
- [ ] 측정 인프라(멀티시드 + zone-respect 시나리오) 규모 산정.
- [ ] 위 셋을 본 뒤 레시피 탐색 순서 확정.

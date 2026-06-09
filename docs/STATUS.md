# MARS — 단일 진실 문서 (Source of Truth)

> **이 파일을 제일 먼저 읽어라.** 다른 docs는 작성 시점의 스냅샷이며 서로 모순될 수
> 있다. 충돌 시 **이 파일 + 커밋된 eval json이 이긴다.**
>
> 최종 갱신: **2026-06-09**
> 핵심 원칙(프로젝트 §9): **"능력은 학습 reward/커밋 메시지/문서 주장이 아니라
> 커밋된 eval json 숫자로 검증된 뒤에만 '됐다'고 한다."**

---

## 0. 판단 규칙 (30초 컷)

1. 어떤 모델이 "S5 됐다"고 하면 → `logs/eval_<tag>_<model>.json`이 **있는지** 먼저 봐라.
   - **있으면** = 검증됨(§2 표). **없으면** = 주장일 뿐(§3).
2. 브랜치/머지 상태 단정 전 **`git fetch origin` 먼저.** `origin/main`이 통합 트렁크다
   (diff-drive-finetune은 이미 머지됨 + phase4-B 123커밋 더 있음).
3. **`ARCHITECTURE.md`의 S1~S6 성능표는 신뢰 금지**(§4 오염 목록).

---

## 1. 체크포인트 인벤토리 (git에 실재하는 것 전부)

| 모델 파일 | 정체 | 커밋일 | obs | eval json? |
|---|---|---|---|---|
| `warehouse_mappo_extobs/model_shelf_final.pt` | **model_800, scenario-only "선반회피 최종"** | 2026-06-03 | 19D | ❌ 없음 |
| `warehouse_mappo_extobs/model_600.pt` | scenario학습 best | 2026-06-03 | 19D | ❌ 없음 |
| `warehouse_mappo/model_10998.pt` | 17D 홀로노믹 ("S6 98%" 주장) | 2026-06-01 | 17D | ❌ 없음 |
| `warehouse_mappo/model_9999.pt` | 17D, 구버전 | 2026-06-01 | 17D | ✅ `eval_17dim_final` |
| `warehouse_mappo/model_4999.pt` | True CTDE MAPPO | 2026-06-01 | 17D | ✅ `eval_mappo_ctde` |
| `warehouse_mappo/model_5399.pt` | 가짜 CTDE(실패) | 2026-06-01 | 17D | ✅ `eval_mappo` (10ep) |
| `warehouse_ippo/model_400.pt` | IPPO baseline | 2026-06-01 | 17D | ✅ `eval_ippo` (10ep) |
| (없음) `model_11998` | safedist18 변형 | — | 17D | ✅ json만 있고 .pt 없음 |

→ **모순:** 성능이 좋다고 *주장*되는 신모델(shelf_final/600/10998)은 **eval json이 없고**,
eval json이 있는 건 전부 **구버전**이다. 검증과 주장이 어긋나 있다.

---

## 2. ✅ 검증된 숫자 (커밋된 eval json = 유일한 ground truth)

도달률(all_reached_rate). col=충돌, dl=교착. **⚠️ 주의: 시나리오 정의가 버전마다 다름**
— `shelf_final`은 신 eval_scenarios.py(S5=혼잡통로3방향, S6 추가) + `--enable_obstacles`,
나머지 구버전은 구 시나리오 정의 + obstacles OFF. **S-번호 직접 비교 금지**(§7-2 참조).

**핵심 비교 — scenario 학습 전(10998) vs 후(shelf_final), 둘 다 obs ON·신 eval정의·100ep:**

| 모델 (json) | 조건 | S1 | S2 | S3 | S4 | S5 | S6 | 평균충돌 | 평균도달 |
|---|---|---|---|---|---|---|---|---|---|
| **★ 10998_newdef** (검증·최강) | 17D, obs ON | 78%(dl22) | 94%(dl6) | 13% | 1% | **91%**(dl9) | **98%**(dl2) | **0.0%** | **62.5%** |
| shelf_final (scenario, 회귀) | 19D, obs ON | 93% | 100% | 0%(col) | 1%(col) | 78%(dl22) | **6%**(dl94) | 21.7% | 46.3% |

구세대(obs 차원·정의 달라 위와 직접비교 불가, 참고용):
| 9999 | 17D obsOFF | 93|84|95|0|0|— · 4999 | 17D 9D실은 9D obsOFF | (현 env 차원불일치) · 5399/400 = 9D 구버전

**검증된 사실 (2026-06-09, 반전):**
- **scenario 학습은 순(純) 회귀였다.** S1/S2만 소폭 오르고(78→93, 94→100), **S6 98→6 붕괴 ·
  S5 91→78 · 충돌 0→21.7% · 평균도달 62.5→46.3.** 커밋 "선반회피 최종 도달100" = 실제론 더 나쁨.
- **catastrophic forgetting 입증:** 19D는 장애물 방향 obs를 *더* 가졌는데도 S6가 더 나쁨 →
  능력부족 아니라 **scenario families(A/B/C)에 동적장애물이 빠져 분포가 좁아진 탓.**
  = "teaching-to-the-test → 일반화 상실" 깨끗한 negative result. (논문 핵심)
- **10998의 S6 98%는 obstacles ON에서 재현됨 → 진짜다.** (이전에 §4에서 "오염"이라 적은 건 정정:
  오염된 건 ARCHITECTURE의 *표 숫자*였지 10998의 S6 능력이 아님.)
- 배포/데모용 최강 모델 = **10998** (충돌0%, S6 98%, 평균도달 최고). shelf_final 쓰지 말 것.
- S3(배터리obs 없는 17D 불공정)·S4(정의상 교착)는 비교대상 아님.

---

## 3. ⚠️ 주장됐으나 미검증 (커밋 메시지·문서뿐, eval json 없음)

이 숫자들은 **재현 확인 전까지 논문·발표에 인용 금지.**

| 출처 | 모델 | 주장 | 검증 상태 |
|---|---|---|---|
| 커밋 `77e9270` (06-03) | model_shelf_final(=800) | S1 100·S2 100·**S5교착 0**·도달 100 | ❌ **반증됨(06-09)** — 실측 S5교착 22%·도달 46%·S6 6%. §2 참조 |
| 커밋 `fc2afee` (06-03) | model_600 (scenario best) | S1 100·S2 100·S5교착 0% | ⏳ 미검증 (shelf_final과 동류라 비슷할 듯 = 과장 의심) |
| `ARCHITECTURE.md`/docs | model_10998 | S6 98% | ✅ **검증됨**(06-09 obstacles ON에서 98% 재현) |
| `DEMO_19D` §5.1 | 19D warm-start model_1200 | S2 100%충돌·S5 37%(퇴행) | (scenario학습으로 대체) |

→ 둘 다 검증 완료(§2). 결론: 10998이 최강, shelf_final은 회귀. **남은 임계 경로 = 반사실 실험(§7).**

---

## 4. ❌ 오염/폐기 — 신뢰 금지 목록

| 항목 | 문제 | 근거 |
|---|---|---|
| **`ARCHITECTURE.md` S1~S6 성능표** (S6 97·S3 8·S4 0·S1 71) | **어느 검증 모델과도 안 맞음**(S1 71? 실측 78, S3 8? 실측 13). S6 97만 우연히 근사(10998 실측 98). 표를 그대로 인용 금지 — §2 검증값 사용 | §2 json |
| "RL이 S3/S4 못 함 = LLM 필요"(레이어 경계 서사) | 절반 미검증. orchestrator가 S3/S4 *푸는* 건 안 보여줌. 단 phase4-B(§5)는 미독해 | `orchestrator.py`는 옛 스텁뿐 봄 |
| MPG 이론근거 **arXiv:2503.22867** | **peer-review 미통과 preprint**(v2 2025-11) + 후속 2603.19188(2026-03)도 preprint. Theorem 5를 "보장"으로 인용 금지 | arXiv 페이지(Journal-ref 없음) |
| 배터리/충전 RL 통합 | 폐기됨(navigation 붕괴). env엔 남기되 `enable_battery=False` | `ARCHITECTURE.md` §배터리 |

---

## 5. ⛔ 내가(Claude) 아직 안 읽은 영역 — 판단 보류

- **phase4-B (origin/main의 123커밋, ~06월).** Nav2 멀티스택, 충전 데모, A2A 멀티에이전트
  대화, orchestrator 결정로그, 발표자료. **orchestrator가 스텁이라는 §4 판단은 이걸
  읽기 전엔 잠정.** B(LLM 전역조율) 주제를 다시 보려면 여기부터.
- diff-drive 계층 컨트롤러(`--diff_drive_ctrl`)의 회피 유지율 — `DIFFDRIVE_PROGRESS.md`가
  "검증 대기"라 적음. eval 미실시.

---

## 6. 문서 인덱스 — 현행/이력/오염 (날짜순)

| 문서 | 날짜 | 상태 | 비고 |
|---|---|---|---|
| **STATUS.md** (이 파일) | 2026-06-09 | 🟢 현행·최우선 | |
| DEMO_FINAL_STATUS.md | 2026-06-04 | 🟢 현행 | 데모 최종 현황 (scenario 학습 이후) |
| DEMO_19D_AVOIDANCE_INVESTIGATION.md | 2026-06-02 | 🟡 이력(중요) | 19D 실패 진단·§9 교훈. scenario학습 *직전* 상태 |
| PROJECT_STATUS.md | 2026-06-01 | 🟡 이력 | scenario학습 전 |
| DIFFDRIVE_PROGRESS.md | 2026-06-01 | 🟡 이력 | (D)컨트롤러 "검증 대기"에서 멈춤 |
| ARCHITECTURE.md | 2026-05-31 | 🔴 **오염** | 성능표 신뢰 금지(§4). 레이어 설계 *개념*만 참고 |
| SESSION_PROGRESS.md | 2026-05-30 | ⚪ 이력 | |
| transport_env_debug.md | 2026-05-29 | ⚪ 이력 | Phase2 |
| paperspace_setup_notes.md | 2026-05-28 | ⚪ 참고 | 구 환경(RunPod로 대체) |
| phase2_transport_debug.md | 2026-05-27 | ⚪ 이력 | Phase2 |
| PROJECT_DESIGN.md | 2026-05-25 | ⚪ 이력 | 초기 설계 |
| PHASE3_PROGRESS.md | 2026-05-23 | ⚪ 이력 | |
| PHASE3_IPPO_VS_MAPPO.md | 2026-05-18 | 🟡 이력(신뢰O) | 표가 eval json(4999/5399/400)과 일치. 단 scenario 이전 |
| MARL_MPG_DECISION.md | 2026-05-15 | 🟡 이력 | MPG 설계근거. 근거논문=preprint(§4) |
| PHASE1_PROGRESS.md / TRAINING_TIME_ESTIMATION.md | 2026-05-15 | ⚪ 이력 | |

범례: 🟢현행 · 🟡이력이나 중요/신뢰가능 · ⚪단순이력 · 🔴오염(신뢰금지)

---

## 7. 다음 액션 (우선순위)

1. ✅ **(완료 06-09)** shelf_final + 10998 둘 다 검증(§2). **결론 반전: 10998 최강,
   scenario 학습(shelf_final)은 순 회귀 + S6 catastrophic forgetting(98→6).**
   - obs 세대: 4999/5399/400=**9D**(현 env 차원불일치로 못돌림), 9999/10998/11998=**17D**,
     600/shelf_final=**19D**. 10998↔shelf_final 비교는 obs차원 confound 있음(논문 limitation).
2. ⏳ **반사실 실험 (논문의 마지막 못)** — scenario 학습을 *동적장애물 포함*으로 1회 재학습 →
   S6 회복되면 "S6 붕괴 원인 = 분포 협소화" 인과 확정.
   ```
   python training/multi_robot/train_marl.py --ippo_ckpt logs/warehouse_mappo/model_10998.pt \
     --extended_obs --scenario_train --enable_dynamic_obstacles --num_envs 256 --headless
   ```
   (학습 후 동일 eval로 S6 재측정. 회복=가설확정 / 미회복=다른원인.)
3. 데모/배포: **10998 확정 사용**(shelf_final 폐기).
4. phase4-B 코드 독해(§5) → orchestrator 실태 확정 → B논문 재판단.

### 논문 방향 (검증 완료)
**"넓은 랜덤골 MARL(MPG+CTDE)은 동적장애물 회피 포함 견고(10998: S6 98%·충돌0%); eval
시나리오에 맞춘 curriculum은 타겟(S1/S2)을 올리는 대신 분포 밖 S6를 파국적으로 망각(98→6)하고
안전을 회귀(충돌 0→22%)시킨다 — MARL의 teaching-to-the-test 경계 사례."**
- 이미 검증 eval 2개로 뒷받침. §7-2 반사실로 인과 닫으면 끝. obs차원 confound는 limitation 명시.

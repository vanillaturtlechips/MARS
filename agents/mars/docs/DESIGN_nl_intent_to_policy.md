# 설계: 자연어 운영자 의도 → 검증된 Fleet 정책 (강 claim)

> ## ⚠️ STATUS: 최신 (CURRENT) — 2026-06-25
> **이 문서가 현재 연구 방향의 최신 설계다.** 이걸 먼저 읽어라.
>
> 저장소의 다른 문서들(`docs/PHASE*`, `PROJECT_*`, `mars_architecture_v2.md`,
> Isaac/Nav2/SLAM/RL 관련 일체, `ISAAC6_SLAM_PROGRESS.md`, S1~S6 시나리오 작업)은
> **과거 자료다.** 환경/시뮬 구축 기록이거나 이전 스코프라서, **다음에 읽을 때
> 현재 방향과 맞는지 반드시 재확인할 것.** 특히:
> - SLAM/멀티로봇 Nav2/RL 하이브리드/OpenCV = 환경(physics) 레이어. 논문 기여 아님.
> - 논문 기여 = **에이전트 레이어**. 그중 이 문서(강)와 `eval/SCHEMA.md`(중강)가 현행.
>
> **관계:** `eval/SCHEMA.md`(중강: 근거기반 진단 + 환각차단 검증)는 **근시일 평가**.
> 이 문서(강: NL 의도→정책)는 **novelty가 더 높은 방향**이며 같은 검증 인프라를
> 공유한다. 둘 다 헤드라인은 "신뢰 못 할 LLM 출력을 결정론적 검증으로 안전하게".

---

## 1. Claim (한 줄)

사람이 평상어로 운영 의도를 말하면("5번 통로 한 시간 비워", "콜드체인 우선",
"충전기 임계 로봇 전용으로") 에이전트가 그걸 **fleet 정책으로 번역**하고,
**결정론적 검증/guardrail이 위험·모순·실행불가·범위초과 의도를 거부·강등**해
— LLM이 환각해도 안전한 fleet 재구성을 한다.

## 2. 왜 논문이 되나 (2026) / 에이전트만 할 수 있는 일

**로봇·고전 스택이 못 하는 것:**
- 로봇은 모호하고 시간에 따라 바뀌는 인간 의도를 해석 못 함.
- 고전 fleet/WMS는 정책마다 **엔지니어가 코딩**해야 함 — 실무 최대 통증: fleet를
  바꾸려면 사람이 붙고 배포가 필요.
- 에이전트는 **open-vocabulary 의도 → 제약된 정책 공간** 매핑 (조건·시간·조합 포함).

**2026 gap (붐비는 영역과 차별):** "LLM이 NL→정책"은 누구나 떠올린다. 안 풀린 건
**안전**이다 — 순진한 NL→정책은 모순/위험/실행불가 정책을 만든다("모든 통로 막아"
→ fleet 정지). 대부분 LLM-로봇 논문이 이 구멍을 무시한다. **기여 = 전역 fleet
상태에 대해 NL 의도를 검증·거부하는 레이어**, 그리고 **raw LLM은 위험하고 검증이
안전하게 만든다는 정량 증명.**

## 3. 아키텍처 (데이터 흐름)

```
운영자 자연어 의도 ("아이슬5 1시간 정비, 비워둬")
        │
        ▼
[Intent Agent] ── LLM ── NL → 구조화 후보 정책(들)
        │   {type, zone, params, duration, condition}  (type은 WHITELIST에서만)
        ▼
[Decision Validator]  ── 신뢰도/근거/일관성 게이팅 (중강과 공유)
        ▼
[Policy Guardrail]  ── 결정론적 안전 검사:
        │   • whitelist 멤버십 (미정의 정책 type 거부)
        │   • 스키마/파라미터 범위
        │   • duration ∈ [MIN=60s, MAX=7200s], cooldown 준수
        │   • 활성 정책과 모순/중복 금지
        │   • ★ 전역 fleet-state 효과 검사 (예: 이 정책이 가용 로봇/충전기를
        │     임계 이하로 떨어뜨리면 거부) — fleet 정지 유발 의도 차단
        ▼
[Policy Manager]  ── 활성화/만료/우선순위
        ▼
효과: Scheduler(미션 스킵/지연) + Nav2(keepout mask reroute) + Charging Service
```

핵심: **번역(Intent Agent)과 안전(Validator+Guardrail)을 분리.** 번역은 LLM(불신뢰),
안전은 결정론(신뢰). 행동에 닿는 모든 정책은 결정론 게이트를 통과해야 한다
(reason/execute 분리 원칙).

## 4. 코드 매핑 (있음 / 빌드 필요)

**있음 (재사용):**
- `config.POLICY_WHITELIST` (5종) + `POLICY_{MIN,MAX}_DURATION_SEC`, `COOLDOWN`
- `mars/agents/operations_strategy.py` — whitelist에서만 정책 추천 (번역 로직의 출발점)
- `mars/guardrail/guardrail.py` — 스키마+whitelist+안전검사
- `mars/policy/policy_manager.py` — 활성/만료/우선순위
- `mars/validators/decision_validator.py`, `retrieval_validator.py` — grounding 게이팅
- 블랙보드 + 효과 경로(avoid_zone→keepout→Nav2, scheduler 스킵)

**빌드 필요:**
- **Intent Agent 프론트엔드** — NL 문장 입력 → 후보 정책(들). 현재 트리거는 "실패
  이벤트"지 "사람 문장"이 아님. 이게 핵심 신규 모듈.
- **전역 fleet-state 효과 검사 강화** — guardrail에 "이 정책 적용 후 가용 로봇/충전
  capacity가 안전선 이하인가" 같은 feasibility 검사 추가.
- **Intent 평가 데이터셋** (§6).
- 조건/시간 의도("다음 1시간", "X면 Y")의 파라미터화 표현.

## 5. 표현 가능한 정책 공간 (현재 whitelist)

| 정책 type | NL 의도 예시 |
|---|---|
| `avoid_zone` | "아이슬5 비워둬 / 거기 가지 마" |
| `delay_low_priority_missions` | "급한 것만, 나머지 뒤로" |
| `reserve_chargers_for_critical` | "충전기 임계 로봇 전용" |
| `lower_target_charge_level` | "빨리빨리 돌려, 80%까지만 충전" |
| `pre_charge_for_demand_spike` | "이따 물량 몰려, 미리 충전" |

→ 평가에서 의도는 이 공간 안/밖을 섞어 낸다 (밖이면 guardrail이 거부해야 정답).

## 6. 실험/평가 설계

**데이터셋: 운영자 의도 N개**, 태그별:
- `safe` — whitelist 내, 실행가능, 모순 없음
- `out_of_scope` — 표현 불가 정책 요구 ("로봇 속도 2배" 등 whitelist 밖)
- `unsafe_global` — 적용 시 fleet 마비 ("모든 통로 막아", "모든 충전기 예약")
- `contradictory` — 활성 정책과 충돌
- `infeasible` — 자원 부족 (없는 zone, capacity 초과)
- `ambiguous` — 모호 → 명확화/거부
- `compositional` / `temporal` — 조합·조건·시간 ("아이슬5는 1시간, 그동안 콜드체인 우선")

**메트릭:**
1. **번역 정확도** — 정책(type/zone/params)이 의도와 일치 (safe subset)
2. **안전 catch (헤드라인)** — guardrail이 unsafe/out_of_scope/contradictory/infeasible를
   잡는 정밀·재현율
3. **safety delta** — guardrail **없는 raw LLM**이 위험 정책 통과시킨 비율 vs 우리가
   막은 비율 ← 핵심 그림
4. **false-block rate** — safe 의도를 잘못 거부한 비율 (높으면 시스템 쓸모없음, 반드시 보고)
5. **효과 검증** — 시뮬/스텁에서 정책이 의도대로 fleet 동작을 바꾸는가
6. 추론 필요 의도(compositional/temporal)에서의 정확도

**Baseline:**
1. **raw LLM** — guardrail 끄고 NL→정책 직행 (safety delta용)
2. **키워드/규칙 파서** — 고전 NL→정책 (조합·모호 의도에서 무력함을 보임)

## 7. 리뷰어 반박 & 방어

- **"intent 분류 / 멋진 설정 UI 아냐?"** → novelty는 번역이 아니라 **open-ended NL을
  전역 제약에 대해 안전 검증**하는 것 + raw LLM이 위험함을 정량 증명. 추론 필요한
  의도(조합·조건·시간)로 평가해 단순 분류와 구분.
- **"왜 LLM, 규칙 파서면?"** → 규칙 파서는 사전 정의 표현만. 조합·모호·신규 어휘
  의도에서 실패함을 baseline으로 제시.
- **"실시간성?"** → LLM은 **느린 감독 레이어**(운영자 상호작용/재구성)지 제어루프가
  아님. 정책 적용은 비실시간이라 정당.
- **"안전을 어떻게 보장? LLM이 guardrail도 속이면?"** → guardrail은 **결정론**
  (LLM 아님). whitelist·범위·전역효과 검사는 코드. LLM은 후보만 제안, 통과는 코드가 결정.

## 8. 다음 단계 (구현 순서)
1. 의도 데이터셋 스키마 + N개 작성 (§6 태그별)
2. Intent Agent 프론트엔드 (NL → 후보 정책, whitelist 강제)
3. guardrail 전역 fleet-state feasibility 검사 강화
4. 러너: 의도 → 번역 → 검증 → (스텁) 효과; 메트릭 집계
5. raw LLM / 규칙 baseline 구현 → safety delta 측정

## 9. 한 줄 요약
**에이전트가 로봇/고전이 못 하는 일 = "사람 말로 fleet를 즉석 재구성", 그리고 그걸
안전하게 만드는 결정론적 검증이 기여다.** 환경(SLAM/Nav2/RL)은 데모 배경일 뿐 기여 아님.

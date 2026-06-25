# 진단 평가 결과 (중강) — 최종

데이터: 150 cases (dev 50 / test 100), 다양한 텍스트 풀 + 난이도 + 수치 랜덤.
모델: OpenAI gpt-4.1-mini (진단), text-embedding-3-small (RAG). 프롬프트는 dev로만
튜닝 후 동결 — test는 unseen. 일자: 2026-06-25.

## 헤드라인 (test n=100)

| 지표 | RAG ON | RAG OFF |
|---|---|---|
| cause 정확도 | **83%** | 43% |
| scope 정확도 | 93% | 90% |
| confident-wrong (틀린 확신) | **0%** | 11% |
| acted-precision (PASS 행동의 정답률) | **84.5%** | 43.5% |
| 거절(unknown) | 29% | 62% |
| over-block (정답인데 DEGRADE) | 12% | 12% |

- **RAG: cause 43→83%, confident-wrong 11→0%, acted-precision 43.5→84.5%.**
  RAG는 정확도뿐 아니라 **안전**도 올림 — 틀릴 때 오도(confident-wrong)하지 않고 거절.
- **fail-safe:** RAG ON 오답은 전부 `unknown`(거절). 확신해서 틀린 행동 0건.

## 난이도별 (RAG ON)
| | cause | scope |
|---|---|---|
| easy | 100% | 100% |
| medium | 100% | 100% |
| hard | 55% | 82% |
모든 오답이 hard(센서 adversarial, fleet_wide)에 집중.

## Retrieval 계측 (precedent 있는 78케이스, RAG ON)
searched 100% · relevant retrieved 100% · **relied 8%**
→ 검색은 완벽; 병목은 **활용**. (RAG OFF: searched 0% — precedent 미시드, 의도된 ablation)

## Validator (안전의 두 번째 기둥 — 별도 입증)
- validator probe 30/30 (100%): 근거 없는/과신/scope-미지원 진단 차단 (block P/R 1.0).
- 단, **근거 있는데 틀린** 진단은 통과 → benign test의 safety-delta=0. validator 가치는
  적대적 probe에서 증명; benign test는 에이전트 거절 행동이 안전을 담당.

## 한계 (정직)
- **fleet_wide**: cause unknown + scope 오판. precedent 검색은 되나 활용 못 함.
- 센서 adversarial(증상≠원인): unknown으로 안전하게 거절(오답이지만 안전).
- over-block 12%: 정밀/재현 트레이드오프 비용.

## 재현
docker compose up -d (pgvector 1536) + .env(OPENAI_API_KEY, LLM_PROVIDER=openai,
EMBEDDING_PROVIDER=openai, EMBEDDING_DIM=1536) →
`python3 -m eval.run_diagnosis --rag both --split test` →
`python3 -m eval.safety_delta test`

## 한계 원인 규명 (analyze_limits, 재집계·LLM 0)
- **fleet_wide / 센서 adversarial (precedent 19개 끌어왔는데 오답):** 19개 전부
  거절(unknown), used-but-wrong 0. relevant precedent **trust 0.71**(정답 케이스와
  동일) → **검색·신뢰 문제 아님.** 충분히 신뢰할 precedent가 있는데도 모델이
  hard 케이스에서 **답으로 통합 못 함 = 추론 한계.** (future work: 더 센 모델/추론 보강)
- **over-block 13개:** 12개가 confidence<0.5 → 에이전트가 불확실했는데 우연히
  맞은 진단을 validator가 강등. **버그 아닌 보수적 안전 비용** (tau↓ 시 over-block↓
  but confident-wrong↑ 트레이드오프).
- **confident-wrong = 0:** RAG ON에서 틀린-확신-PASS 0건. 시스템이 위험한 잘못된
  행동을 한 번도 내지 않음(안전 입증). 잔여 위험은 grounded-but-wrong인데 이 셋엔 없음.

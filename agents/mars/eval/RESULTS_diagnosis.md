# 진단 평가 결과 (중강 Part 1) — 1차

실행: `python3 -m eval.run_diagnosis --rag both --limit 0`
모델: OpenAI gpt-4.1-mini (진단), text-embedding-3-small (RAG), 30 cases.
일자: 2026-06-25 (1차 — 데이터셋/프롬프트 개선 전 baseline).

## 헤드라인

| | cause 정확도 | scope 정확도 | verdict (PASS/DEGRADE) |
|---|---|---|---|
| **RAG ON**  | **23/30 (76.7%)** | 12/30 (40.0%) | 21 / 9 |
| **RAG OFF** | **7/30 (23.3%)**  | 2/30 (6.7%)   | 9 / 21 |

- **RAG가 cause 정확도를 23.3% → 76.7% (3.3×) 향상.** = 메인 기여.
- **RAG OFF → DEGRADE 70%**: 증거 없으면 에이전트가 `unknown`/저신뢰로 떨어지고
  validator가 DEGRADE → **환각 대신 안전 거절**. = 검증/안전 주장 입증.

## 분해

- **scope:** RAG ON 정답 12개 = zone_wide 12개 **전부 ✓**.
  isolated(16) + fleet_wide(2)는 **전부 ✗** — 에이전트가 단일/fleet을 일관되게
  `robot_specific`으로 출력. (프롬프트 spec은 "단일→isolated"인데 미준수.)
  → 한계 + 프롬프트 개선 실험거리.
- **adversarial:** DC-004/006(estop 증상→low_battery) 일부 오진, DC-014~016(센서
  결함→robot_internal_fault) 오진. 증상에 낚이는 hard case.
- **RAG 메커니즘:** zone_blocked/congestion/localization 케이스가 RAG OFF에선
  `unknown`(증거 없음)으로 떨어지고 RAG ON에선 precedent로 정답. RAG가 "novel
  cause를 과거사례로 식별"하는 경로가 데이터로 확인됨.

## 다음 (2차 개선거리)
1. scope: isolated/robot_specific 혼동 — 프롬프트 명확화 후 재측정.
2. fleet_wide 인식 실패 — distribution 신호를 프롬프트에서 강조.
3. adversarial(증상≠원인) 케이스 — RAG precedent 강화 / 검증 규칙.
4. validator probe(30, 100%)와 합쳐 "근거검증+RAG" 통합 figure.

## 재현
`docker compose up -d` (Postgres pgvector 1536) + `.env`(OPENAI_API_KEY,
LLM_PROVIDER=openai, EMBEDDING_PROVIDER=openai, EMBEDDING_DIM=1536) →
`python3 -m eval.run_diagnosis --rag both --limit 0`.

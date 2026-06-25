# 멀티모델 종합 결과 (3 models) — 최종

모델: gpt-4.1-mini, claude-haiku-4-5, Upstage solar-pro. 동일 데이터셋/프롬프트(동결).
진단 test n=100, 의도 test n=39. 임베딩 로컬(bge-small) 또는 OpenAI(모델 무관 RAG).

## 진단 (cause/scope %, precedent 활용, 안전)
| model | cause(ON) | cause(OFF) | scope(ON) | relied% | conf-wrong | acted-prec |
|---|---|---|---|---|---|---|
| gpt-4.1-mini | 81 | 43 | 93 | 9 | 0 | 82 |
| haiku | 93 | 52 | 96 | 92 | 3 | 96 |
| solar | 82 | 35 | 73 | 78 | 0 | 87 |

## 의도 (overall %, must-not 15개 defense-in-depth)
| model | overall | declined(agent) | blocked(guardrail) | leaked |
|---|---|---|---|---|
| gpt-4.1-mini | 77 | 6 | 7 | 2 |
| haiku | 87 | 11 | 4 | 0 |
| solar | 72 | 4 | 7 | 4 |

## 핵심 발견 (멀티모델로 확정)
1. **RAG는 모델 무관하게 cause를 크게 올림** (+38/+41/+47pp). 일관된 이득.
2. **실패 양식은 모델 의존적:** mini는 모르면 거절(unknown→안전), haiku/solar는 추측
   → RAG 없을 때 confident-wrong이 모델마다 다름.
3. **precedent 활용률이 모델 의존적** (9/92/78%) → 진단 정확도 격차의 주원인. "검색은
   되는데 활용 못 함"은 약한 모델(mini)의 추론 한계.
4. **의도 누설(force-fit)도 모델 의존적** (haiku 0 ~ solar 4). guardrail이 못 잡는
   "유효하나 의도와 다른 정책"이라, 약한 자기억제 모델일수록 더 샘.
5. **decision validator의 safety-delta ≈ 0 (전 모델)** — 근거없음/과신은 잡지만
   grounded-but-wrong은 통과. 안전은 RAG + 거절 + guardrail의 상호작용에서 나옴.

## 한 줄
검증·RAG는 **모델 독립적 이득**을 주지만, 잔여 위험(grounded-but-wrong, force-fit)은
**모델 의존적**이며 결정론 검증만으로 못 막는다 = 검증은 필요조건이지 충분조건 아님.

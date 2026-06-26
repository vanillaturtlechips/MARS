# Cover Letter — Journal of Korea Robotics Society (JKROS)

*Fill into KROS_Cover_Letter_Form_0801.docx. English draft; a Korean version
follows. Replace [date] and confirm the paper type before submission.*

---

To the Editor-in-Chief,
Journal of Korea Robotics Society

Date: [date]

Dear Editor,

I am pleased to submit our manuscript, **"Deterministic Validation of LLM
Supervisory Agents for Warehouse Robot Fleets: A Multi-Model Study,"** for
consideration for publication in the Journal of Korea Robotics Society as a
**research (academic) paper**.

Large language models are increasingly proposed as a supervisory layer for
warehouse robot fleets — to diagnose mission failures and to translate operator
instructions into fleet policies — but their unreliability (hallucinated
diagnoses, misread instructions) makes them unsafe to act on directly. This
manuscript studies the machinery that makes such unreliable agent output safe to
act on. We present MARS, a supervisory architecture that gates LLM output and
input through deterministic validation, and we evaluate it on controlled
diagnosis (n=100) and operator-intent (n=39) test sets across three different
LLMs (GPT-4.1-mini, Claude Haiku 4.5, and Upstage Solar-Pro).

Our main contributions are: (1) a supervisory architecture separating LLM
"reasoning" from validated "action" via a decision validator and a policy
guardrail; (2) a multi-model evaluation showing that retrieval-augmented
generation improves diagnosis accuracy by 38–47 percentage points on every model
and that an agent-plus-guardrail defense blocks 73–100% of unsafe operator
intents; and (3) a characterization of where deterministic validation stops —
it catches structurally invalid output but not grounded-but-wrong diagnoses or
valid-but-unintended policies, a limit we further show is exploited when the
agent is incentivized to be accepted (it games the self-reported confidence gate
but not the externally grounded evidence check).

We confirm that this manuscript is original, has not been published elsewhere,
and is not under review by any other journal or conference. The author declares
no conflict of interest. All datasets, evaluation scripts, and figures are
reproducible from the released code.

Thank you for considering our submission.

Sincerely,
Myong-Il Lee
Dept. of Cyber Security, Korea Polytechnic University Gangseo Campus, Seoul, Korea
2220110150@office.kopo.ac.kr

---

## 국문본 (참고용 — 양식은 영문 제출 권장)

편집위원장님께,

논문 **"Deterministic Validation of LLM Supervisory Agents for Warehouse Robot
Fleets: A Multi-Model Study"**를 로봇학회 논문지에 **학술논문**으로 투고합니다.

대규모 언어모델(LLM)은 물류 로봇 fleet의 감독 계층(실패 진단, 운영자 지시→정책 번역)
으로 제안되고 있으나, 환각·오독으로 인해 직접 행동에 옮기기엔 신뢰성이 부족합니다. 본
논문은 신뢰할 수 없는 에이전트 출력을 안전하게 행동으로 옮기게 하는 장치를 연구합니다.
LLM 입출력을 결정론적 검증으로 게이팅하는 감독 아키텍처 MARS를 제안하고, 통제된 진단
(n=100)·의도(n=39) test 셋에서 3개 LLM으로 평가했습니다.

주요 기여: (1) 결정 검증기·정책 가드레일로 LLM "추론"과 검증된 "행동"을 분리하는 감독
아키텍처; (2) RAG가 모든 모델에서 진단 정확도를 38–47%p 올리고 에이전트+가드레일이
위험 의도의 73–100%를 차단함을 보인 다중 모델 평가; (3) 결정론적 검증의 한계 규명 —
구조적 오류는 잡지만 근거 있으나 틀린 진단/유효하나 의도와 다른 정책은 못 잡으며,
에이전트가 수용 인센티브를 받으면 자기보고 confidence 게이트는 게이밍되나 외부검증
근거화 검사는 뚫리지 않음을 확인.

본 논문은 독창적이며 타 학술지·학회에 게재되거나 심사 중이지 않습니다. 이해상충 없음.
모든 데이터셋·평가 스크립트·그림은 공개 코드로 재현 가능합니다.

이명일 드림
한국폴리텍대학 강서캠퍼스 사이버보안과
2220110150@office.kopo.ac.kr

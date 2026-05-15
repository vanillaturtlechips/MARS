# MARL 보상 설계 변경 결정 기록

> MARS 프로젝트 — Multi-Agent Robot System

---

## 배경

Phase 3 (멀티 로봇 협력) 설계 중, 로봇들이 좁은 통로에서 서로 양보하다가
둘 다 멈추는 **교착(deadlock) 문제**를 어떻게 해결할지 논의했다.

---

## 기존 설계 (임의 휴리스틱)

```python
# 처음 설계한 포텐셜 함수
if both_yielding(duration > 2s):
    phi -= 10   # 교착 페널티
if one_yielding_other_moving:
    phi += 2    # 협력 보상
if robot_i.battery < robot_j.battery:
    phi += 5    # 우선순위 보너스
```

**문제점:**
1. **이진 임계값** — "2초"라는 기준이 임의적. 1.9초는 패널티 없음
2. **불연속** — 경계에서 그래디언트가 0 또는 무한대 → 훈련 불안정
3. **MPG인지 불명확** — 이 구조가 실제로 Markov Potential Game인지 사후 검증 필요
4. **수렴 보장 없음** — Nash Equilibrium 존재 여부 이론적 보장 없음

---

## 변경 이유

### 논문 발견

> Yan, H. & Liu, M. (2025). *Markov Potential Game Construction and
> Multi-Agent Reinforcement Learning with Applications to Autonomous Driving.*
> arXiv:2503.22867v2. Virginia Tech. (DARPA 지원)

이 논문이 정확히 우리가 풀려는 문제의 수학적 해답을 제공한다.

### 핵심 기여: MPG를 어떻게 만드는가

논문 이전까지 MPG의 "조건"은 알려져 있었지만, **어떻게 설계하면 MPG가 되는지**
구체적인 방법이 없었다. 이 논문은 충분조건을 constructive하게 제시한다.

**Theorem 5 (핵심):**

```
보상을 아래 구조로 설계하면 → 게임이 자동으로 MPG

rᵢ = α · rᵢ^self(sᵢ, aᵢ)
   + β · Σⱼ≠ᵢ rᵢⱼ(sᵢ, sⱼ, aᵢ, aⱼ)   단, rᵢⱼ = rⱼᵢ (대칭)
```

MPG가 되면 다음이 자동으로 보장된다:
- **순수 Nash Equilibrium 존재** (Proposition 1)
- **Gradient play → NE 수렴** (Theorem 2)
- **MAPPO도 동일하게 수렴** (Leonardos et al., ICLR 2022 [ref 21])

### 새로운 보상 함수

```
자기 보상 (Theorem 3 충족):
  rᵢ^self = -||posᵢ - goalᵢ||²  +  10·[reached_goal]  -  0.01·t

쌍별 충돌 회피 보상 (Theorem 4 충족, 논문 수식 35):
  rᵢⱼ = -1 / sqrt((xᵢ-xⱼ)² + (yᵢ-yⱼ)² + ε),   ε = 1e-5
```

**기존 대비 개선점:**

| | 기존 (휴리스틱) | 변경 후 (논문 기반) |
|--|--------------|------------------|
| 연속성 | ❌ 이진 임계값 | ✅ 항상 연속 |
| 그래디언트 | ❌ 불연속 구간 존재 | ✅ 어디서든 흐름 |
| MPG 보장 | ❌ 사후 검증 필요 | ✅ 설계 자체가 보장 |
| NE 수렴 | ❌ 미보장 | ✅ 수학적 보장 |
| 교착 해소 | 휴리스틱 | ✅ NE가 아니므로 자연 해소 |

### 논문 실험 결과

자율주행 4-차량 교차로 시나리오 (100회 랜덤 초기화):

| 방법 | 충돌률 | 비고 |
|------|--------|------|
| MPG-MARL (우리가 채택) | **0/100** | 주변 차량이 예상 밖 행동해도 |
| Single-agent RL | 11~45/100 | 조건에 따라 크게 차이 |

→ 창고 환경의 안전 요구사항 (목표: 충돌률 < 1%)에 부합

---

## 적용 방식

자율주행과 창고 로봇의 구조적 유사성:

```
논문 (자율주행)              MARS (창고 로봇)
─────────────────────────────────────────────
속도 추적                    목표 지점 도달
  -(v - v_d)²         →       -||pos - goal||²

충돌 회피 (동일)
  -1/√(Δx²+Δy²+ε)    →       그대로 사용

1D 종방향 이동              2D 평면 이동
  a ∈ [-g, g]         →       (vx, vy, ω)

4대 차량                     N대 창고 로봇
```

`rᵢⱼ` 수식은 수정 없이 그대로 쓸 수 있다.
2D 창고 공간에서도 유클리드 거리 기반이므로 동일하게 작동한다.

---

## α, β 튜닝 방향

```
r_i = α · r_i^self  +  β · Σ r_ij

창고 우선순위: 안전 > 효율
  → β를 α보다 크게 설정
  → 초기값: α = 1.0, β = 1.5
  → Phase 3 훈련 후 충돌률 보고 조정
```

---

## 남은 검토 사항

- [ ] MAPPO + MPG 보상 조합 Isaac Lab 구현 (`~/MARS/training/multi_robot/potential_reward.py`)
- [ ] α, β 하이퍼파라미터 그리드 탐색
- [ ] 교착 발생률 측정 및 논문 결과와 비교
- [ ] 3대 이상으로 확장 시 rᵢⱼ 쌍 수 = N(N-1)/2 → 스케일링 확인 필요

---

*작성일: 2026-05-15*
*참고: arXiv:2503.22867v2, ~/docs/PROJECT_DESIGN.md*

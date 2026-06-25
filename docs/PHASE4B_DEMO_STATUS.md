# Phase 4-B 에이전트 데모 현황 (2026-06-07)

`deploy/runpod/run_all_demos.sh` — 에이전트(감독 레이어)만 할 수 있는 것을 보여주는 3개
데모를 한 번에 녹화. RL(로봇 단위 반응)은 못 하는 **함대 차원 결정**이 핵심.

- demo1 keepout : 로봇이 통로의 박스에 박힘 → 에이전트가 그 통로 차단 → 다른 로봇은 다른 통로로
- demo2 charging: 충전소 1개에 critical/low 경합 → 큐가 critical 먼저, low 지연
- demo3 priority: 두 로봇이 같은 충전소 경합 → 큐+CAS로 직렬화(데드락 없음)

출력: `/workspace/demo{1,2,3}_*.mp4`

---

## ✅ 지금까지 한 것 (검증됨)

### 에이전트 파이프라인 (로그로 확인)
- **demo1**: 실패 이벤트 → FailureAnalysis(Haiku) → OperationsStrategy → `avoid_zone` 정책
  → KeepoutService → `/keepout_filter_mask` (Nav2). Haiku scope가 비결정적이라 **2대(원래)
  /1대(현재) abort 누적 시 브릿지에서 avoid_zone을 결정적으로 강제**(LLM 진단은 그대로 로그에 남음).
- **demo2/3**: `ChargingService` 우선순위 큐(CRITICAL>LOW)+예약+CAS로 충전소 사용 순서 결정,
  그 순서대로 `isaac_charging_bridge`가 NavigateToPose로 디스패치. `FleetMonitor`→`FleetStateAgent`
  →`OperationsStrategyAgent` LLM 루프도 best-effort 실행(로그). 로그 예:
  `serve order=['R1','R2']` (critical 먼저), `serve order=['R2','R3']` (직렬화).

### 환경/씬 (실측 좌표 기반)
- **핵심 교훈**: 옛 `warehouse_map.pgm`(make_warehouse_map.py 생성)이 **실제 렌더 창고와 안 맞았다**
  — 데모가 건물 밖(x>5, 창고 바닥은 x=5에서 끝)에 찍혀 "허허벌판/회색 벽"으로 나왔다.
- **해결**: `full_warehouse.usd`를 로컬에서 **usd-core로 직접 읽어 실제 좌표 추출**(렌더/GPU 불필요).
  - 선반 줄: x ≈ -20.5 / -15.5 / -10.5 / -5.5 / -0.5 / 4.5, 각 **Y ∈ [8.5, 25]** (북쪽)
  - 진짜 통로: x ≈ -18 / -13 / -8 / -3 / +2
  - 남쪽 Y<8.5 빈 바닥, 바닥 범위 x[-26, 5] y[-20, 28]
- **`deploy/nav2/warehouse_real.{pgm,yaml}`** (= `make_real_map.py`): 위 실제 좌표로 해석적 생성.
  데모 구역만 덮게 20×30m로 축소(origin -18,-2, 200×300 @0.1). map_server가 이걸 로드.
- **demo1을 진짜 통로 x=-8에 배치**: 박스 (-8,15)가 **full_warehouse의 진짜 선반(x=-10.5/-5.5) 사이**.
  가짜 랙 안 씀. 빨간 keepout 슬랩 2×2 @ (-8,15)(에이전트 avoid_zone 시 노출).
- **카메라 = cam_1** (eye -8,2,3 → target -8,18,1.5): 통로 정면. `cam_sweep.py`로 10각도 렌더 후 선택.
  → 이 화각서 **진짜 창고 통로 + 양옆 선반 + 빨간 구역 + 박스가 제대로 나옴(스샷 확인).**
- 충전소 = 실물 `packing_table.usd` @ (0,3). 박스 = 실물 `SM_CardBoxA_01.usd`.

### 시나리오 (demo1, 사용자 확정)
R1·R2·R3가 통로로 같이 진입 → **R1이 통로 x=-8 올라가 박스에 박힘** → 에이전트가 그 통로 차단(빨강)
→ **R2는 우측 통로 x=-3, R3는 좌측 통로 x=-13로 갈라져 우회.** 충돌 방지 직렬화.

### 도구/방법
- `usd-core`(venv) 로 USD 직접 열어 좌표·에셋 치수 읽기 — sim 렌더 없이. 에셋 치수:
  `SM_RackFrame_03` 0.13×1×3(키큰 기둥), `SM_RackPile_06` 3.15×1.37×1.0, `SM_RackShelf_01` 4×1.08×0.37.
  (독립 "키 큰 선반" 단일 에셋은 없음 — full_warehouse가 조립체로 씀.)
- Isaac 에셋 썸네일: S3 `<folder>/.thumbs/256x256/<name>.usd.png` (Pallet엔 있음, Simple_Warehouse엔 없음).
- `cam_sweep.py`: 현재 씬 10각도 → `/tmp/cam_sweep/cam_1..10.png`.

---

## ⏳ 지금 해야 하는 것 (다음 세션 우선순위)

### 1. demo1: 로봇이 화면에 안 들어옴 — 원인 확정
- 로그 사실: `bt_navigator: Begin navigating from (-8,5) to (-8,22)` → **R1 골 받고 출발은 함(nav 정상)**.
  그 뒤 `BT tick rate 100 exceeded`(CPU 부하) 경고만, **reach/abort 로그 둘 다 없음** → R1이 박스
  도달도, 완주도 안 한 채 런 종료. (느리게 기어가다 만 듯/안 움직임.)
- **`-9` 프로세스 사망은 OOM 아님** — 파드 RAM 251GB(여유 212GB). 그 `-9`는 **스크립트 끝 `killall_`
  정상 정리**였음. (이전 OOM 추측은 틀렸음. 맵 축소는 해는 없으나 그게 해결책은 아니었음.)
- **다음 작업**: 최신 `demo1_keepout.mp4` 보고 R1 거동 분류 →
  - (a) 아예 안 움직임 → `/R1/cmd_vel`→OmniGraph DiffController 작동/네임스페이스 점검
  - (b) 박스 앞에서 멈춤 → 정상(그럼 R2/R3 우회가 핵심, 타이밍만 조정)
  - (c) 박스 지나쳐 위로 → 박스 콜라이더/위치 손봄
  - CPU 경합(Isaac RTX + nav2 3스택 + Haiku 브릿지 + 녹화) 가능성도 점검(컨트롤러/코스트맵 주기↓ 등).

### 2. 정리할 잔여
- Nav2 경고: `inflation_radius(0.25) < inscribed(0.353)` → params에서 inflation_radius ~0.4로.
- demo2/demo3(충전)은 새 레이아웃/충전소(0,3)로 **재검증 안 됨** — demo1 정리 후 확인.

---

## 재구축 (파드 새로 올릴 때)
```bash
cd /workspace && git clone git@github.com:vanillaturtlechips/MARS.git ; cd MARS
git checkout feature/mars-msgs-interfaces && git pull
bash deploy/runpod/setup_phase_b.sh     # ROS2+Nav2+mars_msgs+Isaac6.0(py3.12). Isaac 다운 느림/불안정
#   끊기면: nohup bash deploy/runpod/install_isaac_pip.sh > /tmp/isaac_pip.log 2>&1 &  (재시도 루프)
bash deploy/runpod/setup_postgres.sh    # Postgres+pgvector+warehouse+supervisor deps
# agents/mars/.env : DB_DSN / DB_READONLY_DSN / ANTHROPIC_API_KEY / LLM_PROVIDER=anthropic
```
- 맵(`warehouse_real.*`)은 git에 커밋돼 있어 **재생성 불필요**.
- 실행: `bash deploy/runpod/run_all_demos.sh [1|2|3|all]`

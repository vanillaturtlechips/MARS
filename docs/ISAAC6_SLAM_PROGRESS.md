# Isaac Sim 6.0 + ROS2 Jazzy — Warehouse SLAM 진행 상황

작성: 2026-06-24 (RunPod 세션)

## 목표

Isaac Sim 6.0(헤드리스, RunPod) 안 `full_warehouse.usd`에서 iw_hub에 RTX 라이다를
달고, slam_toolbox로 실제 스캔 기반 웨어하우스 맵을 생성한다. (기존
`warehouse_real.pgm`은 5.1 USD 좌표로 손으로 그린 맵이라, 6.0 기준 실측 맵으로 교체)

## 현재 상태: SLAM 동작 확인됨, 맵 채우는 중

- Isaac 6.0 환경에서 `/scan`(~48Hz), `/odom`, `/tf`, `/clock` 발행 정상
- slam_toolbox가 스캔 등록(`Registering sensor: [Custom Described Lidar]`),
  `/map` 발행 확인 (coverage 모니터 기준 ~325 m² / size 16.8x22.6 m 까지 채움)
- 남은 작업: 통로 6개를 빠짐없이 순회해 맵 완성 → `map_saver_cli`로 저장

## 환경 구성 (humble/5.1 → jazzy/6.0 마이그레이션)

RunPod 템플릿: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
(Jazzy는 Ubuntu 24.04 Noble 전용 — 22.04 이미지는 불가)

수정한 스크립트:
- `deploy/runpod/setup_ros2.sh` — Jammy/Humble → Noble/Jazzy 전면 재작성,
  slam-toolbox/vision-msgs 추가
- `deploy/runpod/setup_phase_b.sh`, `install_isaac_pip.sh` —
  venv311→venv312, isaacsim 5.1→6.0, py3.11→3.12
- `deploy/isaac/env_isaac.sh` — venv 경로 312, **ROS2 libs 경로를
  `isaacsim.ros2.bridge`→`isaacsim.ros2.core`** (6.0에서 .so 위치 이동)
- `deploy/isaac/env_ros2.sh` — venv 필터 312
- isaac 스크립트 asset URL `Isaac/5.1`→`6.0`

### Isaac Sim 6.0 pip 설치 핵심

```bash
uv pip install "isaacsim[all,extscache]==6.0.0" \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match --prerelease=allow
```

- **`[all,extscache]` extras 필수** — 손으로 subpackage 나열하면 Kit 익스텐션
  (`isaacsim.app.about` 등)이 빠져 런타임에 registry에서 받으려다 실패
  ("No versions of isaacsim.app.about"). 그 익스텐션들은 pip extscache wheel로만 배포됨
- `unsafe-best-match` (mujoco-usd-converter), `--prerelease=allow` (tinyobjloader rc) 필요

## RTX 라이다 → /scan: 삽질 기록 (다음에 바로 가려고 정리)

`isaac_warehouse_ros2.py`에 라이다 붙이며 부딪힌 순서대로:

1. **`ROS2PublishLaserScan` 노드는 구형 PhysX 라이다용.** RTX 라이다에 직접 연결하면
   토픽만 광고하고 데이터 0. → render product 경유 필요
2. **`IsaacCreateRenderProduct`의 `cameraPrim` 관계 바인딩이 안 됨** →
   "Render product not attached to RTX Lidar". → replicator로 직접 생성으로 우회
3. **`LidarRtx(prim_path=X)`는 X에 Xform 래퍼를 만들고 실제 OmniLidar는 자식**
   (`X/RPLidar_S2E`). render product는 그 OmniLidar 자식에 붙여야 함
4. **render product를 `world.reset()` 전에 만들면 reset이 파괴**
   ("hydratexture already released") → reset 뒤로 이동
5. **(진짜 원인 2개)**
   - `LidarRtx.__init__`이 **자체 render product를 이미 생성**(128x128).
     별도로 또 만들면 충돌 → `LIDAR.get_render_product_path()` 사용
   - `LidarRtx.__del__`이 그 render product를 **destroy**. 객체를 변수에 안 담으면
     즉시 GC → 센서가 조용히 죽음 → **모듈 레벨 참조(`LIDAR`)로 잡아야 함**
6. **publish는 `RtxLidarROS2PublishLaserScan` writer**를 render product 경로에 attach
   (`rep.writers.get(...)`, `.initialize(topicName, frameId)`, `.attach([rp_path])`)

최종 동작 구성 (`isaac_warehouse_ros2.py`, `--slam` 모드):
```
world.reset()
LIDAR = LidarRtx(prim_path=.../lidar, config_file_name="RPLIDAR_S2E", pos=(0.35,0,0.25))
rp = LIDAR.get_render_product_path()
w = rep.writers.get("RtxLidarROS2PublishLaserScan"); w.initialize(topicName="scan", frameId="lidar_link"); w.attach([rp])
```

## TF / 프레임 설계

```
map   ← slam_toolbox (map→odom)
 └ odom    ← Isaac (--slam: odom→base_link, ground-truth map→base_link 아님)
    └ base_link
       └ lidar_link   ← static TF (0.35 0 0.25), slam.launch.py에서 발행
```

- `iw_hub_slam/config/slam_params.yaml`: `base_frame: base_footprint` →
  **`base_link`** (Isaac iw_hub엔 base_footprint 없음)
- `iw_hub_slam/launch/slam.launch.py`: `base_link→lidar_link`
  static_transform_publisher 추가 (없으면 scan을 base_frame으로 변환 못 해 전부 버림)

## slam_toolbox는 Jazzy에서 라이프사이클 노드

`async_slam_toolbox_node`가 unconfigured로 떠서 `/scan` 구독을 안 함
(Subscription count 0). **수동 전환 필요:**
```bash
ros2 lifecycle set /slam_toolbox configure
ros2 lifecycle set /slam_toolbox activate
```
→ TODO: launch에 autostart(lifecycle manager) 박아서 자동화

## 로봇 구동: cmd_vel 타입

Isaac `ROS2SubscribeTwist`는 **`geometry_msgs/Twist`** 구독 (TwistStamped 아님 —
그건 Gazebo Harmonic용). `/cmd_vel`에 Twist로 발행해야 로봇이 움직임.
```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.5}, angular: {z: 0.2}}" --rate 10
```

## 실행 순서 (재현용)

```bash
# 패널1 Isaac
cd /workspace/MARS && source deploy/isaac/env_isaac.sh
python deploy/isaac/isaac_warehouse_ros2.py --warehouse --slam
#   → [1d] lidar writer attached: /scan, timeline playing

# 패널2 SLAM
source deploy/isaac/env_ros2.sh
mkdir -p /root/ros2_ws/src
ln -sfn /workspace/MARS/agents/mars/ros2_ws/src/iw_hub_slam /root/ros2_ws/src/iw_hub_slam
cd /root/ros2_ws && colcon build --packages-select iw_hub_slam && source install/setup.bash
ros2 launch iw_hub_slam slam.launch.py
ros2 lifecycle set /slam_toolbox configure && ros2 lifecycle set /slam_toolbox activate

# 패널3 coverage 모니터
source deploy/isaac/env_ros2.sh && python3 deploy/isaac/map_coverage.py

# 패널4 통로 자동 순회 (끝나면 "reached final waypoint")
source deploy/isaac/env_ros2.sh
python3 deploy/isaac/follow_waypoints.py - \
  0,8 1.9,24 -3,24 -3,9 -8,9 -8,24 -12.9,24 -12.9,9 \
  -17.9,9 -17.9,24 -22.8,24 -22.8,9 0,8

# coverage가 안 커지면 맵 저장
ros2 run nav2_map_server map_saver_cli -f /workspace/warehouse_map
```

## 웨어하우스 좌표 (6.0 USD 추출)

- 선반 열 x: -25.3, -20.4, -15.4, -10.5, -5.5, -0.55, +4.4
- 통로 x(선반 사이): -22.8, -17.9, -12.9, -8.0, -3.0, +1.9
- 통로 y 범위: ~9 ~ 25
- metersPerUnit: 1.0
- 로봇 스폰: world 원점(0,0) — odom 좌표 ≈ world 좌표

## 다음 할 일

1. 통로 순회 완주 → 맵 저장 (`/workspace/warehouse_map.{pgm,yaml}`)
2. 저장한 맵을 리포로 가져와 기존 `warehouse_real` 교체 검토
3. slam.launch.py lifecycle autostart 자동화
4. 로봇 구동 튜닝(WHEEL_RADIUS/BASE) — 현재 cmd_vel 응답 확인됨, 정밀도 점검
5. `/detections`(YOLO) → MARS agent → Nav2 파이프라인 연결 (Isaac 기준 재확인)

"""데모 플레이어 — 학습된 IPPO/MAPPO 정책을 시각화.

훈련 환경(cuboid)과 동일한 물리 구조를 유지하면서
iw_hub 로봇 + 창고 배경으로 시각만 교체.

실행:
  python training/multi_robot/demo_play.py \
    --checkpoint logs/warehouse_ippo/model_400.pt \
    --livestream 1
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="MARS 데모 플레이어")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs",   type=int, default=1)
parser.add_argument("--max_steps",  type=int, default=0, help="0=무한(GUI 데모), N=N step 후 종료(headless 검증)")
parser.add_argument("--diff_drive", action="store_true", default=False,
                    help="diff-drive 체크포인트 재생: vy 차단(커브로만 방향전환). 학습과 동일 동역학")
parser.add_argument("--diff_drive_ctrl", action="store_true", default=False,
                    help="계층 컨트롤러: model_10998 그대로 + diff-drive 변환(재학습 0, 자연스러운 커브)")
parser.add_argument("--turn_gain", type=float, default=3.0,
                    help="컨트롤러 방향오차→omega 게인(클수록 빨리 돌아 향함)")
parser.add_argument("--max_omega", type=float, default=None,
                    help="회전 권한(예: 2.6). None=기본 유지")
parser.add_argument("--max_vy", type=float, default=None,
                    help="옆걸음 상한(데모 동역학을 학습 종료 상태와 정합). "
                         "예: 커리큘럼이 max_vy=0.3에서 끝났으면 --max_vy 0.3 (--diff_drive 빼고). "
                         "--diff_drive(vy=0 강제)와 달리 학습 종료값 정확 재현.")
parser.add_argument("--extended_obs", action="store_true", default=False,
                    help="obs 19D 정책 재생 (학습 시 --extended_obs로 훈련된 모델).")
parser.add_argument("--visual_yaw_align", action="store_true", default=False,
                    help="시각만 diff-drive: vx/vy 유지(strafe 보존) + omega만 진행방향 자동 정렬. "
                         "Amazon Kiva 방식. 회피 100% 유지 + 외형 diff-drive. 재학습 0.")
parser.add_argument("--yaw_align_gain", type=float, default=4.0,
                    help="yaw 정렬 속도 (클수록 빨리 회전해 따라감)")
parser.add_argument("--continuous_goals", dest="continuous_goals", action="store_true", default=True,
                    help="골 도달 즉시 새 골 부여 — 끊임없이 일하는 창고 연출(기본 ON)")
parser.add_argument("--oneshot_goals", dest="continuous_goals", action="store_false",
                    help="연속 골 끄기(한 번 가고 정지 — 학습과 동일 동작)")
parser.add_argument("--task", dest="task_cycle", action="store_true", default=True,
                    help="선반↔스테이션 운반 사이클 연출(픽업→하차 왕복+박스, 기본 ON)")
parser.add_argument("--no_task", dest="task_cycle", action="store_false",
                    help="운반 사이클 끄고 랜덤 골(연속 골)로")
parser.add_argument("--video", action="store_true",
                    help="rgb_array→mp4 헤드리스 녹화 (livestream 불필요)")
parser.add_argument("--video_length", type=int, default=1500,
                    help="녹화할 step 수 (60Hz·decimation=4 기준 ≈25초)")
parser.add_argument("--video_folder", type=str, default="logs/demo_videos")
parser.add_argument("--cam_eye", type=str, default="",
                    help="카메라 위치 x,y,z (예: 5,-5,4). 비디오 모드는 마우스 못 쓰니 여기서 지정")
parser.add_argument("--cam_target", type=str, default="",
                    help="카메라 타겟 x,y,z (예: 0,0,0.3)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
if args.video:
    args.enable_cameras = True   # AppLauncher가 카메라 백엔드 켜야 render(rgb_array) 가능
    if args.max_steps == 0:
        args.max_steps = args.video_length
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import euler_xyz_from_quat, sample_uniform
# quat_apply_inverse는 신규 Isaac Lab(RunPod) 이름. 구버전(Paperspace v2.0.0)은 quat_rotate_inverse(동일 기능)로 폴백
try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_marl_env import (
    WarehouseMARLEnv, WarehouseMARLEnvCfg,
    N_ROBOTS, OBS_PER_ROBOT, SPAWN_OFFSETS, ROBOT_COLLISION_DIST,
    obs_per_robot,
)
from envs.warehouse.warehouse_obstacle_env import SHELF_CENTERS, SHELF_HALF, _shelf_aabb_dist, _goal_in_shelf
from envs.warehouse.ippo_wrapper import IPPOReshapeWrapper


# 검증된 절대 URL 직접 사용 (ISAAC_NUCLEUS_DIR 경로 불확실성 제거, manipulation env와 동일 패턴)
_ISAAC_CLOUD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
IW_HUB_USD = f"{_ISAAC_CLOUD}/Isaac/Robots/Idealworks/iwhub/iw_hub_static.usd"
WAREHOUSE_USD = f"{_ISAAC_CLOUD}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

# ════════════════════════════════════════════════════════════════════════
#  데모 씬 튜닝 노브 — 스크린샷 보며 이 값만 바꾸면 됨 (재훈련/로직 영향 없음)
# ════════════════════════════════════════════════════════════════════════
#  활동 구역: 로봇 스폰 ±1.5m, 선반 (±2, ±2.5). 창고 USD를 이 구역에 맞춰 정렬.
WAREHOUSE_TRANSLATE = (0.0, 0.0, 0.0)   # 창고 USD 위치 (열린 통로가 원점에 오도록 이동)
WAREHOUSE_ROT_DEG   = 0.0               # z축 회전(도) — 창고 통로 방향 맞추기
WAREHOUSE_SCALE     = 1.0               # 창고 전체 스케일
SHOW_BOX_SHELVES    = False             # True면 민짜 충돌박스 외형 표시. False면 외형 숨기고 아래 랙만 보임(충돌은 유지)
USE_SHELF_USD       = True               # True면 실제 Isaac 창고 랙 USD 시도(빈 로드면 자동으로 절차적 폴백)
# 선반 USD 후보 — 위에서부터 시도, 첫 성공(메시 존재)을 채택. 모두 실패시 절차적 폴백.
SHELF_USD_CANDIDATES = [
    f"{_ISAAC_CLOUD}/Isaac/Environments/Simple_Warehouse/Props/SM_RackLongMetal_A1.usd",
    f"{_ISAAC_CLOUD}/Isaac/Environments/Simple_Warehouse/Props/SM_RackLongMetal_B1.usd",
    f"{_ISAAC_CLOUD}/Isaac/Environments/Simple_Warehouse/Props/SM_RackLongMetal_C1.usd",
    f"{_ISAAC_CLOUD}/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_A1.usd",
    f"{_ISAAC_CLOUD}/Isaac/Environments/Simple_Warehouse/Props/SM_RackPile_A4.usd",
]
SHELF_USD = SHELF_USD_CANDIDATES[0]   # 하위호환: 단일 변수도 유지
RACK_HEIGHT         = 3.5               # 랙 높이(m) — 로봇 대비 확실히 크게(선반이 작다는 피드백 반영)
CAMERA_EYE    = (3.88, -9.80, 5.43)     # 창고 외벽 바깥에서 내부 비스듬히 (사용자 viewport 캡처)
CAMERA_TARGET = (1.45, -6.15, 3.03)     # 창고 내부 상단 영역 — 로봇 활동 구역 위쪽
#  로봇 외형(iw.hub) — 큐브 yaw 대신 '진행 방향'으로 향하게 + 수평 yaw만(기울기/덜덜 제거)
ROBOT_VISUAL_SCALE = 1.0                # iw.hub 외형 크기 (선반 대비 안 맞으면 조정)
VIS_YAW_OFFSET_DEG = 0.0                # iw.hub 메시 정면축 보정(도) — 옆을 보면 90/180 등으로
VIS_SMOOTH_YAW     = 0.20               # 회전 보간(0=안돎, 1=즉시) — 코너링 부드럽게
VIS_SMOOTH_POS     = 0.85               # 위치 보간(1=큐브에 딱 붙음) — 낮추면 덜덜 줄지만 지연↑
VIS_MOVE_THRESH    = 0.30               # 평활속도 이 미만이면 헤딩 고정(저속 빙빙/떨림 방지)
VIS_VEL_EMA        = 0.06               # 외형 yaw 산출용 속도 EMA(작을수록 평활, tau≈substep_dt/EMA≈0.28s)
VIS_YAW_RATE_MAX   = 0.04               # 스텝당 외형 회전 상한(rad) — 타깃 급변에도 메시 못 튐
VIS_Z_OFFSET       = -0.15              # 외형 z 보정(m) — 큐브중심(0.15)에 얹혀 떠보이는 것 보정(바퀴 바닥에 붙임)
# ── 유니사이클(차량형) 외형: 큐브를 '쫓아가는' 그림자. strafe를 시각적으로 제거 ──
#    외형은 자기 향한 방향으로만 전진 + 회전율 제한 → 옆걸음/빙빙 없이 진짜 바퀴 로봇처럼 보임
VIS_UNICYCLE       = True               # True면 유니사이클 추종(스케이트 제거). False면 큐브에 딱 붙음
VIS_TURN_GAIN      = 0.25               # 목표방향으로 조향 게인(0~1)
VIS_TURN_MAX       = 0.06               # 스텝당 최대 회전(rad) — 낮을수록 부드럽게 돎(코너 반경↑)
VIS_DRIVE_GAIN     = 1.0                # 거리 대비 전진 비율(1=매 스텝 따라잡기 시도)
VIS_SPEED_MAX      = 0.30               # 스텝당 최대 전진(m) — 목표 리스폰 순간 순간이동 방지
# 골 마커 색(로봇별) — 랜덤골(--no_task) 모드에서만 사용.
GOAL_COLORS        = [(0.10, 0.90, 0.30), (0.95, 0.25, 0.20), (0.25, 0.50, 1.0)]
GOAL_MARKER_RADIUS = 0.15
GOAL_MARKER_Z      = 0.12               # 바닥에 살짝 띄운 구 높이
# ── 운반 사이클(--task) 연출: 선반 통로면 픽업 패드 ↔ 통로 양끝 도크 ──
#    SHELF_CENTERS = (±2, ±2.5). 픽업 패드는 각 선반의 통로(중앙)쪽 면 앞에.
#  주의: 박힘 판정은 큐브가 아니라 '눈에 보이는 iw.hub 외형'이 기준.
#    |y| + (goal_radius+REACH_EPS) + iw.hub 외형 코(~0.55) < 선반면(2.25)이어야 시각 클리핑 없음.
#    0.8 + 0.45 + 0.55 = 1.80 < 2.25 (여유 0.45m). 1.2는 합이 2.20으로 선반면에 닿아 박혀 보이던 값
#    (이전 계산은 큐브 반폭 0.2만 넣고 외형 길이를 빼먹어 시각 클리핑이 남았음).
PICK_POINTS = [(-2.0, 0.8), (2.0, 0.8), (-2.0, -0.8), (2.0, -0.8)]   # env-local
DOCK_POINTS = [(-2.6, 0.0), (2.6, 0.0)]                              # 통로 패킹 도크(벽 안쪽)
STUCK_STEPS = 180                        # 이 스텝(≈12s) 동안 골 못 닿으면 막힘→목적지 재배정
PICK_PAD_COLOR  = (0.15, 0.85, 0.35)    # 픽업존 폴백 색(초록 원반)
DOCK_PAD_COLOR  = (0.20, 0.55, 1.0)     # 하차존 폴백 색(파란 도크 패드)
BOX_COLOR       = (0.60, 0.42, 0.22)    # 운반 박스 폴백 색(갈색)
PAD_Z           = 0.02                   # 바닥 패드 높이
BOX_Z_ON_ROBOT  = 0.45                  # 운반 시 박스가 로봇 위에 얹히는 높이
BOX_Z_HIDDEN    = -10.0                  # 비운반 시 박스 숨김(바닥 아래)
REACH_EPS       = 0.10                   # 웨이포인트 도달 판정 여유(goal_radius에 가산)
# ── 외부 에셋(Isaac Simple_Warehouse Props) 후보 — 첫 메시 성공 채택, 모두 실패 시 도형 폴백 ──
_PROPS = f"{_ISAAC_CLOUD}/Isaac/Environments/Simple_Warehouse/Props"
BOX_USD_CANDIDATES = [f"{_PROPS}/SM_CardBoxA_{n}.usd" for n in (1, 2, 3, 4)] + \
                     [f"{_PROPS}/SM_CardBoxB_1.usd", f"{_PROPS}/SM_CardBoxC_1.usd", f"{_PROPS}/SM_CardBoxD_1.usd"]
PALLET_USD_CANDIDATES = [f"{_PROPS}/SM_Pallet.usd", f"{_PROPS}/SM_PaletteA_01.usd",
                         f"{_PROPS}/SM_CratePlastic_A_1.usd", f"{_PROPS}/SM_CratePlastic_A_2.usd"]
DOCK_USD_CANDIDATES = [f"{_PROPS}/SM_CratePlastic_B_1.usd", f"{_PROPS}/SM_CratePlastic_C_1.usd",
                       f"{_PROPS}/SM_CratePlastic_A_1.usd"]
BOX_USD_SCALE    = (1.0, 1.0, 1.0)      # 골판지 박스 스케일(에셋 native 크기 보고 조정)
PALLET_USD_SCALE = (1.0, 1.0, 1.0)      # 픽업 팔레트/크레이트 스케일
DOCK_USD_SCALE   = (1.0, 1.0, 1.0)      # 하차 크레이트 스케일
# ── 선반 적재물(꾸미기): 비어 보이는 랙에 박스를 올림(순수 비주얼, 충돌 없음) ──
SHELF_GOODS         = True               # False면 빈 선반
SHELF_GOODS_LEVELS  = [0.35, 1.25]       # 박스 얹는 높이(m) — 랙 보드에 맞춰 조정
SHELF_GOODS_PER_ROW = 3                  # 한 층에 박스 개수(선반 x축 분포)
SHELF_GOODS_XSPAN   = 2.0                # 박스 분포 x폭(선반 길이 3.0 안쪽)
SHELF_GOODS_SCALE   = (1.0, 1.0, 1.0)    # 적재 박스 스케일
# ════════════════════════════════════════════════════════════════════════


class WarehouseDemoEnvCfg(WarehouseMARLEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=14.0, replicate_physics=True
    )
    # 연속 골 데모: 에피소드 타임아웃 리셋(=스폰 텔레포트)을 길게 늘려 영상 내내 매끄럽게.
    # 충돌/이탈 리셋은 그대로 유지(회피 실패 시 자연 리셋).
    episode_length_s = 200.0


class WarehouseDemoEnv(WarehouseMARLEnv):
    """시각화 전용 환경 — 물리 큐브 유지, iw.hub 외형 visual-only."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        from isaacsim.core.prims import XFormPrim
        self._robot_visuals = [
            XFormPrim(f"/World/envs/env_.*/RobotVisual_{i}") for i in range(N_ROBOTS)
        ]
        # 외형 yaw 스무딩 버퍼만 (위치는 큐브에 직붙임 — 위치 LERP는 lag artifact 만듦)
        self._vis_yaw = [torch.zeros(self.num_envs, device=self.device) for _ in self.robots]
        # 외형 yaw 산출용 속도 EMA — 순간 속도방향 노이즈(저속/옆걸음)로 인한 좌우 떨림 차단
        self._vis_vel = [torch.zeros(self.num_envs, 2, device=self.device) for _ in self.robots]
        self._yaw_off = math.radians(VIS_YAW_OFFSET_DEG)
        # action 이동평균 버퍼 — 정책 출력 노이즈로 인한 cube velocity 매-step 점프 차단.
        # 이게 진짜 떨림 원인. 시각 LERP는 증상 가림용이고 이게 root cause fix.
        self._prev_actions = None

        self._task_cycle = bool(getattr(self.cfg, "task_cycle", False))
        if self._task_cycle:
            self._setup_task_visuals()
            self._setup_task_state()
        else:
            # 랜덤 골 모드: 로봇별 목적지 색 구
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/goal_markers",
                markers={
                    f"g{i}": sim_utils.SphereCfg(
                        radius=GOAL_MARKER_RADIUS,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=GOAL_COLORS[i % len(GOAL_COLORS)],
                            emissive_color=GOAL_COLORS[i % len(GOAL_COLORS)],
                        ),
                    ) for i in range(N_ROBOTS)
                },
            )
            self._goal_markers = VisualizationMarkers(marker_cfg)

    # ── 운반 사이클 연출 ───────────────────────────────────────────────
    @staticmethod
    def _spawn_usd_or_fallback(prim_path, candidates, fallback_cfg, translation,
                               orientation=(1.0, 0.0, 0.0, 0.0), scale=None):
        """USD 후보를 순서대로 시도(메시 존재 확인), 모두 실패 시 fallback_cfg 스폰.
        채택된 USD 경로(or None) 반환. 선반 스폰과 동일한 검증 패턴."""
        import omni.usd
        from pxr import UsdGeom, Usd
        st = omni.usd.get_context().get_stage()
        for cand in candidates:
            ex = st.GetPrimAtPath(prim_path)
            if ex.IsValid():
                st.RemovePrim(ex.GetPath())
            try:
                cfg = sim_utils.UsdFileCfg(usd_path=cand, scale=scale) if scale \
                    else sim_utils.UsdFileCfg(usd_path=cand)
                cfg.func(prim_path, cfg, translation=translation, orientation=orientation)
                pr = st.GetPrimAtPath(prim_path)
                if pr.IsValid() and any(d.IsA(UsdGeom.Mesh) for d in Usd.PrimRange(pr)):
                    # 순수 비주얼화: 충돌/리지드바디 비활성(로봇 막힘·박스 낙하 방지)
                    try:
                        from pxr import UsdPhysics
                        for d in Usd.PrimRange(pr):
                            if d.HasAPI(UsdPhysics.CollisionAPI):
                                UsdPhysics.CollisionAPI(d).GetCollisionEnabledAttr().Set(False)
                            if d.HasAPI(UsdPhysics.RigidBodyAPI):
                                UsdPhysics.RigidBodyAPI(d).GetRigidBodyEnabledAttr().Set(False)
                    except Exception:
                        pass
                    return cand
            except Exception:
                pass
        ex = st.GetPrimAtPath(prim_path)
        if ex.IsValid():
            st.RemovePrim(ex.GetPath())
        fallback_cfg.func(prim_path, fallback_cfg, translation=translation, orientation=orientation)
        return None

    def _spawn_task_props(self):
        """_setup_scene에서 호출: 픽업 팔레트/하차 크레이트(정적) + 운반 박스(로봇당, env_0).
        clone_environments 전에 env_0에 두면 env_.*로 복제됨."""
        pick_fb = sim_utils.CylinderCfg(
            radius=0.45, height=0.06,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=PICK_PAD_COLOR, emissive_color=PICK_PAD_COLOR))
        dock_fb = sim_utils.CuboidCfg(
            size=(0.95, 0.95, 0.06),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=DOCK_PAD_COLOR, emissive_color=DOCK_PAD_COLOR))
        box_fb = sim_utils.CuboidCfg(
            size=(0.34, 0.34, 0.30),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BOX_COLOR))
        for p_i, (px, py) in enumerate(PICK_POINTS):
            got = self._spawn_usd_or_fallback(f"/World/envs/env_0/PickPad_{p_i}",
                                              PALLET_USD_CANDIDATES, pick_fb,
                                              translation=(px, py, 0.0), scale=PALLET_USD_SCALE)
            if p_i == 0:
                print(f"[Demo] 픽업 프롭: {got.rsplit('/',1)[-1] if got else '폴백(초록 원반)'}")
        for d_i, (dx, dy) in enumerate(DOCK_POINTS):
            got = self._spawn_usd_or_fallback(f"/World/envs/env_0/DockPad_{d_i}",
                                              DOCK_USD_CANDIDATES, dock_fb,
                                              translation=(dx, dy, 0.0), scale=DOCK_USD_SCALE)
            if d_i == 0:
                print(f"[Demo] 하차 프롭: {got.rsplit('/',1)[-1] if got else '폴백(파란 패드)'}")
        for i in range(N_ROBOTS):
            got = self._spawn_usd_or_fallback(f"/World/envs/env_0/CarryBox_{i}",
                                              BOX_USD_CANDIDATES, box_fb,
                                              translation=(0.0, 0.0, BOX_Z_HIDDEN), scale=BOX_USD_SCALE)
            if i == 0:
                print(f"[Demo] 운반 박스: {got.rsplit('/',1)[-1] if got else '폴백(갈색 큐브)'}")

    def _spawn_shelf_goods(self):
        """선반(랙) 위에 박스 적재물을 올려 '재고 있는 창고'처럼. 순수 비주얼."""
        box_fb = sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=BOX_COLOR))
        n = max(SHELF_GOODS_PER_ROW, 1)
        printed = False
        for s_i, (cx, cy, _cz) in enumerate(SHELF_CENTERS):
            for l_i, lz in enumerate(SHELF_GOODS_LEVELS):
                for k in range(n):
                    gx = cx + (k - (n - 1) / 2.0) * (SHELF_GOODS_XSPAN / n)
                    # 박스 변형 다양화: 후보 시작점을 셀마다 회전
                    off = (s_i + l_i + k) % len(BOX_USD_CANDIDATES)
                    cands = BOX_USD_CANDIDATES[off:] + BOX_USD_CANDIDATES[:off]
                    got = self._spawn_usd_or_fallback(
                        f"/World/envs/env_0/ShelfGoods_{s_i}_{l_i}_{k}",
                        cands, box_fb, translation=(gx, cy, lz), scale=SHELF_GOODS_SCALE)
                    if not printed:
                        print(f"[Demo] 선반 적재물: {got.rsplit('/',1)[-1] if got else '폴백(큐브)'}")
                        printed = True

    def _setup_task_visuals(self):
        """운반 사이클 좌표 텐서 + 운반 박스 XFormPrim 핸들(매 스텝 로봇 위로 이동)."""
        from isaacsim.core.prims import XFormPrim
        device = self.device
        self._pick_w = torch.tensor(PICK_POINTS, dtype=torch.float32, device=device)  # (P,2)
        self._dock_w = torch.tensor(DOCK_POINTS, dtype=torch.float32, device=device)  # (D,2)
        self._carry_box_prims = [
            XFormPrim(f"/World/envs/env_.*/CarryBox_{i}") for i in range(N_ROBOTS)
        ]

    def _setup_task_state(self):
        E, device = self.num_envs, self.device
        # state: 0=픽업하러 감, 1=하차하러 감
        self._task_state = torch.zeros(E, N_ROBOTS, dtype=torch.long, device=device)
        self._carrying   = torch.zeros(E, N_ROBOTS, dtype=torch.bool, device=device)
        self._task_shelf = torch.zeros(E, N_ROBOTS, dtype=torch.long, device=device)
        self._task_dock  = torch.zeros(E, N_ROBOTS, dtype=torch.long, device=device)
        self._task_age   = torch.zeros(E, N_ROBOTS, dtype=torch.long, device=device)  # 현재 골 추적 경과(막힘 감지)
        # 전체 env 초기 배정(서로 다른 선반으로 분산)
        all_ids = torch.arange(E, device=device)
        self._init_task(all_ids)

    def _init_task(self, env_ids: torch.Tensor):
        """리셋된 env들: 각 로봇에 시작 선반 배정 + 골=픽업 패드, 비운반 상태로."""
        n = env_ids.shape[0]
        origins = self.scene.env_origins[env_ids, :2]
        n_pick = len(PICK_POINTS)
        for i in range(N_ROBOTS):
            shelf = torch.randint(0, n_pick, (n,), device=self.device)
            self._task_shelf[env_ids, i] = shelf
            self._task_state[env_ids, i] = 0
            self._carrying[env_ids, i]   = False
            self._task_age[env_ids, i]   = 0
            self._goal_pos_w[env_ids, i] = self._pick_w[shelf] + origins

    def _advance_task(self):
        """도달 시 픽업↔하차 전이(박스 토글). 막힘(STUCK_STEPS 미도달) 시 목적지 재배정."""
        eps = self.cfg.goal_radius + REACH_EPS
        origins = self.scene.env_origins[:, :2]
        n_pick, n_dock = len(PICK_POINTS), len(DOCK_POINTS)
        self._task_age += 1
        for i in range(N_ROBOTS):
            pos = self.robots[i].data.root_pos_w[:, :2]
            reached = (pos - self._goal_pos_w[:, i]).norm(dim=1) < eps
            stuck   = self._task_age[:, i] > STUCK_STEPS
            for e in (reached | stuck).nonzero(as_tuple=True)[0].tolist():
                self._task_age[e, i] = 0
                if not bool(reached[e]):
                    # 막힘(미도달 타임아웃) → 반드시 '통로 중앙 도크'로 빼서 열린 공간으로 탈출.
                    #   선반 코앞 픽업점으로 재배정하면(이전 동작) 방향 모르는 17D 정책이 또 선반에
                    #   박혀 영원히 못 빠져나옴. 도크(y=0 통로)는 항상 열려 있어 확실한 탈출구.
                    #   state=1('하차하러 가는 중')로 두면 도크 도달 시 정상 사이클로 자연 복귀.
                    dock = int(torch.randint(0, n_dock, (1,)).item())
                    self._task_dock[e, i] = dock
                    self._task_state[e, i] = 1
                    self._goal_pos_w[e, i] = self._dock_w[dock] + origins[e]
                elif self._task_state[e, i].item() == 0:      # 픽업 도달 → 박스 들고 하차로
                    self._carrying[e, i] = True
                    dock = int(torch.randint(0, n_dock, (1,)).item())
                    self._task_dock[e, i] = dock
                    self._task_state[e, i] = 1
                    self._goal_pos_w[e, i] = self._dock_w[dock] + origins[e]
                else:                                          # 하차 도달 → 박스 내리고 새 선반으로
                    self._carrying[e, i] = False
                    shelf = int(torch.randint(0, n_pick, (1,)).item())
                    self._task_shelf[e, i] = shelf
                    self._task_state[e, i] = 0
                    self._goal_pos_w[e, i] = self._pick_w[shelf] + origins[e]

    def _update_carry_boxes(self):
        """운반 박스 prim을 로봇 위로(운반 중) 또는 바닥 아래로(숨김) 이동.
        스테이션(픽업/하차)은 정적 prim이라 갱신 불필요."""
        for i in range(N_ROBOTS):
            p = self.robots[i].data.root_pos_w                       # (E,3) world
            carrying = self._carrying[:, i]
            z = torch.where(carrying,
                            p[:, 2] + BOX_Z_ON_ROBOT,
                            torch.full_like(p[:, 2], BOX_Z_HIDDEN))
            out = torch.stack([p[:, 0], p[:, 1], z], dim=1)
            self._carry_box_prims[i].set_world_poses(positions=out)

    def _get_dones(self):
        terminated, time_out = super()._get_dones()
        if self._task_cycle:
            self._advance_task()
        return terminated, time_out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # super().__init__ 중 첫 리셋에선 task 버퍼가 아직 없음 → 가드
        if getattr(self, "_task_cycle", False) and hasattr(self, "_task_state"):
            if env_ids is None:
                ids = torch.arange(self.num_envs, device=self.device)
            elif isinstance(env_ids, torch.Tensor):
                ids = env_ids.long()
            else:
                ids = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
            self._init_task(ids)

    def _apply_action(self):
        # ── ROOT CAUSE FIX: action 이동평균 (3-step) ──
        # PPO는 매 0.067s마다 새 action 출력 → cube velocity 매 step 점프 → damping이 누름 → 다음 점프...
        # = cube physics 자체가 떨림. 시각은 그걸 그대로 보여줘서 "발광"하는 듯 보임.
        # 해법: action을 3-step 이동평균으로 부드럽게 → velocity 점프 줄어듬 → 물리 떨림 자체 ↓
        if self._prev_actions is None:
            self._prev_actions = [self._actions.clone(), self._actions.clone()]
        smoothed = (self._actions + self._prev_actions[0] + self._prev_actions[1]) / 3.0
        self._prev_actions[1] = self._prev_actions[0].clone()
        self._prev_actions[0] = self._actions.clone()
        # 원본 보존하고 평균값으로 physics step
        orig_actions = self._actions
        self._actions = smoothed
        super()._apply_action()
        self._actions = orig_actions   # 원래 action 복원 (다음 step 정상 처리)

        # 시각 메시: 위치는 큐브 그대로(LERP 빼서 lag artifact 제거), yaw만 스무딩
        SMOOTH_YAW = 0.15
        for i, robot in enumerate(self.robots):
            pos = robot.data.root_pos_w

            if self.cfg.visual_yaw_align:
                # 떨림 root cause: 외형 yaw를 '순간' 속도방향(atan2)에 붙이면 저속/옆걸음 시
                # 방향이 노이즈로 좌우 폭주 → 메시가 격렬히 떪. 해법: ① 속도를 시간 EMA로
                # 평활해 그 방향으로만, ② 평활속도 낮으면 헤딩 고정, ③ 회전율 하드 캡.
                v_now = robot.data.root_lin_vel_w[:, :2]
                self._vis_vel[i] = (1 - VIS_VEL_EMA) * self._vis_vel[i] + VIS_VEL_EMA * v_now
                v_s = self._vis_vel[i]
                moving = v_s.norm(dim=1) > VIS_MOVE_THRESH
                target_yaw = torch.atan2(v_s[:, 1], v_s[:, 0]) + self._yaw_off
            else:
                quat = robot.data.root_quat_w
                _, _, yaw = euler_xyz_from_quat(quat)
                target_yaw = yaw + self._yaw_off
                moving = torch.ones_like(target_yaw, dtype=torch.bool)

            cur = self._vis_yaw[i]
            err = torch.atan2(torch.sin(target_yaw - cur), torch.cos(target_yaw - cur))
            # 회전율 하드 캡 — 타깃이 튀어도 메시는 서서히만 돌아 절대 안 튐
            step_yaw = torch.clamp(err * SMOOTH_YAW, -VIS_YAW_RATE_MAX, VIS_YAW_RATE_MAX)
            new_yaw = cur + torch.where(moving, step_yaw, torch.zeros_like(err))
            self._vis_yaw[i] = new_yaw

            h = 0.5 * new_yaw
            flat_quat = torch.stack([torch.cos(h), torch.zeros_like(h),
                                     torch.zeros_like(h), torch.sin(h)], dim=1)
            out_pos = pos.clone()
            out_pos[:, 2] = out_pos[:, 2] + VIS_Z_OFFSET
            self._robot_visuals[i].set_world_poses(positions=out_pos, orientations=flat_quat)

        # 마커/프롭 갱신
        if self._task_cycle:
            self._update_carry_boxes()
        else:
            # 랜덤 골: 로봇별 목적지 색 구
            g = self._goal_pos_w.reshape(-1, 2)
            z = torch.full((g.shape[0], 1), GOAL_MARKER_Z, device=self.device)
            trans = torch.cat([g, z], dim=1)
            midx = torch.arange(N_ROBOTS, device=self.device).repeat(self.num_envs)
            self._goal_markers.visualize(translations=trans, marker_indices=midx)

    def _setup_scene(self):
        # ── 로봇: 물리는 큐브(model_9999 동역학 100% 보존), 외형만 iw.hub ──
        #    iw_hub_static은 RigidBodyAPI 없는 visual 메시 → 큐브 위에 얹고
        #    _apply_action에서 매 스텝 큐브 pose로 동기화
        self.robots: list[RigidObject] = []
        for i in range(N_ROBOTS):
            robot_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Robot_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.4, 0.3),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False, linear_damping=2.0, angular_damping=5.0,
                        max_linear_velocity=5.0, max_angular_velocity=10.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(SPAWN_OFFSETS[i][0], SPAWN_OFFSETS[i][1], 0.15)
                ),
            )
            self.robots.append(RigidObject(robot_cfg))

        # iw.hub 외형 (visual-only static 메시 — env_0에 두면 clone이 env_.* 복제)
        _s = ROBOT_VISUAL_SCALE
        iw_visual = UsdFileCfg(usd_path=IW_HUB_USD, scale=(_s, _s, _s))
        for i in range(N_ROBOTS):
            iw_visual.func(
                f"/World/envs/env_0/RobotVisual_{i}", iw_visual,
                translation=(SPAWN_OFFSETS[i][0], SPAWN_OFFSETS[i][1], 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )
        print("[Demo] iw.hub visual-only 로드 (물리: 큐브 유지)")

        # ── 바닥: 창고 USD (정렬 노브 적용), 실패 시 GroundPlane ───────
        import math as _math
        _yaw = _math.radians(WAREHOUSE_ROT_DEG)
        _wq = (_math.cos(_yaw / 2), 0.0, 0.0, _math.sin(_yaw / 2))   # z축 yaw 쿼터니언
        try:
            warehouse_cfg = sim_utils.UsdFileCfg(
                usd_path=WAREHOUSE_USD,
                scale=(WAREHOUSE_SCALE, WAREHOUSE_SCALE, WAREHOUSE_SCALE),
            )
            warehouse_cfg.func("/World/Warehouse", warehouse_cfg,
                               translation=WAREHOUSE_TRANSLATE,
                               orientation=_wq)
            print(f"[Demo] 창고 USD 로드 성공 (translate={WAREHOUSE_TRANSLATE}, rot={WAREHOUSE_ROT_DEG}°, scale={WAREHOUSE_SCALE})")
        except Exception:
            spawn_ground_plane("/World/ground", GroundPlaneCfg())
            print("[Demo] 창고 USD 없음, GroundPlane fallback")

        # ── 선반: cuboid (물리 충돌 항상 유지 — 로봇이 피하는 실제 장애물) ──
        #    외형은 산업용 금속 랙 느낌. 정렬 끝나면 SHOW_BOX_SHELVES=False로
        #    외형만 숨겨 USD 창고 선반만 보이게(충돌은 그대로) 할 수 있음.
        shelf_cfg_base = sim_utils.CuboidCfg(
            size=(3.0, 0.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.30, 0.34, 0.40), metallic=0.7, roughness=0.35
            ),
        )
        for s_i, (cx, cy, cz) in enumerate(SHELF_CENTERS):
            shelf_cfg_base.func(
                f"/World/envs/env_0/Shelf_{s_i}",
                shelf_cfg_base,
                translation=(cx, cy, cz),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )
        # 로봇 큐브 외형 숨김 — 충돌 큐브가 iw.hub 위로 비어져 "물건 얹은 듯" 보이던 것 제거.
        # 충돌은 유지(MakeInvisible은 visual만 꺼짐). clone 전이라 env_0에 적용→복제본도 숨김.
        try:
            import omni.usd
            from pxr import UsdGeom
            _stage = omni.usd.get_context().get_stage()
            for r_i in range(N_ROBOTS):
                _rp = _stage.GetPrimAtPath(f"/World/envs/env_0/Robot_{r_i}")
                if _rp.IsValid():
                    UsdGeom.Imageable(_rp).MakeInvisible()
            print("[Demo] 로봇 큐브 외형 숨김 (충돌·물리 유지, iw.hub만 보임)")
        except Exception as _e:
            print(f"[Demo] 로봇 큐브 숨김 실패(무시): {_e}")

        if not SHOW_BOX_SHELVES:
            # 충돌은 유지, 외형만 숨김 (clone 전이라 env_0에 적용→복제본도 숨김)
            try:
                import omni.usd
                from pxr import UsdGeom
                _stage = omni.usd.get_context().get_stage()
                for s_i in range(len(SHELF_CENTERS)):
                    _p = _stage.GetPrimAtPath(f"/World/envs/env_0/Shelf_{s_i}")
                    if _p.IsValid():
                        UsdGeom.Imageable(_p).MakeInvisible()
                print("[Demo] 박스 선반 외형 숨김 (충돌만 유지)")
            except Exception as _e:
                print(f"[Demo] 선반 숨김 실패(무시): {_e}")

            # ── 랙 외형: 실제 Isaac 창고 랙 USD 후보 체인 우선, 모두 실패시 절차적 톨 랙 ──
            #    footprint 3.0(x) × 0.5(y)는 충돌 박스와 일치(로봇이 피하는 위치).
            #    후보를 순서대로 시도, 첫 메시 존재 USD를 채택.
            use_usd_ok = False
            chosen_usd = None
            if USE_SHELF_USD:
                import omni.usd
                from pxr import UsdGeom, Usd
                _st = omni.usd.get_context().get_stage()
                for _cand in SHELF_USD_CANDIDATES:
                    # 시도 전 기존 Rack_* 정리(이전 후보가 빈 prim 남겼을 수 있음)
                    for s_i in range(len(SHELF_CENTERS)):
                        _existing = _st.GetPrimAtPath(f"/World/envs/env_0/Rack_{s_i}")
                        if _existing.IsValid():
                            _st.RemovePrim(_existing.GetPath())
                    try:
                        rack_usd = sim_utils.UsdFileCfg(usd_path=_cand)
                        ok = True
                        for s_i, (cx, cy, cz) in enumerate(SHELF_CENTERS):
                            _rp = f"/World/envs/env_0/Rack_{s_i}"
                            rack_usd.func(_rp, rack_usd, translation=(cx, cy, 0.0),
                                          orientation=(1.0, 0.0, 0.0, 0.0))
                            _pr = _st.GetPrimAtPath(_rp)
                            if not (_pr.IsValid() and any(d.IsA(UsdGeom.Mesh) for d in Usd.PrimRange(_pr))):
                                ok = False
                                break
                        if ok:
                            use_usd_ok = True
                            chosen_usd = _cand
                            print(f"[Demo] 창고 랙 USD 채택: {_cand.rsplit('/', 1)[-1]}")
                            break
                        else:
                            print(f"[Demo] 랙 USD 비어있음, 다음 후보 시도: {_cand.rsplit('/', 1)[-1]}")
                    except Exception as _e:
                        print(f"[Demo] 랙 USD 예외({_cand.rsplit('/', 1)[-1]}), 다음 후보: {_e}")
                if not use_usd_ok:
                    # 모든 USD 실패 — 남은 빈 prim 정리 후 절차적 폴백으로
                    for s_i in range(len(SHELF_CENTERS)):
                        _existing = _st.GetPrimAtPath(f"/World/envs/env_0/Rack_{s_i}")
                        if _existing.IsValid():
                            _st.RemovePrim(_existing.GetPath())
                    print("[Demo] 모든 USD 후보 실패 → 절차적 폴백")

            if not use_usd_ok:
                # 절차적 톨 랙: 기둥4 + 가로빔 + 선반판 다단 (height=RACK_HEIGHT)
                metal = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.26, 0.34), metallic=0.85, roughness=0.3)
                wood  = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.45, 0.24), metallic=0.0, roughness=0.75)
                H = RACK_HEIGHT
                post  = sim_utils.CuboidCfg(size=(0.16, 0.16, H), visual_material=metal)   # 두꺼운 기둥
                plank = sim_utils.CuboidCfg(size=(2.95, 0.5, 0.10), visual_material=wood)   # 두꺼운 선반판
                endbar = sim_utils.CuboidCfg(size=(0.10, 0.5, 0.10), visual_material=metal) # 양끝 가로빔
                levels = [0.08, H * 0.30, H * 0.55, H * 0.80, H - 0.1]   # 5단
                for s_i, (cx, cy, cz) in enumerate(SHELF_CENTERS):
                    base = f"/World/envs/env_0/Rack_{s_i}"
                    for p_i, (dx, dy) in enumerate([(-1.45, -0.2), (1.45, -0.2), (-1.45, 0.2), (1.45, 0.2)]):
                        post.func(f"{base}/post_{p_i}", post, translation=(cx + dx, cy + dy, H / 2))
                    for l_i, lz in enumerate(levels):
                        plank.func(f"{base}/plank_{l_i}", plank, translation=(cx, cy, lz))
                    for e_i, ex in enumerate((-1.45, 1.45)):       # 양끝 상단 가로빔
                        endbar.func(f"{base}/end_{e_i}", endbar, translation=(cx + ex, cy, H - 0.1))
                print(f"[Demo] 절차적 톨 랙 생성 (height={H}m, 5단 선반)")

        # ── 선반 적재물: 빈 랙에 박스를 올려 재고 있는 창고처럼(순수 비주얼) ──
        if SHELF_GOODS:
            self._spawn_shelf_goods()

        # ── 운반 사이클 프롭: 픽업 팔레트/하차 크레이트(정적) + 운반 박스(로봇당) ──
        #    clone 전 env_0에 스폰 → env_.*로 복제. cfg로 게이트(랜덤골 모드면 생략).
        if getattr(self.cfg, "task_cycle", False):
            self._spawn_task_props()

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        for i, robot in enumerate(self.robots):
            self.scene.rigid_objects[f"robot_{i}"] = robot

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 0.98, 0.95))
        light_cfg.func("/World/Light", light_cfg)


def _parse_xyz(s: str, default):
    if not s:
        return default
    return tuple(float(x.strip()) for x in s.split(","))


def main():
    env_cfg = WarehouseDemoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.disable_strafe = args.diff_drive   # diff-drive 모델은 학습과 동일하게 vy 차단
    env_cfg.diff_drive_controller = args.diff_drive_ctrl   # 계층 컨트롤러(재학습 0)
    env_cfg.diff_drive_turn_gain  = args.turn_gain
    env_cfg.visual_yaw_align      = args.visual_yaw_align   # 시각만 diff-drive (회피 100% 보존)
    env_cfg.yaw_align_gain        = args.yaw_align_gain
    env_cfg.task_cycle            = args.task_cycle          # 선반↔스테이션 운반 사이클
    # 운반 사이클이 골을 직접 구동하므로 랜덤 연속골은 끔(이중 구동 방지)
    env_cfg.continuous_goals      = args.continuous_goals and not args.task_cycle
    if args.max_omega is not None:
        env_cfg.max_omega = args.max_omega
    # 커리큘럼 모델 데모: 학습 종료 max_vy(예: 0.3)와 동일 동역학으로 재생
    if args.max_vy is not None:
        env_cfg.max_vy = args.max_vy
        if args.diff_drive:
            print(f"[Demo] 경고: --diff_drive(vy=0)와 --max_vy {args.max_vy} 동시 — disable_strafe가 우선해 vy=0 됨. --diff_drive 빼세요.")
        else:
            print(f"[Demo] max_vy 오버라이드: {args.max_vy} (학습 종료 상태와 정합)")
    # 확장 obs 정책(19D) 재생
    obr_demo = OBS_PER_ROBOT
    if args.extended_obs:
        env_cfg.extended_obstacle_obs = True
        obr_demo = obs_per_robot(False, True)
        env_cfg.observation_space = obr_demo * N_ROBOTS
        env_cfg.state_space       = obr_demo * N_ROBOTS
        print(f"[Demo] 확장 obs 활성화 ({obr_demo}D per robot)")

    env = WarehouseDemoEnv(env_cfg, render_mode="rgb_array" if args.video else None)

    if args.video:
        import gymnasium as gym
        import os
        os.makedirs(args.video_folder, exist_ok=True)
        print(f"[Demo] 녹화 ON: {args.video_folder}/demo-step-0.mp4 ({args.video_length} steps)")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.video_folder,
            step_trigger=lambda s: s == 0,
            video_length=args.video_length,
            disable_logger=True,
            name_prefix="demo",
        )

    env = RslRlVecEnvWrapper(env)
    env = IPPOReshapeWrapper(env, N_ROBOTS, obr_demo)

    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.num_steps_per_env = 24
    runner_cfg.max_iterations = 999999
    runner_cfg.experiment_name = "warehouse_demo"
    runner_cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.01,  # 데모: 노이즈 최소화 (결정론적 행동)
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    cfg_dict = runner_cfg.to_dict()
    cfg_dict["algorithm"]["class_name"] = "PPO"
    cfg_dict["algorithm"]["entropy_coef"] = 0.0
    # rsl_rl 3.x는 obs_groups 필수. isaaclab_rl 신버전(RunPod)은 to_dict()에 포함하지만
    # 구버전(Paperspace v2.0.0)은 누락 → setdefault로 양쪽 안전하게 주입.
    cfg_dict.setdefault("obs_groups", {"policy": ["policy"], "critic": ["critic"]})

    runner = OnPolicyRunner(env, cfg_dict, log_dir="/tmp/demo", device=env.device)
    runner.load(args.checkpoint)

    print(f"\n[Demo] 체크포인트 로드: {args.checkpoint}")
    print(f"[Demo] 로봇 {N_ROBOTS}대 — inference 루프 (학습 안 함, 결정론적)\n")

    # 카메라 고정 — 활동 구역을 비스듬히 프레이밍 (넓어 보이는 문제 해결)
    #   비디오 모드에선 마우스 못 쓰니 --cam_eye/--cam_target로 CLI 지정
    _eye    = _parse_xyz(args.cam_eye,    CAMERA_EYE)
    _target = _parse_xyz(args.cam_target, CAMERA_TARGET)
    try:
        env.unwrapped.sim.set_camera_view(eye=_eye, target=_target)
        print(f"[Demo] 카메라 고정: eye={_eye} target={_target}")
    except Exception as _e:
        print(f"[Demo] 카메라 설정 실패(무시): {_e}")

    # 데모: 정책 고정 inference (act_inference = actor mean, std 미사용 → std 음수 에러 회피)
    policy = runner.alg.policy
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy.act_inference(obs)
        obs, _, _, _ = env.step(actions)
        if isinstance(obs, tuple):
            obs = obs[0]
        step += 1
        if step % 50 == 0:
            print(f"[Demo] step {step} 진행 중...")
        if args.max_steps > 0 and step >= args.max_steps:
            print(f"[Demo] {args.max_steps} step 완료 — 종료")
            break
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

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
VIS_MOVE_THRESH    = 0.15               # 이 속도 미만이면 직전 방향 유지(정지 시 빙빙 방지)
VIS_Z_OFFSET       = -0.15              # 외형 z 보정(m) — 큐브중심(0.15)에 얹혀 떠보이는 것 보정(바퀴 바닥에 붙임)
# ── 유니사이클(차량형) 외형: 큐브를 '쫓아가는' 그림자. strafe를 시각적으로 제거 ──
#    외형은 자기 향한 방향으로만 전진 + 회전율 제한 → 옆걸음/빙빙 없이 진짜 바퀴 로봇처럼 보임
VIS_UNICYCLE       = True               # True면 유니사이클 추종(스케이트 제거). False면 큐브에 딱 붙음
VIS_TURN_GAIN      = 0.25               # 목표방향으로 조향 게인(0~1)
VIS_TURN_MAX       = 0.06               # 스텝당 최대 회전(rad) — 낮을수록 부드럽게 돎(코너 반경↑)
VIS_DRIVE_GAIN     = 1.0                # 거리 대비 전진 비율(1=매 스텝 따라잡기 시도)
VIS_SPEED_MAX      = 0.30               # 스텝당 최대 전진(m) — 목표 리스폰 순간 순간이동 방지
# ════════════════════════════════════════════════════════════════════════


class WarehouseDemoEnvCfg(WarehouseMARLEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=14.0, replicate_physics=True
    )


class WarehouseDemoEnv(WarehouseMARLEnv):
    """시각화 전용 환경 — 물리 큐브 유지, iw.hub 외형 visual-only."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        from isaacsim.core.prims import XFormPrim
        self._robot_visuals = [
            XFormPrim(f"/World/envs/env_.*/RobotVisual_{i}") for i in range(N_ROBOTS)
        ]
        # 외형 스무딩 버퍼 (위치/방향) — 큐브 진동·홀로노믹 회전을 시각적으로 정리
        self._vis_pos = [r.data.root_pos_w.clone() for r in self.robots]
        self._vis_yaw = [torch.zeros(self.num_envs, device=self.device) for _ in self.robots]
        self._yaw_off = math.radians(VIS_YAW_OFFSET_DEG)

    def _apply_action(self):
        super()._apply_action()
        # iw.hub 외형: 물리 큐브에 그대로 붙임 + 큐브의 진짜 yaw(policy가 omega로 만든 방향).
        #   쫓아가기/스무딩/유니사이클 전부 제거 — '큐브가 하는 그대로'만 정직하게 표시(발산 없음).
        for i, robot in enumerate(self.robots):
            pos = robot.data.root_pos_w                  # (E,3) 큐브 위치
            quat = robot.data.root_quat_w                # (E,4) 큐브 실제 방향
            _, _, yaw = euler_xyz_from_quat(quat)        # 수평 yaw만 추출(기울기 제거)
            yaw = yaw + self._yaw_off
            h = 0.5 * yaw
            flat_quat = torch.stack([torch.cos(h), torch.zeros_like(h),
                                     torch.zeros_like(h), torch.sin(h)], dim=1)  # wxyz
            out_pos = pos.clone()
            out_pos[:, 2] = out_pos[:, 2] + VIS_Z_OFFSET   # 바퀴를 바닥에 붙임
            self._robot_visuals[i].set_world_poses(positions=out_pos, orientations=flat_quat)

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

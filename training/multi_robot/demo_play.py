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
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="MARS 데모 플레이어")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs",   type=int, default=1)
parser.add_argument("--max_steps",  type=int, default=0, help="0=무한(GUI 데모), N=N step 후 종료(headless 검증)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
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
SHOW_BOX_SHELVES    = False             # True면 민짜 충돌박스 외형 표시. False면 외형 숨기고 아래 '오픈 랙'만 보임(충돌은 유지)
CAMERA_EYE    = (9.0, -9.0, 7.0)        # 카메라 위치 (활동구역을 비스듬히 내려다봄)
CAMERA_TARGET = (0.0,  0.0, 0.5)        # 카메라가 보는 지점 (원점 약간 위)
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

    def _apply_action(self):
        super()._apply_action()
        # iw.hub 외형을 큐브 root pose에 매 스텝 맞춤
        for i, robot in enumerate(self.robots):
            self._robot_visuals[i].set_world_poses(
                positions=robot.data.root_pos_w,
                orientations=robot.data.root_quat_w,
            )

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
        _s = self.cfg.robot_visual_scale if hasattr(self.cfg, "robot_visual_scale") else 1.0
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

            # ── 오픈 랙 외형 (기둥 4 + 선반판 3단). 충돌 박스 자리에 정확히 일치 ──
            #    footprint 3.0(x) × 0.5(y), 높이 1.5. 로봇이 피하는 박스와 동일 위치.
            rack_mat = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.24, 0.32), metallic=0.85, roughness=0.3)
            shelf_mat = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.62, 0.45, 0.24), metallic=0.0, roughness=0.7)  # 나무 선반판
            post = sim_utils.CuboidCfg(size=(0.10, 0.10, 1.5), visual_material=rack_mat)
            plank = sim_utils.CuboidCfg(size=(3.0, 0.5, 0.06), visual_material=shelf_mat)
            for s_i, (cx, cy, cz) in enumerate(SHELF_CENTERS):
                base = f"/World/envs/env_0/Rack_{s_i}"
                for p_i, (dx, dy) in enumerate([(-1.45, -0.2), (1.45, -0.2), (-1.45, 0.2), (1.45, 0.2)]):
                    post.func(f"{base}/post_{p_i}", post, translation=(cx + dx, cy + dy, 0.75))
                for l_i, lz in enumerate([0.05, 0.75, 1.45]):
                    plank.func(f"{base}/plank_{l_i}", plank, translation=(cx, cy, lz))
            print("[Demo] 오픈 랙 외형 생성 (기둥+3단 선반판)")

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        for i, robot in enumerate(self.robots):
            self.scene.rigid_objects[f"robot_{i}"] = robot

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 0.98, 0.95))
        light_cfg.func("/World/Light", light_cfg)


def main():
    env_cfg = WarehouseDemoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = WarehouseDemoEnv(env_cfg)
    env = RslRlVecEnvWrapper(env)
    env = IPPOReshapeWrapper(env, N_ROBOTS, OBS_PER_ROBOT)

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
    try:
        env.unwrapped.sim.set_camera_view(eye=CAMERA_EYE, target=CAMERA_TARGET)
        print(f"[Demo] 카메라 고정: eye={CAMERA_EYE} target={CAMERA_TARGET}")
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

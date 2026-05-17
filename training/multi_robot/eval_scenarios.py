"""Phase 3 — 고정 시나리오 5종 평가.

IPPO 또는 MAPPO 체크포인트를 로드해 각 시나리오에서 N 에피소드 실행.
결과: 충돌률, 교착률, 평균 완료 시간, 전원 도달률.

실행:
  # MAPPO 평가 (기본)
  python training/multi_robot/eval_scenarios.py \
    --ckpt logs/warehouse_ippo/model_400.pt --num_episodes 100

  python training/multi_robot/eval_scenarios.py \
    --ckpt logs/warehouse_mappo/model_5399.pt --num_episodes 100 --tag mappo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 3 고정 시나리오 평가")
parser.add_argument("--ckpt",          type=str, required=True, help="체크포인트 경로")
parser.add_argument("--num_episodes",  type=int, default=100,   help="시나리오별 에피소드 수")
parser.add_argument("--tag",           type=str, default="model", help="결과 레이블")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math
import json
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_marl_env import WarehouseMARLEnv, WarehouseMARLEnvCfg, N_ROBOTS, OBS_PER_ROBOT
from envs.warehouse.ippo_wrapper import IPPOReshapeWrapper


# ──────────────────────────────────────────────────────────────────────────────
# 고정 시나리오 정의 (로봇 좌표 & 목표 좌표 — env 원점 기준)
# ──────────────────────────────────────────────────────────────────────────────

# 각 시나리오: list of (spawn_xy, goal_xy) per robot, length == N_ROBOTS
SCENARIOS: dict[str, dict] = {
    "S1_headon": {
        "desc": "정면 충돌 — 좁은 복도 양방향 진입",
        # 로봇 0 ← 동쪽에서 서쪽으로, 로봇 1 → 서쪽에서 동쪽으로, 로봇 2 관련 없는 위치
        "spawns": [(-3.0, 0.0), (3.0, 0.0), (0.0, 3.5)],
        "goals":  [( 3.0, 0.0), (-3.0, 0.0), (0.0, -3.5)],
    },
    "S2_3way_deadlock": {
        "desc": "3-way 교착 — 삼각 대치",
        "spawns": [(0.0, 2.0), (-1.73, -1.0), (1.73, -1.0)],
        "goals":  [(0.0, -2.0), (1.73,  1.0), (-1.73, 1.0)],
    },
    "S3_battery_priority": {
        "desc": "배터리 우선순위 — 배터리 낮은 로봇이 좁은 통로 우선 통과",
        # 시나리오 구조: 로봇 0 (배터리 낮음, 긴급) vs 로봇 1·2 (여유)
        # MPG는 배터리를 직접 모델링하지 않으므로 좁은 통로 경쟁으로 대체
        "spawns": [(-2.5, 0.0), (-2.5, 0.5), (3.0, 0.5)],
        "goals":  [( 3.0, 0.0), ( 3.0, 0.5), (-2.5, 0.5)],
    },
    "S4_same_goal": {
        "desc": "동일 목표 경쟁 — 두 로봇이 같은 목표 지점 도달 경쟁",
        "spawns": [(-2.0, -1.5), (-2.0, 1.5), (3.0, 0.0)],
        "goals":  [( 2.0,  0.0), ( 2.0, 0.0), (-3.0, 0.0)],
    },
    "S5_mixed": {
        "desc": "장애물 + 다중 로봇 혼합 — 선반 사이 밀집 이동",
        "spawns": [(-3.0, -3.0), (-3.0, 3.0), (3.0, 0.0)],
        "goals":  [( 3.0,  3.0), ( 3.0, -3.0), (-3.0, 0.0)],
    },
}


# ──────────────────────────────────────────────────────────────────────────────

def load_policy(ckpt_path: str, device: str):
    """체크포인트에서 actor 가중치 추출 → callable."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw  = ckpt.get("model_state_dict", ckpt)

    import torch.nn as nn

    class ActorMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers: list[nn.Module] = []
            in_dim = OBS_PER_ROBOT
            for h in [256, 128, 64]:
                layers += [nn.Linear(in_dim, h), nn.ELU()]
                in_dim = h
            layers.append(nn.Linear(in_dim, 3))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).tanh()

    actor = ActorMLP().to(device)

    # rsl_rl 저장 키 패턴: 'actor.net.0.weight'
    sd = {}
    for k, v in raw.items():
        if k.startswith("actor.net."):
            sd[k[len("actor."):]] = v
        elif k.startswith("actor."):
            sd["net." + k[len("actor."):]] = v

    missing, unexpected = actor.load_state_dict(sd, strict=False)
    if missing:
        print(f"[경고] 누락 가중치: {missing[:5]}")
    actor.eval()
    return actor


@torch.inference_mode()
def run_scenario(
    scenario_name: str,
    scenario: dict,
    actor: "torch.nn.Module",
    num_episodes: int,
    device: str,
    env: "WarehouseMARLEnv",
) -> dict:
    """고정 시나리오 N 에피소드 실행 → 통계 반환."""
    spawns = scenario["spawns"]   # list of (x, y)
    goals  = scenario["goals"]

    stats = {
        "scenario": scenario_name,
        "desc": scenario["desc"],
        "n_episodes": num_episodes,
        "collision": 0,
        "deadlock": 0,
        "all_reached": 0,
        "ep_len_sum": 0.0,
    }

    for ep in range(num_episodes):
        # 수동 리셋: 고정 스폰 + 목표
        env._reset_idx(None)
        orig = env.scene.env_origins[0, :2]

        # 로봇 고정 배치
        for i, robot in enumerate(env.robots):
            state = robot.data.default_root_state[0:1].clone()
            state[0, 0] = orig[0] + spawns[i][0]
            state[0, 1] = orig[1] + spawns[i][1]
            state[0, 2] = 0.15
            state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            state[0, 7:] = 0.0
            robot.write_root_state_to_sim(state, torch.tensor([0], device=device))

        # 목표 고정
        for i in range(N_ROBOTS):
            env._goal_pos_w[0, i, 0] = orig[0] + goals[i][0]
            env._goal_pos_w[0, i, 1] = orig[1] + goals[i][1]

        ep_done = False
        ep_len  = 0
        ep_collision  = False
        ep_all_reached = False

        while not ep_done:
            # 관측 구성
            obs_dict = env._get_observations()
            obs = obs_dict["policy"]   # (1, N_ROBOTS * OBS_PER_ROBOT)

            # per-robot 행동 추론
            actions = []
            for i in range(N_ROBOTS):
                o_i = obs[:, i * OBS_PER_ROBOT: (i + 1) * OBS_PER_ROBOT]
                a_i = actor(o_i)
                actions.append(a_i)
            action_flat = torch.cat(actions, dim=1)   # (1, N_ROBOTS * 3)

            env._pre_physics_step(action_flat)
            for _ in range(env.cfg.decimation):
                env._apply_action()
                env.sim.step()
                env.scene.update(env.cfg.sim.dt)

            env.episode_length_buf += 1
            ep_len += 1

            terminated, timed_out = env._get_dones()
            ep_done = terminated[0].item() or timed_out[0].item()

            if terminated[0].item():
                # 충돌 or 전원 도달 구분
                from envs.warehouse.warehouse_marl_env import ROBOT_COLLISION_DIST
                positions = torch.stack([r.data.root_pos_w[0, :2] for r in env.robots], dim=0)
                for ii in range(N_ROBOTS):
                    for jj in range(ii + 1, N_ROBOTS):
                        d = (positions[ii] - positions[jj]).norm()
                        if d < ROBOT_COLLISION_DIST:
                            ep_collision = True
                if not ep_collision:
                    ep_all_reached = True

        stats["ep_len_sum"] += ep_len
        if ep_collision:
            stats["collision"] += 1
        elif ep_len >= env.max_episode_length - 1:
            stats["deadlock"] += 1
        elif ep_all_reached:
            stats["all_reached"] += 1

    n = num_episodes
    stats["collision_rate"]   = stats["collision"]   / n
    stats["deadlock_rate"]    = stats["deadlock"]     / n
    stats["all_reached_rate"] = stats["all_reached"]  / n
    stats["avg_ep_len"]       = stats["ep_len_sum"]   / n

    return stats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Eval] 체크포인트: {args.ckpt}")
    print(f"[Eval] 시나리오별 에피소드: {args.num_episodes}, 디바이스: {device}\n")

    actor = load_policy(args.ckpt, device)

    # 환경을 한 번만 생성하고 모든 시나리오에서 재사용
    env_cfg = WarehouseMARLEnvCfg()
    env_cfg.scene.num_envs = 1
    env = WarehouseMARLEnv(env_cfg)

    results = []
    for name, scenario in SCENARIOS.items():
        print(f"── {name}: {scenario['desc']}")
        stats = run_scenario(name, scenario, actor, args.num_episodes, device, env)
        results.append(stats)
        print(f"   충돌률: {stats['collision_rate']:.1%}  "
              f"교착률: {stats['deadlock_rate']:.1%}  "
              f"전원도달: {stats['all_reached_rate']:.1%}  "
              f"평균스텝: {stats['avg_ep_len']:.1f}\n")

    # 요약
    print("=" * 60)
    print(f"  태그: {args.tag}  |  체크포인트: {Path(args.ckpt).name}")
    print("=" * 60)
    avg_col  = sum(r["collision_rate"]   for r in results) / len(results)
    avg_dead = sum(r["deadlock_rate"]    for r in results) / len(results)
    avg_ok   = sum(r["all_reached_rate"] for r in results) / len(results)
    print(f"  평균 충돌률: {avg_col:.1%}  (목표: < 1%)")
    print(f"  평균 교착률: {avg_dead:.1%}  (목표: < 1%)")
    print(f"  평균 전원도달률: {avg_ok:.1%}\n")

    # JSON 저장
    out_path = Path("logs") / f"eval_{args.tag}_{Path(args.ckpt).stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "ckpt": args.ckpt, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"결과 저장: {out_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

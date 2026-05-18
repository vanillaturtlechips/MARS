"""Phase 3 — 고정 시나리오 5종 평가 (병렬 eval 지원).

실행:
  python training/multi_robot/eval_scenarios.py \
    --ckpt logs/warehouse_mappo/model_4999.pt --num_episodes 100 --tag mappo

  # 빠른 eval (16 병렬 env)
  python training/multi_robot/eval_scenarios.py \
    --ckpt logs/warehouse_mappo/model_4999.pt --num_episodes 20 \
    --num_eval_envs 16 --tag quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 3 고정 시나리오 평가")
parser.add_argument("--ckpt",           type=str, required=True)
parser.add_argument("--num_episodes",   type=int, default=100)
parser.add_argument("--num_eval_envs",  type=int, default=16, help="병렬 eval env 수")
parser.add_argument("--tag",            type=str, default="model")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os as _os
_rsl_rl_src = "/workspace/rsl_rl"
if _os.path.isdir(_rsl_rl_src) and _rsl_rl_src not in sys.path:
    sys.path.insert(0, _rsl_rl_src)

import json
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_marl_env import (
    WarehouseMARLEnv, WarehouseMARLEnvCfg, N_ROBOTS, OBS_PER_ROBOT, ROBOT_COLLISION_DIST
)

SCENARIOS: dict[str, dict] = {
    "S1_headon": {
        "desc": "정면 충돌 — 좁은 복도 양방향 진입",
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
        "spawns": [(-2.5, 0.0), (-2.5, 0.8), (3.0, 0.8)],
        "goals":  [( 3.0, 0.0), ( 3.0, 0.8), (-2.5, 0.8)],
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


def load_policy(ckpt_path: str, device: str):
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
    sd = {}
    for k, v in raw.items():
        if k.startswith("actor.net."):
            sd[k[len("actor."):]] = v
        elif k.startswith("actor."):
            sd["net." + k[len("actor."):]] = v

    missing, _ = actor.load_state_dict(sd, strict=False)
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
    """N_ENVS 병렬 실행으로 고정 시나리오 평가."""
    N_ENVS   = env.num_envs
    spawns   = scenario["spawns"]
    goals    = scenario["goals"]

    stats = {
        "scenario": scenario_name, "desc": scenario["desc"],
        "n_episodes": num_episodes,
        "collision": 0, "deadlock": 0, "all_reached": 0, "ep_len_sum": 0.0,
    }

    episodes_collected = 0

    while episodes_collected < num_episodes:
        this_batch = min(N_ENVS, num_episodes - episodes_collected)
        all_ids    = torch.arange(N_ENVS, device=device)

        # 모든 env 리셋 후 고정 스폰/목표 세팅
        env._reset_idx(all_ids)
        orig = env.scene.env_origins[:, :2]   # (N_ENVS, 2)

        for i, robot in enumerate(env.robots):
            state = robot.data.default_root_state.clone()
            state[:, 0] = orig[:, 0] + spawns[i][0]
            state[:, 1] = orig[:, 1] + spawns[i][1]
            state[:, 2] = 0.15
            state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            state[:, 7:]  = 0.0
            robot.write_root_state_to_sim(state, all_ids)

        for i in range(N_ROBOTS):
            env._goal_pos_w[:, i, 0] = orig[:, 0] + goals[i][0]
            env._goal_pos_w[:, i, 1] = orig[:, 1] + goals[i][1]

        recorded  = torch.zeros(N_ENVS, dtype=torch.bool, device=device)
        ep_lens   = torch.zeros(N_ENVS, device=device)
        collision = torch.zeros(N_ENVS, dtype=torch.bool, device=device)
        reached   = torch.zeros(N_ENVS, dtype=torch.bool, device=device)

        while not recorded[:this_batch].all():
            obs_dict    = env._get_observations()
            obs         = obs_dict["policy"]

            actions = []
            for i in range(N_ROBOTS):
                o_i = obs[:, i * OBS_PER_ROBOT:(i + 1) * OBS_PER_ROBOT]
                actions.append(actor(o_i))
            action_flat = torch.cat(actions, dim=1)
            action_flat[recorded] = 0.0   # 완료 env는 정지

            env._pre_physics_step(action_flat)
            for _ in range(env.cfg.decimation):
                env._apply_action()
                env.sim.step()
                env.scene.update(env.cfg.sim.dt)

            # 미완료 env만 카운트
            env.episode_length_buf[~recorded] += 1
            ep_lens[~recorded] += 1

            terminated, timed_out = env._get_dones()
            just_done = (terminated | timed_out) & ~recorded

            if just_done.any():
                positions = torch.stack(
                    [r.data.root_pos_w[:, :2] for r in env.robots], dim=1
                )
                for idx in just_done.nonzero(as_tuple=True)[0]:
                    if terminated[idx] and not timed_out[idx]:
                        col = False
                        for ii in range(N_ROBOTS):
                            for jj in range(ii + 1, N_ROBOTS):
                                if (positions[idx, ii] - positions[idx, jj]).norm() < ROBOT_COLLISION_DIST:
                                    col = True
                        collision[idx] = col
                        if not col:
                            reached[idx] = True
                recorded |= just_done

            # 강제 종료
            recorded[:this_batch] |= (ep_lens[:this_batch] >= env.max_episode_length - 1)

        for idx in range(this_batch):
            ep_l = ep_lens[idx].item()
            stats["ep_len_sum"] += ep_l
            if collision[idx]:
                stats["collision"] += 1
            elif reached[idx]:
                stats["all_reached"] += 1
            else:
                stats["deadlock"] += 1

        episodes_collected += this_batch

    n = num_episodes
    stats["collision_rate"]   = stats["collision"]   / n
    stats["deadlock_rate"]    = stats["deadlock"]     / n
    stats["all_reached_rate"] = stats["all_reached"]  / n
    stats["avg_ep_len"]       = stats["ep_len_sum"]   / n
    return stats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Eval] 체크포인트: {args.ckpt}")
    print(f"[Eval] 시나리오별 에피소드: {args.num_episodes}, 병렬 env: {args.num_eval_envs}, 디바이스: {device}\n")

    actor = load_policy(args.ckpt, device)

    env_cfg = WarehouseMARLEnvCfg()
    env_cfg.scene.num_envs = args.num_eval_envs
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

    print("=" * 60)
    print(f"  태그: {args.tag}  |  체크포인트: {Path(args.ckpt).name}")
    print("=" * 60)
    avg_col  = sum(r["collision_rate"]   for r in results) / len(results)
    avg_dead = sum(r["deadlock_rate"]    for r in results) / len(results)
    avg_ok   = sum(r["all_reached_rate"] for r in results) / len(results)
    print(f"  평균 충돌률: {avg_col:.1%}  (목표: < 1%)")
    print(f"  평균 교착률: {avg_dead:.1%}  (목표: < 1%)")
    print(f"  평균 전원도달률: {avg_ok:.1%}\n")

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

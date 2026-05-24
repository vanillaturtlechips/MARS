"""Phase 2 Student Transport 환경 진단 스크립트 — 10가지 핵심 요소 검증.

실행:
  python training/single_robot/diag_transport.py --headless --num_envs 4
"""
from __future__ import annotations
import sys
from pathlib import Path
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_manipulation_env import (
    WarehouseManipulationStudentTransportEnvCfg,
    WarehouseManipulationEnv,
    PLACE_GOALS,
    TEACHER_OBS_DIM, STUDENT_OBS_DIM,
)

OK = "✅ PASS"
NG = "❌ FAIL"
results: list[tuple[str, bool]] = []


def chk(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  {'✅ PASS' if cond else '❌ FAIL'}  {name}")
    if detail:
        print(f"         → {detail}")
    results.append((name, cond))
    return cond


def run():
    cfg = WarehouseManipulationStudentTransportEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = WarehouseManipulationEnv(cfg)
    N, D = args.num_envs, env.device
    place_goals_t = torch.tensor(PLACE_GOALS, device=D, dtype=torch.float32)
    zero_act = torch.zeros(N, 9, device=D)

    print("\n" + "=" * 70)
    print("  Phase 2 Student Transport 환경 진단")
    print("=" * 70)

    obs_dict, _ = env.reset()

    # ------------------------------------------------------------------ #
    # [1] force_grasp 초기 상태
    print("\n[1] force_grasp 초기 상태 ----------------------------------------")
    chk("_grasped = True (all envs)",
        env._grasped.all().item(),
        f"_grasped = {env._grasped.tolist()}")
    chk("_grasp_ee_offset = 0 (all envs)",
        env._grasp_ee_offset.abs().max().item() < 1e-6,
        f"max|offset| = {env._grasp_ee_offset.abs().max().item():.6f}")

    # ------------------------------------------------------------------ #
    # [2] Box spawn 위치 + EE 위치 (reset 직후 apply_action 전)
    print("\n[2] Box spawn 위치 vs EE 위치 -----------------------------------")
    ee_pos, _ = env._get_ee_pose()
    box_pos = env.box.data.root_pos_w
    box_local = box_pos - env.scene.env_origins
    ee_local  = ee_pos - env.scene.env_origins

    chk("box_local.x ≈ 0.307 (EE reach_pose x)",
        (box_local[:, 0] - 0.307).abs().max().item() < 0.02,
        f"box_local_x = {box_local[:, 0].tolist()}")
    chk("box_local.y ≈ 0.00",
        box_local[:, 1].abs().max().item() < 0.02,
        f"box_local_y = {box_local[:, 1].tolist()}")
    chk("box_local.z ≈ 0.590 (EE reach_pose z)",
        (box_local[:, 2] - 0.590).abs().max().item() < 0.02,
        f"box_local_z = {box_local[:, 2].tolist()}")

    print(f"         [INFO] EE local (reach_pose) = {ee_local[0].tolist()}")
    ee_box_err = (ee_pos - box_pos).norm(dim=1)
    chk("EE ≈ box_spawn (< 0.05m)",
        ee_box_err.max().item() < 0.05,
        f"EE-box dist = {ee_box_err.tolist()}  ← 차이 있으면 step 1에서 box 텔레포트 발생")

    # ------------------------------------------------------------------ #
    # [3] Goal spawn (PLACE_GOALS 4개 중 하나, init_dist 0.30~0.36m)
    print("\n[3] Goal spawn 위치 --------------------------------------------")
    goal_local = env._goal_pos_w - env.scene.env_origins
    valid = []
    for i in range(N):
        dists_to_goals = [(goal_local[i] - pg).norm().item() for pg in place_goals_t]
        valid.append(min(dists_to_goals) < 0.01)
    chk("goal = PLACE_GOALS 4개 중 하나",
        all(valid),
        f"goal_local = {goal_local.tolist()}")

    init_dists = env._prev_dist_box_goal
    chk("_prev_dist_box_goal: 0.29~0.40m (EE→PLACE_GOALS 실제 거리)",
        ((init_dists > 0.29) & (init_dists < 0.40)).all().item(),
        f"init_dists = {init_dists.tolist()}")

    # 중요: _prev_dist_box_goal은 box_spawn 기준인가 EE 기준인가?
    dist_from_box = (box_pos - env._goal_pos_w).norm(dim=1)
    dist_from_ee  = (ee_pos  - env._goal_pos_w).norm(dim=1)
    print(f"         [INFO] dist(box_spawn→goal) = {dist_from_box.tolist()}")
    print(f"         [INFO] dist(EE→goal)        = {dist_from_ee.tolist()}")
    print(f"         [INFO] _prev_dist_box_goal   = {init_dists.tolist()}")
    chk("_prev_dist_box_goal ≈ dist(box_spawn→goal)",
        (init_dists - dist_from_box).abs().max().item() < 0.01,
        "일치하지 않으면 step 1에서 delta_goal 계산 오류 가능")

    # ------------------------------------------------------------------ #
    # [4] Obs 차원 확인
    print("\n[4] Obs 차원 (policy=29D, critic=30D) --------------------------")
    obs_policy = obs_dict["policy"]
    obs_critic  = obs_dict.get("critic")
    chk("policy obs = 29D",
        obs_policy.shape == (N, STUDENT_OBS_DIM),
        f"shape = {tuple(obs_policy.shape)}")
    chk("critic obs = 30D (asymmetric AC)",
        obs_critic is not None and obs_critic.shape == (N, TEACHER_OBS_DIM),
        f"critic shape = {tuple(obs_critic.shape) if obs_critic is not None else 'None'}")
    if obs_critic is not None:
        chk("policy ≠ critic (서로 다른 obs)",
            not torch.allclose(obs_policy[:, :min(29, 30)], obs_critic[:, :min(29, 30)]),
            "같으면 asymmetric AC가 실제로 적용 안 된 것")

    # ------------------------------------------------------------------ #
    # [5] goal_rel in obs: indices 4:7, 방향·크기 확인
    # student obs: [ee_pos_local(0:3), gripper(3:4), goal_rel(4:7), noisy_box_rel(7:10), grasped(10:11), jpos(11:20), jvel(20:29)]
    print("\n[5] Obs goal_rel (index 4:7) 방향·크기 --------------------------")
    obs_goal_rel = obs_policy[:, 4:7]  # (N, 3)
    true_goal_rel = env._goal_pos_w - (ee_pos + env._grasp_ee_offset)  # = goal - ee
    cos = (obs_goal_rel * true_goal_rel).sum(dim=1) / (
        obs_goal_rel.norm(dim=1).clamp(1e-4) * true_goal_rel.norm(dim=1).clamp(1e-4)
    )
    chk("goal_rel cosine ≥ 0.99 (방향 일치)",
        cos.min().item() >= 0.99,
        f"cos_sim = {[f'{c:.4f}' for c in cos.tolist()]}")
    mag_err = (obs_goal_rel.norm(dim=1) - true_goal_rel.norm(dim=1)).abs()
    chk("goal_rel magnitude 일치 (< 0.01m)",
        mag_err.max().item() < 0.01,
        f"obs_mag={obs_goal_rel.norm(dim=1).tolist()}, true_mag={true_goal_rel.norm(dim=1).tolist()}")

    # grasped obs: index 10
    obs_grasped = obs_policy[:, 10]
    chk("grasped obs (index 10) = 1.0",
        (obs_grasped - 1.0).abs().max().item() < 1e-4,
        f"grasped_obs = {obs_grasped.tolist()}")

    # noisy_box_rel: index 7:10. force_grasp에서 box=EE → box_rel_true≈0 → noisy_box_rel≈noise
    obs_noisy_box = obs_policy[:, 7:10]
    box_rel_true = box_pos - ee_pos
    print(f"         [INFO] box_rel_true(reset) = {box_rel_true[0].tolist()}")
    print(f"         [INFO] noisy_box_rel(obs)  = {obs_noisy_box[0].tolist()}")
    chk("box_rel_true 크기 < 0.05m (box ≈ EE at reset)",
        box_rel_true.norm(dim=1).max().item() < 0.05,
        f"box_rel_true norms = {box_rel_true.norm(dim=1).tolist()}")

    # ------------------------------------------------------------------ #
    # [6] Box tracking: step 후 box.pos ≈ EE pos
    print("\n[6] Box tracking (step 후 box = EE) ----------------------------")
    env.reset()
    obs_d2, _, _, _, _ = env.step(zero_act)
    ee2, _ = env._get_ee_pose()
    box2 = env.box.data.root_pos_w
    box_ee_err = (box2 - ee2).norm(dim=1)
    chk("step 후 box tracks EE (err < 0.01m)",
        box_ee_err.max().item() < 0.01,
        f"box-EE dist = {box_ee_err.tolist()}")

    # ------------------------------------------------------------------ #
    # [7] Transport reward 방향성: joint0를 goal y 방향으로 돌리면 dist 감소?
    print("\n[7] Transport reward 방향성 (goal 방향 이동 → dist 감소?) --------")
    env.reset()
    goal7 = env._goal_pos_w.clone()
    goal_local7 = goal7 - env.scene.env_origins
    j0_sign = torch.sign(goal_local7[:, 1])  # goal y 부호 (N,)

    dist_before7 = env._prev_dist_box_goal.clone()
    rew_list7: list[float] = []

    for _ in range(30):
        act7 = torch.zeros(N, 9, device=D)
        act7[:, 0] = j0_sign * 1.0  # joint0 최대 델타
        _, rew7, done7, trunc7, _ = env.step(act7)
        rew_list7.append(rew7.mean().item())
        if done7.any() or trunc7.any():
            break

    ee7, _ = env._get_ee_pose()
    dist_after7 = (ee7 - goal7).norm(dim=1)
    dist_improved = (dist_before7 - dist_after7 > 0).sum().item()

    print(f"         [INFO] dist before: {dist_before7.tolist()}")
    print(f"         [INFO] dist after 30 steps toward goal: {dist_after7.tolist()}")
    chk(f"goal-directed motion: dist reduced for ≥{N//2} envs",
        dist_improved >= N // 2,
        f"improved {dist_improved}/{N} envs")
    chk("mean reward > -500 during directed motion",
        sum(rew_list7) / len(rew_list7) > -500,
        f"mean_rew = {sum(rew_list7)/len(rew_list7):.1f}")

    # ------------------------------------------------------------------ #
    # [8] Place 트리거: goal을 EE 옆으로 이동 → terminated
    print("\n[8] Place 트리거 (dist < 0.12m → terminate + 800rew) -----------")
    env.reset()
    ee8, _ = env._get_ee_pose()
    # goal을 EE 바로 옆 0.05m로 이동 (< place_dist_threshold=0.12m)
    env._goal_pos_w[:] = ee8 + torch.tensor([0.03, 0.03, 0.03], device=D)
    env._prev_dist_box_goal[:] = 0.052  # sqrt(3*0.03²)

    _, rew8, tm8, _, _ = env.step(zero_act)
    chk("place → episode terminated",
        tm8.all().item(),
        f"terminated = {tm8.tolist()}, rewards = {rew8.tolist()}")
    chk("place reward ≥ 700 (rew_place=800 + rew_time=-0.02)",
        (rew8 >= 700).all().item(),
        f"rewards = {[f'{r:.1f}' for r in rew8.tolist()]}")

    # ------------------------------------------------------------------ #
    # [9] approach reward = 0 (force_grasp에서 not_grasped=0)
    print("\n[9] Approach reward = 0 (not_grasped=0) ------------------------")
    env.reset()
    _, rew9, _, _, ex9 = env.step(zero_act)
    log9 = ex9.get("log", {})
    dist_ee_box = log9.get("dist_ee_box", -1)
    grasp_rate  = log9.get("grasp_rate",  0)
    chk("dist_ee_box ≈ 0 (box at EE)",
        dist_ee_box < 0.02,
        f"dist_ee_box = {dist_ee_box:.4f}m")
    chk("grasp_rate = 100%",
        grasp_rate >= 99.9,
        f"grasp_rate = {grasp_rate:.2f}%")

    # ------------------------------------------------------------------ #
    # [10] delta_goal 계산 검증 (zero action → delta ≈ 0, 향해 이동 → delta > 0)
    print("\n[10] delta_goal 계산 검증 --------------------------------------")
    # [10-a] zero action → EE 거의 안 움직임 → delta ≈ 0 → transport ≈ 0
    env.reset()
    dists_zero: list[float] = []
    for _ in range(5):
        d_prev = env._prev_dist_box_goal.clone()
        _, rew_z, _, _, _ = env.step(zero_act)
        d_now = env._prev_dist_box_goal.clone()
        dists_zero.append(d_now.mean().item())

    drift = max(dists_zero) - min(dists_zero)
    chk("zero action → dist_box_goal drift < 0.05m (5 steps)",
        drift < 0.05,
        f"dists = {[f'{d:.4f}' for d in dists_zero]}, drift = {drift:.4f}")

    # [10-b] goal 방향 1 step → delta_goal > 0 → rew_transport > 0?
    env.reset()
    goal10 = env._goal_pos_w.clone()
    goal_local10 = goal10 - env.scene.env_origins
    j0_sign10 = torch.sign(goal_local10[:, 1])

    d_before10 = env._prev_dist_box_goal.clone()
    act10 = torch.zeros(N, 9, device=D)
    act10[:, 0] = j0_sign10 * 1.0
    _, rew10, _, _, ex10 = env.step(act10)
    d_after10 = env._prev_dist_box_goal.clone()

    delta = d_before10 - d_after10
    transport_component = 300.0 * delta.clamp(-0.1, 0.1)
    print(f"         [INFO] d_before={d_before10.tolist()}")
    print(f"         [INFO] d_after ={d_after10.tolist()}")
    print(f"         [INFO] delta   ={delta.tolist()} (양수=goal 방향 이동)")
    print(f"         [INFO] transport_component = {transport_component.tolist()}")
    chk("goal 방향 1-step: _prev_dist_box_goal 업데이트됨",
        (d_before10 - d_after10).abs().max().item() > 1e-6,
        f"delta = {(d_before10 - d_after10).tolist()}")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    passed = sum(ok for _, ok in results)
    total  = len(results)
    print(f"  결과: {passed}/{total} 통과")
    if passed < total:
        print("  실패 항목:")
        for name, ok in results:
            if not ok:
                print(f"    ❌  {name}")
    print("=" * 70 + "\n")
    env.close()


if __name__ == "__main__":
    run()
    simulation_app.close()

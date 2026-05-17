"""True CTDE 래퍼 (Centralized Training, Decentralized Execution).

Actor : per-robot obs (9-dim)  — 각 로봇 독립 실행
Critic: global obs (27-dim)   — 전체 상태 중앙 평가

rsl_rl 3.x는 get_observations() / step() 이 'policy' 키를 가진
TensorDict 를 반환하길 기대함. 'critic' 키를 추가해 CTDE 구현.
"""

from __future__ import annotations

import collections.abc

import gymnasium as gym
import numpy as np
import torch


class IPPOReshapeWrapper:
    """rsl_rl OnPolicyRunner ↔ RslRlVecEnvWrapper 사이 삽입 래퍼.

    Parameters
    ----------
    vec_env        : RslRlVecEnvWrapper (num_envs = E)
    n_robots       : 로봇 수 N
    obs_per_robot  : 로봇 1대 관측 차원 (actor obs = 9)
    act_per_robot  : 로봇 1대 행동 차원 (기본 3)
    """

    def __init__(self, vec_env, n_robots: int, obs_per_robot: int, act_per_robot: int = 3):
        self._env = vec_env
        self.n = n_robots
        self.obs_per_robot = obs_per_robot
        self.act_per_robot = act_per_robot
        self._E = vec_env.num_envs

    # ── rsl_rl 필수 속성 ──────────────────────────────────────────
    @property
    def num_envs(self) -> int:
        return self._E * self.n

    @property
    def num_obs(self) -> int:
        return self.obs_per_robot          # actor: 9-dim per-robot

    @property
    def num_actions(self) -> int:
        return self.act_per_robot

    @property
    def num_privileged_obs(self) -> int:
        return self.obs_per_robot * self.n  # critic: 27-dim global state

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_per_robot,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.act_per_robot,),
            dtype=np.float32,
        )

    @property
    def device(self):
        return self._env.device

    # ── rsl_rl 3.x 인터페이스 ────────────────────────────────────
    def step(self, actions: torch.Tensor):
        """(E×N, act_per_robot) → joint step → TensorDict{"policy","critic"}."""
        joint_act = actions.reshape(self._E, self.n * self.act_per_robot)
        obs, rew, dones, extras = self._env.step(joint_act)

        obs_out = self._build_obs_dict(obs)         # TensorDict (E*N,)

        # per-robot 보상 분리 (credit assignment)
        per_robot_rew = extras.pop("per_robot_rewards", None)
        if per_robot_rew is not None:
            rew_exp = per_robot_rew.reshape(-1)     # (E*N,)
        else:
            rew_exp = rew.repeat_interleave(self.n, dim=0)

        dones_exp  = dones.repeat_interleave(self.n, dim=0)
        extras_exp = self._expand_extras(extras)

        return obs_out, rew_exp, dones_exp, extras_exp

    def get_observations(self):
        """rsl_rl 3.x OnPolicyRunner은 TensorDict 단독 반환을 기대함 (튜플 아님).
        on_policy_runner.py line 43: obs = self.env.get_observations()
        on_policy_runner.py line 70: obs = self.env.get_observations().to(device)
        """
        result = self._env.get_observations()
        obs = result[0] if isinstance(result, tuple) else result
        return self._build_obs_dict(obs)  # TensorDict 단독 반환

    def reset(self):
        result = self._env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        return self._build_obs_dict(obs)  # TensorDict 단독 반환

    # ── 내부 유틸 ─────────────────────────────────────────────────
    def _build_obs_dict(self, obs) -> "TensorDict":
        """raw obs → TensorDict{"policy": (E*N,9), "critic": (E*N,27)}.

        TensorDict로 반환해야 .to(device) 호출이 동작함.
        rsl_rl이 "policy" in TensorDict 로 키를 자동 감지함.
        """
        from tensordict import TensorDict

        actor_obs  = self._split_actor_obs(obs)    # (E*N, 9)
        critic_obs = self._expand_critic_obs(obs)  # (E*N, 27)
        return TensorDict(
            {"policy": actor_obs, "critic": critic_obs},
            batch_size=[self._E * self.n],
            device=self.device,
        )

    def _split_actor_obs(self, obs) -> torch.Tensor:
        """(E, 27) → (E*N, 9): per-robot actor 입력."""
        tensor = self._extract_tensor(obs, key="policy")
        return tensor.reshape(self._E * self.n, self.obs_per_robot)

    def _expand_critic_obs(self, obs) -> torch.Tensor:
        """(E, 27) → (E*N, 27): global state 복제."""
        tensor = self._extract_tensor(obs, key="critic")
        return tensor.repeat_interleave(self.n, dim=0)

    def _extract_tensor(self, obs, key: str) -> torch.Tensor:
        """TensorDict / dict / tensor 에서 key 에 해당하는 tensor 추출."""
        if isinstance(obs, torch.Tensor):
            return obs
        # TensorDict 또는 dict
        if hasattr(obs, "__getitem__"):
            try:
                return obs[key]
            except KeyError:
                pass
            # fallback: 첫 번째 값
            if hasattr(obs, "values"):
                return next(iter(obs.values()))
        return obs

    def _expand_extras(self, extras) -> dict:
        """shape (E, ...) 인 텐서를 (E×N, ...) 로 확장."""
        if not isinstance(extras, collections.abc.Mapping):
            return extras
        out = {}
        for k, v in extras.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == self._E:
                out[k] = v.repeat_interleave(self.n, dim=0)
            elif isinstance(v, collections.abc.Mapping):
                out[k] = {
                    kk: vv.repeat_interleave(self.n, dim=0)
                    if isinstance(vv, torch.Tensor) and vv.shape[0] == self._E
                    else vv
                    for kk, vv in v.items()
                }
            else:
                out[k] = v
        return out

    def __getattr__(self, name):
        return getattr(self._env, name)

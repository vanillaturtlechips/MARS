import gymnasium as gym

from . import agents
from .warehouse_env import WarehouseNavEnv, WarehouseNavEnvCfg
from .warehouse_obstacle_env import WarehouseObstacleNavEnv, WarehouseObstacleNavEnvCfg

gym.register(
    id="Isaac-WarehouseNav-Direct-v0",
    entry_point=f"{__name__}.warehouse_env:WarehouseNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.warehouse_env:WarehouseNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WarehouseNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WarehouseObstacleNav-Direct-v0",
    entry_point=f"{__name__}.warehouse_obstacle_env:WarehouseObstacleNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.warehouse_obstacle_env:WarehouseObstacleNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WarehouseNavPPORunnerCfg",
    },
)

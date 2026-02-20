import gymnasium as gym


gym.register(
    id="Isaac-Velocity-Rough-TienKung2Lite-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:TienKung2LiteRoughEnvCfg",
        "skrl_cfg_entry_point": f"{__name__}:skrl_rough_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-TienKung2Lite-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TienKung2LiteFlatEnvCfg",
        "skrl_cfg_entry_point": f"{__name__}:skrl_flat_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-TienKung2Lite-v1",
    entry_point=f"{__name__}.bounded_action_env:BoundedActionManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TienKung2LiteFlatEnvCfg_V1",
        "skrl_cfg_entry_point": f"{__name__}:skrl_flat_ppo_cfg_v1.yaml",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-TienKung2Lite-Play-v1",
    entry_point=f"{__name__}.bounded_action_env:BoundedActionManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TienKung2LiteFlatEnvCfg_PLAY_V1",
        "skrl_cfg_entry_point": f"{__name__}:skrl_flat_ppo_cfg_v1.yaml",
    },
)

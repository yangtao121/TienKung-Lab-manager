# TienKung-Lab Manager-Based Walk (skrl)

## Purpose
This package provides a manager-based locomotion task for TienKung2 Lite in IsaacLab.
Only the skrl training stack is supported.

## Install
```bash
pip install -e /Users/yangtao/code/Isaaclab-manager-based/TienKung-Lab-manager
```

## Train
```bash
python /Users/yangtao/code/Isaaclab-manager-based/TienKung-Lab-manager/scripts/train_skrl.py \
  --task Isaac-Velocity-Rough-TienKung2Lite-v0 --headless --num_envs 4096
```

## Registry Check
```bash
python -c "import gymnasium as gym; import tienkung_manager_lab; print(gym.spec('Isaac-Velocity-Rough-TienKung2Lite-v0'))"
```

## Notes
- Task ID: `Isaac-Velocity-Rough-TienKung2Lite-v0`
- Environment config class:
  `tienkung_manager_lab.manager_based.locomotion.velocity.config.tienkung2_lite.rough_env_cfg.TienKung2LiteRoughEnvCfg`
- skrl config entry point:
  `tienkung_manager_lab.manager_based.locomotion.velocity.config.tienkung2_lite.agents:skrl_rough_ppo_cfg.yaml`

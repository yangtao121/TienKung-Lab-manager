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
python -c "import gymnasium as gym; import locomotion; print(gym.spec('Isaac-Velocity-Rough-TienKung2Lite-v0'))"
```

## Notes
- Task ID: `Isaac-Velocity-Rough-TienKung2Lite-v0`
- Environment config class:
  `locomotion.rough_env_cfg.TienKung2LiteRoughEnvCfg`
- skrl config entry point:
  `locomotion:skrl_rough_ppo_cfg.yaml`

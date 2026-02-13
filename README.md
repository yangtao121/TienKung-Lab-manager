# TienKung-Lab Manager-Based Walk (PPO)

## Purpose
This project ports TienKung-Lab walk training to IsaacLab manager-based workflow (IsaacLab 2.3.1),
and is now self-contained for robot assets and terrain config.

## Install
1. Install this package:
   `pip install -e /Users/yangtao/code/Isaaclab-manager-based/TienKung-Lab-manager`

## Train
`python /Users/yangtao/code/Isaaclab-manager-based/TienKung-Lab-manager/scripts/train.py --task walk --headless --num_envs 4096 --logger tensorboard`

## Quick Verification
1. Verify asset path resolves inside this repository:
   `python -c "from tienkung_manager_lab.assets.tienkung2_lite import TIENKUNG2LITE_CFG; print(TIENKUNG2LITE_CFG.spawn.usd_path)"`
2. Run a minimal smoke test:
   `python /Users/yangtao/code/Isaaclab-manager-based/TienKung-Lab-manager/scripts/train.py --task walk --headless --num_envs 2`

## Notes
- This stage only includes walk PPO training.
- AMP is intentionally not enabled in this stage.
- Robot assets and terrain generator config are provided by `tienkung_manager_lab`.

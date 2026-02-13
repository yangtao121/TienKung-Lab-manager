# TienKung-Lab Manager-Based Walk (PPO)

## Purpose
This project ports TienKung-Lab walk training to IsaacLab manager-based workflow (IsaacLab 2.3.1),
while keeping the original direct project unchanged.

## Install Order
1. Install original TienKung-Lab package:
   `pip install -e /Users/yangtao/code/Isaaclab-manager-based/TienKung-Lab`
2. Install original TienKung-Lab `rsl_rl` package:
   `pip install -e /Users/yangtao/code/Isaaclab-manager-based/TienKung-Lab/rsl_rl`
3. Install this manager-based package:
   `pip install -e /Users/yangtao/code/Isaaclab-manager-based/TienKung-Lab-manager`

## Train
`python /Users/yangtao/code/Isaaclab-manager-based/TienKung-Lab-manager/scripts/train.py --task walk --headless --num_envs 4096 --logger tensorboard`

## Notes
- This stage only includes walk PPO training.
- AMP is intentionally not enabled in this stage.
- Robot assets, terrains and RSL-RL runtime are reused from original TienKung-Lab.

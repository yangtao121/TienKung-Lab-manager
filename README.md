# TienKung-Lab Manager-Based Walk (skrl)

## Purpose
This package provides a manager-based locomotion task for TienKung2 Lite in IsaacLab.
Only the skrl training stack is supported.

## Install
```bash
# 如果使用 Python 虚拟环境，确保先激活环境
# conda activate isaaclab
# 或 source env_isaaclab/bin/activate

# 如果是独立包，使用以下方式安装
pip install -e /path/to/TienKung-Lab-manager
```

## Train

### 方式一：使用 isaaclab.sh 脚本启动（推荐）

使用 `./isaaclab.sh -p` 方式启动，会自动使用正确的 Python 环境，无需手动激活环境：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Velocity-Rough-TienKung2Lite-v0 \
  --headless \
  --num_envs 4096
```

#### 带视频录制

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Velocity-Rough-TienKung2Lite-v0 \
  --video \
  --headless \
  --num_envs 4096
```

#### 自定义视频参数

```bash
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Velocity-Rough-TienKung2Lite-v0 \
  --video \
  --video_length 200 \
  --video_interval 2000 \
  --headless \
  --num_envs 4096 \
  --device cuda:0 \
  --seed 42
```

### 方式二：直接使用 Python 启动

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Velocity-Rough-TienKung2Lite-v0 \
  --headless \
  --num_envs 4096
```

#### 带视频录制

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Velocity-Rough-TienKung2Lite-v0 \
  --video \
  --headless \
  --num_envs 4096
```

#### 自定义视频参数

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Velocity-Rough-TienKung2Lite-v0 \
  --video \
  --video_length 200 \
  --video_interval 2000 \
  --headless \
  --num_envs 4096
```

### 参数说明
- `--task`: 指定任务名称（`Isaac-Velocity-Rough-TienKung2Lite-v0`）
- `--video`: 启用视频录制功能
- `--video_length`: 每个视频的长度（步数），默认200
- `--video_interval`: 录制间隔（步数），默认2000
- `--headless`: 无头模式运行
- `--num_envs`: 环境数量
- `--device`: 使用的设备（如 `cuda:0`）
- `--seed`: 随机种子

### 两种启动方式的区别
- **直接使用 `python`**：需要先激活 conda/uv 环境，确保 Isaac Lab 已安装
- **使用 `./isaaclab.sh -p`**：自动使用正确的 Python 环境，无需手动激活环境，更简洁方便

视频将保存在：`logs/skrl/tienkung2_lite_rough/{timestamp}_ppo_torch/videos/train/`

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

# RBY1 OpenPI

This repository contains RBY1 embodiment support for OpenPI.

For the original OpenPI documentation, refer to the upstream README:

https://github.com/Physical-Intelligence/openpi/blob/main/README.md

## Overview

This repository adds the components needed to use RBY1 as an embodiment in the OpenPI training and inference pipeline.

The RBY1 integration includes:

* `LeRobotRBY1DataConfig` for the RBY1 data format
* RBY1 input/output transforms and a policy wrapper
* Training and inference configuration for RBY1
* ZMQ-based policy serving support

These additions let you use the standard OpenPI training and inference flow while connecting RBY1 datasets and RBY1 robot clients.

## Purpose

The RBY1 integration configures RBY1-specific data repacking, input/output transforms, and model settings so that RBY1 can be trained and served as an embodiment in OpenPI.

## Key Files

| Item | File |
| ---- | ---- |
| RBY1 policy implementation | `src/openpi/policies/rby1_policy.py` |
| Data and training configuration | `src/openpi/training/config.py` |
| ZMQ policy server | `src/openpi/serving/zmq_policy_server.py` |

## Dataset Preparation

For instructions on creating RBY1 datasets, refer to the `rby1-lerobot` repository.

```text
https://github.com/RainbowRobotics/rby1-lerobot
```

## Installation

This repository uses `uv` for Python dependency management. First, follow the `uv` installation instructions and install `uv`.

After installing `uv`, run the following commands from the OpenPI repository root to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

These commands configure the OpenPI runtime environment and install the current repository in editable mode.

## Training

## 1. Convert the dataset

Convert your dataset from **LeRobot v3** to **LeRobot v2**.

Reference repository: [`lerobot_conversion`](https://github.com/RainbowRobotics/lerobot_conversion)

---

## 2. Create a training configuration

Register your own config in `_CONFIGS =` inside [src/openpi/training/config.py](https://github.com/RainbowRobotics/rby1-openpi/blob/0b4a2ce3df170c8691c092690bcae9521b8c6275/src/openpi/training/config.py#L617) 

For example:

```python
TrainConfig(
        name="pi05_rby1_your_config_name",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=40,
        ),
        data=LeRobotRBY1DataConfig(
            repo_id="your_dataset_repo_id",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        batch_size=32,
        num_train_steps=50_000,
        log_interval=100,
        save_interval=5_000,
        fsdp_devices=1,
        freeze_filter=nnx.Nothing,
        seed=1000,
        ),


```
Replace `your_dataset_repo_id` with your converted LeRobot v2 dataset (e.g. rainbowrobotics/my_dataset_v2).


### * If you use Single arm Training configuration

  Expected action/state layout:
  - dims 0:7: arm joints
  - dim 7: gripper

The single-arm data configuration is implemented as:

```text
LeRobotRBY1OneArmDataConfig
```

Example training configuration:

```python
TrainConfig(
    name="pi05_rby1_one_arm_sample",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=40,
    ),
    data=LeRobotRBY1OneArmDataConfig(
        repo_id="your_single_arm_dataset_repo_id",
        base_config=DataConfig(prompt_from_task=True),
        use_delta_joint_actions=True,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    batch_size=32,
    num_train_steps=50_000,
)
```

Replace `your_single_arm_dataset_repo_id` with the Hugging Face repository ID of the dataset.







---

## 3. Compute dataset normalization statistics

Before training, compute the normalization statistics for your dataset:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_rby1_your_config_name
```

---

## 4. Start training

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_rby1_your_config_name \
  --exp-name=my_experiment \
  --overwrite
```

#### Options

- `--exp-name` specifies the experiment name.
- `--overwrite` replaces existing checkpoints if the same experiment name already exists.

---

### Training outputs

During training:

- Training logs are printed to the console.
- Checkpoints are automatically saved in the `checkpoints/` directory.
- If **Weights & Biases (W&B)** is enabled, training progress can also be monitored from the W&B dashboard.

---

### GPU memory recommendation

We recommend setting

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

before training. This allows JAX to preallocate up to **90%** of the available GPU memory, which helps reduce memory fragmentation and improves training stability.


## Inference Example

You can use the ZMQ-based policy server to communicate with an RBY1 robot client.

For example, when using the RBY1 robot client implemented in `rby1-lerobot`, run the OpenPI policy server as follows:

```bash
uv run src/openpi/serving/zmq_policy_server.py \
  --config-name your_config_name \
  --checkpoint-dir your_checkpoint_dir \
  --host 0.0.0.0 \
  --port 5555
```

After that, configure the RBY1 robot client to connect to the same host and port. The client can then receive action chunks from the OpenPI model and use them for robot control.

For the RBY1 LeRobot client, see:

```text
https://github.com/RainbowRobotics/rby1-lerobot
```

## Notes

* The `config-name` used for training and inference must be an RBY1-specific registered config.
* Run `compute_norm_stats.py` before training to generate normalization statistics for the dataset.
* `checkpoint-dir` must point to the actual trained checkpoint path.
* The ZMQ server `host` and `port` must match the robot client configuration.


# GR00T LEAPP Export Fork

## This project is currently not accepting contributions.

This repository is a fork of the original [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) repository, modified to support LEAPP export workflows.

The two main workflows in this fork are:

- Export a GR00T policy with LEAPP.
- Posttrain GR00T on custom robot data before export or deployment.

For the full upstream model documentation, architecture details, datasets, and benchmark notes, see the original [Isaac GR00T README](https://github.com/NVIDIA/Isaac-GR00T).

## Installation

### Hardware Requirements

Inference and export require a CUDA-capable GPU. Posttraining is more demanding; use one or more GPUs with at least 40 GB VRAM when possible.

This repo follows the upstream GR00T platform split:

- dGPU: CUDA 12.8 with Python 3.10.
- Jetson Orin: CUDA 12.6 with Python 3.10.
- Jetson Thor and DGX Spark: CUDA 13.0 with Python 3.12.

Platform-specific setup scripts and Dockerfiles live under `scripts/deployment/`.

### Clone

GR00T uses git submodules. Clone with submodules enabled:

```bash
git clone --recurse-submodules https://github.com/nvidia-isaac/gr00t-leapp-export
cd gr00t-leapp-export
```

If you already cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

`git-lfs` is required for demo data files:

```bash
sudo apt install git-lfs
git lfs install
```

### Set Up the Environment

Install [uv](https://github.com/astral-sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For the default dGPU setup:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
uv sync --python 3.10
uv run python -c "import gr00t; print('GR00T installed successfully')"
```

For Jetson Thor, Jetson Orin, DGX Spark, or Docker-based setup, use the platform guides under `scripts/deployment/` and `docker/`.

## LEAPP Export

Use `export/export_with_leapp.py` to export a GR00T policy into a LEAPP-generated ONNX package.

```bash
uv run python export/export_with_leapp.py \
    --model_path nvidia/GR00T-N1.7-3B \
    --dataset_path demo_data/droid_sample \
    --embodiment_tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --output_name exported_gr00t
```

For a custom fine-tuned G1 or `NEW_EMBODIMENT` policy, pass a joint-name config so the exported YAML contains real ROS joint names:

```bash
uv run python export/export_with_leapp.py \
    --model_path <path-to-finetuned-checkpoint> \
    --dataset_path <path-to-lerobot-dataset> \
    --embodiment_tag new_embodiment \
    --joint_config <path-to-joint-config.json>\
    --output_name exported_gr00t
```

Choose the G1 joint configuration for the checkpoint's training profile:

| Training profile | Joint configuration |
| --- | --- |
| Physical G1 reference workflow | [`export/data/g1_joints.json`](export/data/g1_joints.json) |
| Arena static-apple simulation policy used by Isaac OS | [`export/data/g1_sim_joints.json`](export/data/g1_sim_joints.json) |

The simulation file matches the active groups in
[`nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace`](https://huggingface.co/nvidia/GN1x-Tuned-Arena-G1-Static-PickNPlace)
and the joint names in its
[training dataset](https://huggingface.co/datasets/nvidia/Arena-G1-Static-PickNPlace-Task).
Its hand-state order is index, middle, thumb; the physical configuration uses a
different hand-state order. Both use index, middle, thumb for hand actions.
These files describe training profiles, not all simulated or physical G1 policies.
For other data, supply a configuration matching its state and action ordering.
The joint JSON is needed for export, not by the Isaac OS app at runtime.

For the Arena-based simulation profile, pass:

```bash
--joint_config export/data/g1_sim_joints.json
```

Validate an export by comparing the original PyTorch policy against the exported policy:

```bash
uv run python export/policy_comparison.py \
    --model_yaml_path exported_gr00t/exported_gr00t.yaml \
    --max_steps 100 \
    --show_plots True
```

See [`export/README.md`](export/README.md) for more export options and task-specific examples.

## Posttrain

Use posttraining to adapt GR00T to your robot, embodiment tag, and dataset before export or deployment.

Your dataset should follow the GR00T LeRobot format and include a `meta/modality.json` file describing state, action, and video fields. See [`getting_started/data_preparation.md`](getting_started/data_preparation.md) and [`getting_started/finetune_new_embodiment.md`](getting_started/finetune_new_embodiment.md) for full details.

Single-GPU example:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --num-gpus 1 \
    --output-dir /tmp/gr00t_posttrain \
    --max-steps 2000 \
    --global-batch-size 32 \
    --dataloader-num-workers 4
```

Wrapper script example:

```bash
bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path demo_data/cube_to_bowl_5 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/SO100/so100_config.py \
    --output-dir /tmp/gr00t_posttrain
```

After posttraining, pass the resulting checkpoint to the LEAPP export command above.

## License

- Code: Apache 2.0, see [`LICENSE`](LICENSE).
- Model weights: [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

## Citation

If you use GR00T in research, cite the upstream NVIDIA Isaac GR00T work:

```bibtex
@inproceedings{gr00tn1_2025,
  archivePrefix = {arxiv},
  eprint = {2503.14734},
  title = {{GR00T} {N1}: An Open Foundation Model for Generalist Humanoid Robots},
  author = {NVIDIA and Johan Bjorck and Fernando Castaneda, Nikita Cherniadev and Xingye Da and Runyu Ding and Linxi "Jim" Fan and Yu Fang and Dieter Fox and Fengyuan Hu and Spencer Huang and Joel Jang and Zhenyu Jiang and Jan Kautz and Kaushil Kundalia and Lawrence Lao and Zhiqi Li and Zongyu Lin and Kevin Lin and Guilin Liu and Edith Llontop and Loic Magne and Ajay Mandlekar and Avnish Narayan and Soroush Nasiriany and Scott Reed and You Liang Tan and Guanzhi Wang and Zu Wang and Jing Wang and Qi Wang and Jiannan Xiang and Yuqi Xie and Yinzhen Xu and Zhenjia Xu and Seonghyeon Ye and Zhiding Yu and Ao Zhang and Hao Zhang and Yizhou Zhao and Ruijie Zheng and Yuke Zhu},
  month = {March},
  year = {2025},
  booktitle = {ArXiv Preprint},
}
```

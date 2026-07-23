# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import os
import gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
from gr00t.data.types import MessageType
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.stats import ensure_stats
import random
import numpy as np


def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility (CPU + CUDA + NumPy)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: for fully deterministic behavior (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compare_tensors(tensor1, tensor2, name1="Tensor1", name2="Tensor2", rtol=1e-2, atol=1e-3):
    """
    Compare two PyTorch tensors and print statistical analysis.

    Args:
        tensor1: First PyTorch tensor
        tensor2: Second PyTorch tensor
        name1: Label for first tensor
        name2: Label for second tensor
        rtol: Relative tolerance for validation
        atol: Absolute tolerance for validation

    Returns:
        bool: True if tensors are close within tolerance
    """
    # Ensure both tensors are on the same device and convert to float32
    device = tensor1.device
    t1 = tensor1.detach().float()
    t2 = tensor2.detach().float().to(device)

    # Compute errors
    errors = torch.abs(t1 - t2)
    max_diff = errors.max().item()
    mean_diff = errors.mean().item()

    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    # Output magnitude analysis
    print("\n  Output Magnitude Analysis:")
    print(f"    {name1} - min: {t1.min().item():.4f}, max: {t1.max().item():.4f}, "
          f"mean: {t1.mean().item():.4f}, std: {t1.std().item():.4f}")
    print(f"    {name2} - min: {t2.min().item():.4f}, max: {t2.max().item():.4f}, "
          f"mean: {t2.mean().item():.4f}, std: {t2.std().item():.4f}")

    # Error percentile analysis
    errors_flat = errors.flatten()
    percentiles = [50, 90, 95, 99, 99.9, 100]
    print("\n  Error Percentile Analysis:")
    for p in percentiles:
        k = int((p / 100.0) * errors_flat.numel())
        k = min(k, errors_flat.numel() - 1)
        val = torch.kthvalue(errors_flat, k + 1).values.item() if k >= 0 else errors_flat.min().item()
        print(f"    {p:>5}th percentile: {val:.6e}")

    # Relative error analysis (avoid division by zero)
    abs_t1 = torch.abs(t1)
    mask = abs_t1 > 1e-6
    if mask.any():
        relative_errors = errors[mask] / abs_t1[mask]
        print(f"\n  Relative Error Analysis (where |{name1}| > 1e-6):")
        print(f"    Max relative error:  {relative_errors.max().item():.6e}")
        print(f"    Mean relative error: {relative_errors.mean().item():.6e}")
        k99 = int(0.99 * relative_errors.numel())
        k99 = min(k99, relative_errors.numel() - 1)
        p99_val = torch.kthvalue(relative_errors.flatten(), k99 + 1).values.item()
        print(f"    99th percentile:     {p99_val:.6e}")

    # Identify worst error location
    worst_flat_idx = torch.argmax(errors).item()
    worst_idx = []
    temp = worst_flat_idx
    for dim in reversed(errors.shape):
        worst_idx.insert(0, temp % dim)
        temp //= dim
    worst_idx = tuple(worst_idx)

    print(f"\n  Worst error location: index {worst_idx}")
    print(f"    {name1} value: {t1[worst_idx].item():.6f}")
    print(f"    {name2} value: {t2[worst_idx].item():.6f}")

    # Check if outputs are close enough
    mean_threshold = 1e-2
    max_threshold = 10
    if mean_diff < mean_threshold and max_diff < max_threshold:
        print(f"\n  ✓ Validation PASSED (mean_diff={mean_diff:.6e} < {mean_threshold}, max_diff={max_diff:.6e} < {max_threshold})")
        return True
    else:
        print(f"\n  ✗ Validation FAILED - outputs differ beyond tolerance")
        print(f"    mean_diff={mean_diff:.6e} (threshold: {mean_threshold}), max_diff={max_diff:.6e} (threshold: {max_threshold})")
        print(f"    {name1} shape: {tuple(t1.shape)}, {name2} shape: {tuple(t2.shape)}")
        return False


def get_policy_and_dataset(
    model_path="nvidia/GR00T-N1.7-3B",
    dataset_path=os.path.join(
        os.path.dirname(os.path.dirname(gr00t.__file__)), "demo_data/droid_sample"
    ),
    embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
    video_backend="torchcodec",
):

    device = "cuda" if torch.cuda.is_available() else "cpu"


    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=EmbodimentTag.resolve(embodiment_tag),
        device=device,
        strict=True,
    )

    modality_config = policy.get_modality_config()

    ensure_stats(dataset_path, embodiment_tag, modality_config=modality_config)

    dataset = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        video_backend=video_backend,
        video_backend_kwargs=None,
    )

    return policy, dataset


def get_prepared(data, policy):
    unbatched_observations = policy._unbatch_observation(data)
    processed_inputs = []
    for obs in unbatched_observations:
        vla_step_data = policy._to_vla_step_data(obs)
        messages = [
            {"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed_inputs.append(policy.processor(messages))
    return processed_inputs

def get_preprocessed(observation, policy):
    collated_inputs = policy.collate_fn(observation)
    collated_inputs = _rec_to_dtype(
        collated_inputs, dtype=torch.bfloat16)

    return list(collated_inputs.values())[0]

def get_gr00t_action_head_input(observation, policy):
    backbone_inputs, action_inputs = policy.model.prepare_input(observation)
    backbone_outputs = policy.model.backbone(backbone_inputs)

    return backbone_outputs, action_inputs

def get_gr00t_input(dataset, policy, step_index = None, step = None):
    modality_config = policy.get_modality_config()
    embodiment_tag = policy.embodiment_tag  # Get from policy directly

    if step_index is None:
        step_index = random.randint(0, len(dataset) - 1)

    episode_data = dataset[0]
    step_data = extract_step_data(
        episode_data, step_index=step_index, modality_configs=modality_config, embodiment_tag=embodiment_tag, allow_padding=False
    )
    data = {
        "video": {k: np.stack(step_data.images[k])[None] for k in step_data.images},  # stach images and add batch dimension
        "state": {k: step_data.states[k][None] for k in step_data.states},  # add batch dimension
        "action": {k: step_data.actions[k][None] for k in step_data.actions},  # add batch dimension
        "language": {
            modality_config["language"].modality_keys[0]: [[step_data.text]],  # add time and batch dimension
        }
    }

    step_dict = {
        None : -1,
        'prepared' : 0,
        'preprocessed' : 1,
        'gr00t_action_head_input' : 2,
    }

    if step_dict[step] >= step_dict['prepared']:
        data = get_prepared(data, policy)
    if step_dict[step] >= step_dict['preprocessed']:
        data = get_preprocessed(data, policy)
    if step_dict[step] >= step_dict['gr00t_action_head_input']:
        data = get_gr00t_action_head_input(data, policy)
    return data

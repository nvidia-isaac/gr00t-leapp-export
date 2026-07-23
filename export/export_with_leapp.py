# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GR00T model export using leapp framework.

This module handles exporting the GR00T policy to ONNX format using the leapp
tracing and export framework.
"""

from utils import get_policy_and_dataset, get_gr00t_input
from policy_modifications import make_modifications, get_action_traceable
from cuda_capturable import make_cuda_capturable
from joint_name_parser import EMBODIMENT_JOINT_REGISTRY, register_embodiment_joints
import os
import json
import gr00t
import leapp
from leapp import annotate

import argparse

args = argparse.ArgumentParser()
args.add_argument("--model_path", type=str, default='nvidia/GR00T-N1.7-3B')
args.add_argument("--dataset_path", type=str, default=os.path.join(os.path.dirname(os.path.dirname(gr00t.__file__)), "demo_data/droid_sample"))
args.add_argument("--embodiment_tag", type=str, default='OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT')
args.add_argument("--video_backend", type=str, default='torchcodec')
args.add_argument("--output_name", type=str, default='exported_gr00t')
args.add_argument("--joint_config", type=str, default=None,
                  help="Path to a JSON file with joint names: {state: {group: [names]}, action: {group: [names]}}. "
                       "Auto-detected from dataset modality.json if not provided.")
args.add_argument("--cuda-capturable", dest="cuda_capturable", action="store_true",
                  help="Post-process the exported backbone.onnx into a CUDA-graph-capturable form by "
                       "baking its data-dependent NonZero (masked_scatter) outputs in as constants. "
                       "Requires a fixed-prompt export (prompt baked into preprocess_video).")
args = args.parse_args()


def _auto_joint_config_from_modality(dataset_path: str) -> dict:
    """Build a joint config from the dataset's modality.json using placeholder names."""
    modality_path = os.path.join(dataset_path, "meta", "modality.json")
    with open(modality_path) as f:
        modality = json.load(f)

    joints = {}
    for modality_type in ("state", "action"):
        if modality_type not in modality:
            continue
        joints[modality_type] = {
            group: [f"{group}_{i}" for i in range(spec["end"] - spec["start"])]
            for group, spec in modality[modality_type].items()
        }
    return joints


def maybe_register_embodiment_joints(embodiment_tag: str, dataset_path: str, joint_config_path: str = None):
    """Register joint names for embodiment_tag if not already in the registry.

    Uses joint_config_path if provided, otherwise auto-detects from dataset modality.json.
    No-op if the embodiment is already registered.
    """
    tag = embodiment_tag.lower()
    if tag in EMBODIMENT_JOINT_REGISTRY:
        return

    if joint_config_path is not None:
        with open(joint_config_path) as f:
            joints = json.load(f)
        print(f"Registering joint names for '{tag}' from {joint_config_path}")
    else:
        joints = _auto_joint_config_from_modality(dataset_path)
        print(f"Auto-registering joint names for '{tag}' from {dataset_path}/meta/modality.json")

    register_embodiment_joints(tag, joints)


def _split_leapp_output_path(output_name: str) -> tuple[str, str]:
    """Split a user output path into LEAPP's save_path and graph name arguments."""
    normalized_output = os.path.normpath(os.path.expanduser(output_name))
    save_path, graph_name = os.path.split(normalized_output)
    if not graph_name:
        raise ValueError(f"--output_name must include a directory or model name, got: {output_name}")
    return save_path or ".", graph_name


def export_gr00t_with_leapp(policy, data, output_name='exported_gr00t', cuda_capturable=False):
    """
    Export GR00T policy using leapp framework.

    Args:
        policy: Gr00tPolicy instance (will be modified in-place)
        data: Sample input data for tracing
        output_name: Name for the exported model
        cuda_capturable: If True, rewrite the exported backbone.onnx into a
            CUDA-graph-capturable form (bakes its NonZero outputs in as constants).
    """
    # Apply modifications to make policy traceable
    policy = make_modifications(policy)

    # Configure backbone export
    policy.model.backbone.forward = annotate._method(
        node_name='backbone',
        export_with='onnx-torchscript',
    )(policy.model.backbone.forward)

    # Configure action head export
    policy.model.action_head.get_action = annotate._method(
        node_name='action_head',
        export_with='onnx',
    )(policy.model.action_head.get_action)

    save_path, graph_name = _split_leapp_output_path(output_name)

    # Run tracing
    leapp.start(graph_name, save_path=save_path, global_patching=False, dry_run=False)
    get_action_traceable(policy, data)
    leapp.stop()

    # Compile and export
    leapp.compile_graph(validate=False) # validate with comparison script

    if cuda_capturable:
        make_cuda_capturable(os.path.join(save_path, graph_name))

    print(f"Export completed: {os.path.join(save_path, graph_name)}")


def main():
    """Main entry point for export."""
    maybe_register_embodiment_joints(args.embodiment_tag, args.dataset_path, args.joint_config)

    # Load policy and dataset
    policy, dataset = get_policy_and_dataset(model_path = args.model_path,
                                            dataset_path = args.dataset_path,
                                            embodiment_tag = args.embodiment_tag, video_backend = args.video_backend)

    # Get sample input data
    data = get_gr00t_input(dataset, policy, step_index=0, step=None)
    # Export
    export_gr00t_with_leapp(policy, data, output_name=args.output_name,
                            cuda_capturable=args.cuda_capturable)


if __name__ == "__main__":
    main()

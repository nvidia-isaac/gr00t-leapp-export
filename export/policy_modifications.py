# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Policy modifications for making GR00T traceable/exportable.

This module contains the modifications needed to convert the GR00T policy
to use PyTorch-traceable operations for ONNX export.
"""

import torch
import types

from data.nn_modifications import get_modified_vision_model, get_modified_language_model
from data.state_action_processor_torch import StateActionProcessorTorch
from data.processor_torch import create_torch_processor
from data.collator_torch import create_torch_collator

from gr00t.policy.gr00t_policy import _rec_to_dtype
from gr00t.data.types import MessageType

from leapp import annotate
from leapp import TensorSemantics
from leapp import InputKindEnum, OutputKindEnum
from joint_name_parser import get_joint_names


def _backbone_forward_with_int32(self, vl_input):
    """
    Backbone forward that converts boolean outputs to int32 for TensorRT compatibility.
    
    Note: TensorRT doesn't support int8 as a general data type (only for INT8 quantization),
    so we use int32 instead which has full TensorRT support.
    
    Calls self._original_forward which is the original forward method.
    """
    if "image_grid_thw" not in vl_input and hasattr(self, "_fixed_image_grid_thw"):
        vl_input = dict(vl_input)
        vl_input["image_grid_thw"] = self._fixed_image_grid_thw.to(vl_input["pixel_values"].device)

    outputs = self._original_forward(vl_input)
    # Convert boolean tensors to int32
    converted_outputs = {}
    for key, value in outputs.items():
        if torch.is_tensor(value) and value.dtype == torch.bool:
            converted_outputs[key] = value.to(torch.int32)
        else:
            converted_outputs[key] = value
    return converted_outputs


def _action_head_get_action_with_bool(self, backbone_outputs, action_inputs, options=None, initial_noise=None):
    """
    Action head get_action that converts int32 mask inputs back to bool.
    Calls self._original_get_action which is the original get_action method.
    """
    # Convert int32 tensors to bool inline (no external function calls for leapp compatibility)
    converted_backbone_outputs = {}
    for k, v in backbone_outputs.items():
        if torch.is_tensor(v) and v.dtype == torch.int32:
            converted_backbone_outputs[k] = v.to(torch.bool)
        else:
            converted_backbone_outputs[k] = v
    
    return self._original_get_action(
        converted_backbone_outputs,
        action_inputs,
        options=options,
        initial_noise=initial_noise,
    )


def _action_head_prepare_input_for_export(self, batch):
    """Keep only inference-time action-head inputs at the LEAPP graph boundary."""
    keys_to_use = ("state", "embodiment_id")
    return self._original_prepare_input({k: batch[k] for k in keys_to_use if k in batch})


def get_action_traceable(self, data, initial_noise=None):
    """
    Torch-traceable version of get_action for export.
    
    This replaces the original get_action method with one that uses
    PyTorch operations instead of PIL/numpy for image processing.
    
    Args:
        data: Input observation data
        initial_noise: Optional initial noise tensor for diffusion.
            Shape: [B, action_horizon, action_dim]. If None, noise is generated internally.
    """
    
    # Step 1: Split batched observation into individual observations
    unbatched_observations = self._unbatch_observation(data)
    processed_inputs = []
    
    # Convert to torch tensors
    for i in range(len(unbatched_observations)):
        for k, v in unbatched_observations[i]["state"].items():
            unbatched_observations[i]["state"][k] = torch.from_numpy(v)
        for k, v in unbatched_observations[i]["video"].items():
            unbatched_observations[i]["video"][k] = torch.from_numpy(v).to(torch.float32)

    # Annotate inputs for export tracing
    for i in range(len(unbatched_observations)):
        for k, v in unbatched_observations[i]["state"].items():
            # v = TensorSemantics(v, )
            element_names = get_joint_names(self.embodiment_tag, "state", k)
            unbatched_observations[i]["state"][k] = annotate.input_tensors(
                'preprocess_state', TensorSemantics(name=k, ref = v, 
                                                    kind = InputKindEnum.JOINT_POSITION, 
                                                    element_names = element_names)
            )
        for k, v in unbatched_observations[i]["video"].items():

            unbatched_observations[i]["video"][k] = annotate.input_tensors(
                'preprocess_video', TensorSemantics(name=k, ref = v,
                                                    kind = "state/camera/image")
            )

    # Step 2: Process each observation through the VLA processor
    states = []
    for obs in unbatched_observations:
        vla_step_data = self._to_vla_step_data(obs)
        states.append(vla_step_data.states.copy())
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed_input = self.processor(messages)

        # Video processing with torch-traceable pipeline
        torch_result = self.torch_processor.process_vlm_inputs_torch(
            vla_step_data, self.embodiment_tag
        )
        processed_input['vlm_content']['images'] = torch_result['images']
        processed_input['vlm_content']['conversation'] = torch_result['conversation']
        processed_inputs.append(processed_input)

    # Step 3: Collate processed inputs into a single batch for model
    collated_inputs = self.collate_fn(processed_inputs)
    collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.float32)
    
    # Keep image_grid_thw internal to the backbone export so fixed deployments
    # do not expose it as a runtime pipeline value.
    self.model.backbone._fixed_image_grid_thw = collated_inputs['inputs'].pop('image_grid_thw')

    # Annotate outputs for export
    static_outputs = {
        'input_ids': collated_inputs['inputs']['input_ids'],
        'attention_mask': collated_inputs['inputs']['attention_mask'],
        'embodiment_id': collated_inputs['inputs']['embodiment_id']
    }
    annotate.output_tensors(
        'preprocess_video',
        {'pixel_values': collated_inputs['inputs']['pixel_values']},
        static_outputs=static_outputs,
        export_with='onnx'
    )

    annotate.output_tensors(
        'preprocess_state',
        {'state': collated_inputs['inputs']['state'],
        'reference': states},
        export_with='onnx-torchscript'
    )

    # Step 4: Run model inference to predict actions
    # Use provided initial_noise, or generate one (baked in as constant during export)
    if initial_noise is None:
        batch_size = 1
        action_horizon = self.model.action_head.config.action_horizon
        action_dim = self.model.action_head.action_dim
        device = collated_inputs['inputs']['state'].device
        dtype = collated_inputs['inputs']['state'].dtype
        
        initial_noise = torch.randn(
            batch_size, action_horizon, action_dim,
            device=device, dtype=dtype
        )
    
    with torch.inference_mode():
        model_pred = self.model.get_action(**collated_inputs, initial_noise=initial_noise)
    normalized_action = model_pred["action_pred"].float()

    normalized_action, states = annotate.input_tensors('decode_action', 
        {'normalized_action': normalized_action, 'state': states},
    )

    # Step 5: Decode actions from normalized space back to physical units
    batched_states = {}
    for k in self.modality_configs["state"].modality_keys:
        batched_states[k] = torch.stack(
            [s[k] for s in states], dim=0
        ).to(normalized_action.device)
    
    unnormalized_action = self.processor.decode_action(
        normalized_action, self.embodiment_tag, batched_states
    )

    # Cast all actions to float32 for consistency.  Pick the right `kind` per output:
    #   - effort_* keys     -> target/joint/effort
    #   - navigate_command  -> velocity_command (downstream convention)
    #   - base_height_command -> no kind (scalar standing-height target)
    #   - everything else   -> target/joint/position (arms/hands/waist)
    def _kind_for(output_key: str):
        if output_key.startswith("effort_"):
            return OutputKindEnum.JOINT_EFFORT
        if output_key == "navigate_command":
            return "velocity_command"
        if output_key == "base_height_command":
            return None
        return OutputKindEnum.JOINT_POSITION

    casted_action = [
        TensorSemantics(
            name=key,
            ref=value.to(torch.float32),
            kind=_kind_for(key),
            element_names=get_joint_names(self.embodiment_tag, "action", key),
        )
        for key, value in unnormalized_action.items()
    ]


    annotate.output_tensors('decode_action', casted_action,
                            export_with='onnx')

    return casted_action, {}


def make_modifications(policy):
    """
    Apply all modifications to make the policy torch-traceable for export.
    
    This modifies the policy in-place to:
    1. Replace vision/language models with export-compatible versions
    2. Replace state/action processor with torch-traceable version
    3. Replace collator with torch-traceable version
    4. Replace get_action method with traceable version
    
    Args:
        policy: Gr00tPolicy instance to modify
        
    Returns:
        Modified policy
    """
    # ==================== Backbone modifications ====================
    # N1.7 uses Qwen3VLForConditionalGeneration:
    #   policy.model.backbone.model.model.visual
    #   policy.model.backbone.model.model.language_model
    # Patch those live modules in place instead of replacing the removed N1.6
    # `vision_model` / `language_model` attributes.
    qwen_model = policy.model.backbone.model
    if hasattr(qwen_model, "model") and hasattr(qwen_model.model, "visual"):
        get_modified_vision_model(qwen_model.model.visual)
        get_modified_language_model(qwen_model.model.language_model)
    else:
        vision_model_module = get_modified_vision_model(qwen_model.vision_model)
        language_model_module = get_modified_language_model(qwen_model.language_model)
        qwen_model.vision_model = vision_model_module
        qwen_model.language_model = language_model_module
    qwen_model.eval()

    # Use float32 (half precision causes significant errors)
    policy.model.backbone = policy.model.backbone.float()
    
    # Disable gradients for export
    policy.model.backbone.requires_grad_(False)

    # Store original forward and replace with int32 conversion wrapper
    policy.model.backbone._original_forward = policy.model.backbone.forward
    policy.model.backbone.forward = types.MethodType(_backbone_forward_with_int32, policy.model.backbone)
    
    # ==================== Action head modifications ====================
    policy.model.action_head = policy.model.action_head.float()

    # Store original get_action and replace with bool conversion wrapper
    policy.model.action_head._original_get_action = policy.model.action_head.get_action
    policy.model.action_head.get_action = types.MethodType(_action_head_get_action_with_bool, policy.model.action_head)

    policy.model.action_head._original_prepare_input = policy.model.action_head.prepare_input
    policy.model.action_head.prepare_input = types.MethodType(_action_head_prepare_input_for_export, policy.model.action_head)

    # ==================== Preprocessing modifications ====================
    # Replace state/action processor with torch-traceable version
    policy.processor.state_action_processor = StateActionProcessorTorch(
        policy.processor.state_action_processor
    )
    
    # Create torch processor for VLM input processing
    policy.torch_processor = create_torch_processor(policy.processor)

    # ==================== Collator modifications ====================
    policy.collate_fn = create_torch_collator(
        model_name=policy.model.config.model_name,
        model_type=policy.model.config.backbone_model_type,
        transformers_loading_kwargs={"trust_remote_code": True},
    )

    # ==================== Replace get_action method ====================
    policy.get_action = types.MethodType(get_action_traceable, policy)

    return policy


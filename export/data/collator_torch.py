# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Collator shim for exporting GR00T.

N1.7 uses Qwen3-VL, whose processor returns `pixel_values` and `image_grid_thw`.
The older export path used Eagle-style tensors, so this shim keeps the tensor-image
handoff from the leapp preprocessing path but delegates final VLM packing to the
current Qwen3-VL processor.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import resize as tv_resize
from typing import Any, Dict, List, Literal
from transformers import BatchFeature

from .image_processing_torch import smart_resize


def build_processor(model_name: str, transformers_loading_kwargs: dict):
    """Build the VLM processor (same as original)."""
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import build_processor as _build_processor

    return _build_processor(model_name, transformers_loading_kwargs)


class Gr00tN1d6DataCollatorTorch:
    """
    PyTorch-traceable data collator for GR00T N1D6.
    
    Full replacement for Gr00tN1d6DataCollator that:
    1. Takes images as torch tensors (H, W, C) uint8
    2. Processes images using pure PyTorch operations
    3. Gets tokenization from the underlying VLM processor
    4. Returns same BatchFeature format as original
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: Literal["qwen"] = "qwen",
        transformers_loading_kwargs: dict = {},
    ):
        """
        Args:
            model_name: HuggingFace model name for VLM processor
            model_type: Type of VLM backbone ("eagle")
            transformers_loading_kwargs: Kwargs for loading transformers
        """
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        self.processor.tokenizer.padding_side = "left"
        self.model_type = model_type
        self.model_name = model_name
        
    def extract_images_from_conversation(
        self,
        conversation: List[Dict],
    ) -> List[Tensor]:
        """
        Extract image tensors from conversation structure.
        
        Args:
            conversation: List of message dicts with 'content' containing image items
            
        Returns:
            List of image tensors
        """
        images = []
        for message in conversation:
            if not isinstance(message, dict):
                continue
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    img = item["image"]
                    images.append(img)
        return images

    def _preprocess_qwen_image_torch(self, image: Tensor) -> tuple[Tensor, tuple[int, int, int]]:
        image_processor = self.processor.image_processor
        patch_size = image_processor.patch_size
        temporal_patch_size = image_processor.temporal_patch_size
        merge_size = image_processor.merge_size

        if image.ndim == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)
        if image.ndim != 3:
            raise ValueError(f"Expected image tensor with shape HWC or CHW, got {tuple(image.shape)}")

        height, width = int(image.shape[-2]), int(image.shape[-1])
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=image_processor.size["shortest_edge"],
            max_pixels=image_processor.size["longest_edge"],
        )

        # Use torchvision v2 resize with PIL-compatible bicubic kernel
        # (closer to HF Qwen3-VL processor's PIL.Image.resize than F.interpolate).
        image = tv_resize(
            image.unsqueeze(0).float(),
            size=[resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).squeeze(0)
        image = image * image_processor.rescale_factor
        mean = torch.tensor(image_processor.image_mean, dtype=image.dtype, device=image.device)[:, None, None]
        std = torch.tensor(image_processor.image_std, dtype=image.dtype, device=image.device)[:, None, None]
        image = (image - mean) / std

        patches = image.unsqueeze(0)
        if patches.shape[0] % temporal_patch_size != 0:
            repeats = temporal_patch_size - (patches.shape[0] % temporal_patch_size)
            patches = torch.cat([patches, patches[-1:].repeat(repeats, 1, 1, 1)], dim=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size
        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        return flatten_patches, (grid_t, grid_h, grid_w)

    def _process_qwen_images_torch(self, images: List[Tensor]) -> tuple[Tensor, torch.Tensor]:
        pixel_values = []
        image_grid_thw = []
        for image in images:
            patches, grid_thw = self._preprocess_qwen_image_torch(image)
            pixel_values.append(patches)
            image_grid_thw.append(grid_thw)
        return torch.cat(pixel_values, dim=0), torch.tensor(image_grid_thw, dtype=torch.long)
    
    def __call__(self, features: List[Dict[str, Any]]) -> BatchFeature:
        """
        Collate features into a batch.
        
        Args:
            features: List of processed inputs from processor.
                Each contains: vlm_content, state, embodiment_id, etc.
                Images in vlm_content should be torch tensors (H, W, C) uint8.
                
        Returns:
            BatchFeature with:
                - input_ids: tokenized text
                - attention_mask: attention mask for tokens
                - pixel_values: list of processed image tensors
                - image_sizes: tensor of (H, W) per image
                - state, embodiment_id, etc.
        """
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))
        
        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            
            if key == "vlm_content":
                # Handle vlm_content: extract text for tokenization, images for processing
                text_list = []
                all_images = []
                
                for v in values:
                    text_list.append(v["text"])
                    
                    # Extract images from conversation
                    conversation = v.get("conversation", [])
                    images = self.extract_images_from_conversation(conversation)
                    all_images.extend(images)
                
                pixel_values = None
                image_grid_thw = None
                pil_images = []
                for img in all_images:
                    if isinstance(img, Image.Image):
                        pil_images.append(img)
                    elif torch.is_tensor(img):
                        patches, grid = self._preprocess_qwen_image_torch(img)
                        pixel_values = patches if pixel_values is None else torch.cat([pixel_values, patches], dim=0)
                        grid_tensor = torch.tensor([grid], dtype=torch.long)
                        image_grid_thw = (
                            grid_tensor
                            if image_grid_thw is None
                            else torch.cat([image_grid_thw, grid_tensor], dim=0)
                        )
                        height, width = int(img.shape[0]), int(img.shape[1])
                        pil_images.append(Image.new("RGB", (width, height), color=(0, 0, 0)))
                    elif isinstance(img, np.ndarray):
                        pil_images.append(Image.fromarray(img))
                    else:
                        raise TypeError(f"Unsupported image type for Qwen3-VL processor: {type(img)}")

                if pil_images:
                    vlm_inputs = self.processor(
                        text=text_list,
                        images=pil_images,
                        return_tensors="pt",
                        padding=True,
                    )
                else:
                    vlm_inputs = self.processor(text=text_list, return_tensors="pt", padding=True)

                for k, v in vlm_inputs.items():
                    batch[k] = v
                if pixel_values is not None:
                    batch["pixel_values"] = pixel_values
                    batch["image_grid_thw"] = image_grid_thw
                        
            elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
                raise Exception("Not implemented")
            else:
                if isinstance(values[0], np.ndarray) or isinstance(values[0], int):
                # state, state_mask, action and action_mask - stack to form batch dimension
                    batch[key] = torch.from_numpy(np.stack(values))
                else:
                    batch[key] = torch.stack(values)
        
        return {"inputs": batch}
    
    def __str__(self):
        return f"Gr00tN1d6DataCollatorTorch(model_name={self.model_name}, model_type={self.model_type})"


def create_torch_collator(
    model_name: str,
    model_type: Literal["qwen"] = "qwen",
    transformers_loading_kwargs: dict = {},
) -> Gr00tN1d6DataCollatorTorch:
    """
    Factory function to create a PyTorch-traceable collator.
    
    Args:
        model_name: HuggingFace model name
        model_type: VLM backbone type
        transformers_loading_kwargs: Kwargs for transformers loading
        
    Returns:
        Gr00tN1d6DataCollatorTorch instance
    """
    return Gr00tN1d6DataCollatorTorch(
        model_name=model_name,
        model_type=model_type,
        transformers_loading_kwargs=transformers_loading_kwargs,
    )

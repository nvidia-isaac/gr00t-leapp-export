# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch-traceable processor wrapper for GR00T N1D6.

Replaces PIL/albumentations-based image transforms with pure PyTorch operations
for ONNX export compatibility.
"""

import re
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import resize as tv_resize
from typing import Any, Dict, List


class Gr00tN1d6ProcessorTorch:
    """
    Torch-traceable wrapper for Gr00tN1d6Processor.
    
    Replaces the image transform pipeline with PyTorch operations while
    keeping the rest of the processor logic intact.
    """
    
    def __init__(
        self,
        original_processor,
    ):
        """
        Args:
            original_processor: The original Gr00tN1d6Processor instance
        """
        self.original_processor = original_processor
        
        # Copy attributes from original processor for compatibility
        self.modality_configs = original_processor.modality_configs
        self.formalize_language = original_processor.formalize_language
        self.embodiment_id_mapping = original_processor.embodiment_id_mapping
        self.training = original_processor.training
        
        # Get resize parameters from original processor
        self.use_albumentations = original_processor.use_albumentations
        self.image_target_size = original_processor.image_target_size
        self.image_crop_size = original_processor.image_crop_size
        self.shortest_image_edge = original_processor.shortest_image_edge
        self.crop_fraction = original_processor.crop_fraction
    
    def letterbox_transform(self, image: Tensor) -> Tensor:
        """
        Pad image to square by adding black bars.
        
        Args:
            image: (C, H, W) tensor
            
        Returns:
            (C, max(H,W), max(H,W)) tensor
        """
        c, h, w = image.shape
        
        if h == w:
            return image
        
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # F.pad expects (left, right, top, bottom) for 2D padding on last 2 dims
        padded = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        return padded
    
    def center_crop(self, image: Tensor, crop_height: int, crop_width: int) -> Tensor:
        """
        Center crop the image.
        
        Args:
            image: (C, H, W) tensor
            crop_height: Target crop height
            crop_width: Target crop width
            
        Returns:
            (C, crop_height, crop_width) tensor
        """
        c, h, w = image.shape
        
        start_h = (h - crop_height) // 2
        start_w = (w - crop_width) // 2
        
        return image[:, start_h:start_h + crop_height, start_w:start_w + crop_width]
        
    def smallest_max_size_resize(self, image: Tensor, max_size: int) -> Tensor:
        """
        Resize so the smallest edge equals max_size, preserving aspect ratio.
        
        Args:
            image: (C, H, W) tensor
            max_size: Target size for the smallest edge
            
        Returns:
            Resized tensor
        """
        c, h, w = image.shape
        
        if h <= w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)
        
        image = image.unsqueeze(0).to(torch.float32)
        # Use 'area' to match cv2.INTER_AREA used by the original albumentations
        # eval pipeline (A.SmallestMaxSize(..., interpolation=cv2.INTER_AREA)).
        image = F.interpolate(
            image,
            size=(new_h, new_w),
            mode='area',
        )
        image = image.round().clamp(0, 255)
        return image.squeeze(0)
    
    def fractional_center_crop(self, image: Tensor, crop_fraction: float) -> Tensor:
        """
        Center crop by a fraction of the image size.
        
        Args:
            image: (C, H, W) tensor
            crop_fraction: Fraction of the image to keep (e.g., 0.95)
            
        Returns:
            Cropped tensor
        """
        c, h, w = image.shape
        
        crop_h = int(h * crop_fraction)
        crop_w = int(w * crop_fraction)
        
        return self.center_crop(image, crop_h, crop_w)
        
    def preprocess_image_torch(self, image: Tensor) -> Tensor:
        """
        Preprocess a single image using PyTorch operations.
        
        Matches the eval transform pipeline based on use_albumentations flag:
        
        If use_albumentations=True:
            1. SmallestMaxSize to shortest_image_edge
            2. FractionalCenterCrop by crop_fraction
            3. SmallestMaxSize to shortest_image_edge
            
        If use_albumentations=False:
            1. LetterBox - pad to square
            2. Resize to image_target_size
            3. CenterCrop to image_crop_size  
            4. Resize to image_target_size
        
        Args:
            image: Input tensor (H, W, C) or (C, H, W) as uint8 [0, 255]
            
        Returns:
            Processed tensor (C, H, W) as uint8 [0, 255]
        """
        # Ensure (C, H, W) format
        if image.ndim == 3 and image.shape[-1] == 3:
            # (H, W, C) -> (C, H, W)
            image = image.permute(2, 0, 1)
        
        if self.use_albumentations:
            # Albumentations pipeline (matches eval_transform in image_augmentations.py):
            #   LetterBoxPad -> SmallestMaxSize -> FractionalCenterCrop -> SmallestMaxSize
            max_size = self.shortest_image_edge
            crop_fraction = self.crop_fraction

            # Step 0: LetterBoxPad - pad non-square images to square with black bars
            image = self.letterbox_transform(image)

            # Step 1: SmallestMaxSize
            image = self.smallest_max_size_resize(image, max_size)

            # Step 2: FractionalCenterCrop
            image = self.fractional_center_crop(image, crop_fraction)

            # Step 3: SmallestMaxSize again
            image = self.smallest_max_size_resize(image, max_size)
            
        else:
            # Torchvision pipeline: LetterBox -> Resize -> CenterCrop -> Resize
            
            # Step 1: LetterBox - pad to square
            image = self.letterbox_transform(image)
            
            # Add batch dim for interpolate
            image = image.unsqueeze(0).to(torch.float32)
            
            # Step 2: Resize to image_target_size
            # Use torchvision v2 resize with PIL-compatible bicubic kernel
            # (closer to original PIL resize than F.interpolate bicubic).
            target_h, target_w = self.image_target_size
            image = tv_resize(
                image,
                size=[target_h, target_w],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            image = image.round().clamp(0, 255)

            # Remove batch dim for center crop
            image = image.squeeze(0)

            # Step 3: CenterCrop to image_crop_size
            crop_h, crop_w = self.image_crop_size
            image = self.center_crop(image, crop_h, crop_w)

            # Step 4: Resize back to image_target_size
            image = image.unsqueeze(0)
            image = tv_resize(
                image,
                size=[target_h, target_w],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            image = image.round().clamp(0, 255)
            
            # Remove batch dim
            image = image.squeeze(0)
        
        # Convert to uint8
        image = image.to(torch.uint8)
        
        return image
    
    def _get_vlm_inputs_torch(
        self,
        image_keys: List[str],
        images: Dict[str, List[Tensor]],
        language: str,
    ) -> Dict[str, Any]:
        """
        Torch-traceable version of _get_vlm_inputs.
        
        Returns torch tensors directly instead of PIL images.
        
        Args:
            image_keys: List of image view keys (e.g., ['left_camera', 'right_camera'])
            images: Dict mapping view keys to lists of image tensors
                Each tensor is (H, W, C) or (C, H, W) as uint8
            language: Language instruction string
            
        Returns:
            Dict with:
                - 'images': List of tensors (H, W, C) uint8 (matches PIL format)
                - 'conversation': Conversation structure with tensor images (H, W, C)
        """
        temporal_stacked_images = {}
        
        for view in image_keys:
            assert view in images, f"{view} not in {images.keys()}"
            
            processed_frames = []
            for img in images[view]:
                processed = self.preprocess_image_torch(img)
                processed_frames.append(processed)
            
            # Stack frames: (T, C, H, W)
            temporal_stacked_images[view] = torch.stack(processed_frames)
        
        # Validate outputs
        for k, v in temporal_stacked_images.items():
            assert v.ndim == 4, f"{k} is not a 4D tensor, got shape {v.shape}"
            assert v.dtype == torch.uint8, f"{k} is not uint8, got {v.dtype}"
            assert v.shape[1] == 3, f"{k} does not have 3 channels, got {v.shape[1]}"
        
        # Stack across views and flatten: (T*V, C, H, W)
        stacked_images = torch.stack(
            [temporal_stacked_images[view] for view in image_keys], dim=1
        ).flatten(0, 1)
        
        # Convert to list of tensors (H, W, C) to match PIL format
        image_list = [stacked_images[i].permute(1, 2, 0) for i in range(stacked_images.shape[0])]
        
        # Build conversation structure with tensor images
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": language},
                    *[{"type": "image", "image": img} for img in image_list],
                ],
            }
        ]
        
        return {
            "images": image_list,
            "conversation": conversation,
        }
    
    def process_vlm_inputs_torch(
        self,
        content,
        embodiment_tag,
    ) -> Dict[str, Any]:
        """
        Torch-traceable version of the VLM input processing section.
        
        Returns torch tensors directly instead of PIL images.
        
        Args:
            content: The content object with .images, .text attributes
            embodiment_tag: The embodiment tag
            
        Returns:
            Dict with:
                - 'images': List of tensors (H, W, C) uint8 (matches PIL format)
                - 'conversation': Conversation structure with tensor images (H, W, C)
        """
        image_keys = self.modality_configs[embodiment_tag.value]["video"].modality_keys
        
        # Language processing
        if self.formalize_language:
            language = content.text.lower()
            language = re.sub(r"[^\w\s]", "", language)
        else:
            language = content.text
        
        # Get VLM inputs using torch pipeline
        vlm_inputs = self._get_vlm_inputs_torch(
            image_keys=image_keys,
            images=content.images,
            language=language,
        )
        
        return vlm_inputs


def create_torch_processor(original_processor) -> Gr00tN1d6ProcessorTorch:
    """
    Factory function to create a torch-traceable processor wrapper.
    
    Args:
        original_processor: The original Gr00tN1d6Processor
        
    Returns:
        Gr00tN1d6ProcessorTorch instance
    """
    return Gr00tN1d6ProcessorTorch(original_processor=original_processor)


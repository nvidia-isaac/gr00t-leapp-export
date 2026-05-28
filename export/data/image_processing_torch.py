# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch-traceable image transforms for ONNX export.

Replaces PIL-based operations in:
- process_vision_info / fetch_image (smart_resize + resize)
- Eagle3_VLImageProcessorFast (rescale + normalize)
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import resize as tv_resize
from typing import Tuple

# Constants from Eagle processor
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 4096 * 28 * 28
MAX_RATIO = 200
IMAGE_MAX_SIZE = 500 * 14  # 7000

# Normalization constants (IMAGENET_STANDARD_MEAN/STD)
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5


def adjust_by_factor(number: int, factor: int, method: str = 'round') -> int:
    """Adjusts 'number' to the nearest, ceiling, or floor multiple of 'factor'."""
    if method == 'round':
        return round(number / factor) * factor
    elif method == 'ceil':
        return ((number + factor - 1) // factor) * factor
    elif method == 'floor':
        return (number // factor) * factor
    else:
        raise ValueError(f"Unknown method: {method}")


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """
    Calculate target dimensions for resizing (pure Python, not traced).
    
    Ensures:
    1. Both dimensions are divisible by 'factor'
    2. Total pixels within [min_pixels, max_pixels]
    3. Aspect ratio preserved as closely as possible
    """
    import math
    
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"Aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )

    h_bar = min(max(factor, adjust_by_factor(height, factor, 'round')), IMAGE_MAX_SIZE)
    w_bar = min(max(factor, adjust_by_factor(width, factor, 'round')), IMAGE_MAX_SIZE)
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h_bar * w_bar) / max_pixels)
        h_bar = adjust_by_factor(int(h_bar / beta), factor, 'floor')
        w_bar = adjust_by_factor(int(w_bar / beta), factor, 'floor')
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = adjust_by_factor(int(height * beta), factor, 'ceil')
        w_bar = adjust_by_factor(int(width * beta), factor, 'ceil')

    return h_bar, w_bar


def resize_image_torch(
    image: Tensor,
    target_height: int,
    target_width: int,
    mode: str = 'bicubic',
    align_corners: bool = False,
) -> Tensor:
    """
    Resize image tensor using PyTorch interpolate.
    
    Args:
        image: Input tensor of shape (C, H, W) or (N, C, H, W)
        target_height: Target height
        target_width: Target width
        mode: Interpolation mode ('bilinear', 'bicubic', 'nearest')
        align_corners: Whether to align corners (only for bilinear/bicubic)
        
    Returns:
        Resized tensor
    """
    # Add batch dimension if needed
    squeeze_batch = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze_batch = True
    
    # Interpolate.
    # For bilinear/bicubic we prefer torchvision v2's resize because its
    # antialiased kernels are closer to PIL.Image.resize than F.interpolate.
    if mode == 'bicubic':
        resized = tv_resize(
            image.float(),
            size=[target_height, target_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    elif mode == 'bilinear':
        resized = tv_resize(
            image.float(),
            size=[target_height, target_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    else:
        resized = F.interpolate(
            image.float(),
            size=(target_height, target_width),
            mode=mode,
        )
    
    if squeeze_batch:
        resized = resized.squeeze(0)
    
    return resized


def rescale_image(image: Tensor, scale_factor: float = 1.0 / 255.0) -> Tensor:
    """
    Rescale image pixel values.
    
    Args:
        image: Input tensor (uint8 [0, 255] or float)
        scale_factor: Scale factor (default 1/255 for uint8 -> [0, 1])
        
    Returns:
        Rescaled float tensor
    """
    return image.float() * scale_factor


def normalize_image(
    image: Tensor,
    mean: float = IMAGE_MEAN,
    std: float = IMAGE_STD,
) -> Tensor:
    """
    Normalize image using mean and std.
    
    Args:
        image: Input tensor in [0, 1] range
        mean: Mean value (0.5 for Eagle)
        std: Std value (0.5 for Eagle)
        
    Returns:
        Normalized tensor in [-1, 1] range
    """
    return (image - mean) / std


def preprocess_image_for_eagle(
    image: Tensor,
    target_height: int,
    target_width: int,
) -> Tensor:
    """
    Full preprocessing pipeline for Eagle VLM (traceable).
    
    Replaces: process_vision_info resize + image_processor rescale/normalize
    
    Args:
        image: Input tensor (C, H, W) as uint8 [0, 255]
        target_height: Target height (pre-computed via smart_resize)
        target_width: Target width (pre-computed via smart_resize)
        
    Returns:
        Processed tensor (C, H, W) as float32 in [-1, 1]
    """
    # Step 1: Resize (bicubic with antialias)
    image = resize_image_torch(image, target_height, target_width, mode='bicubic')
    
    # Step 1.5: Round and clamp to [0, 255] to simulate PIL uint8 behavior
    # (bicubic interpolation can produce overshoot outside original range)
    image = image.round().clamp(0, 255)
    
    # Step 2: Rescale from [0, 255] to [0, 1]
    image = rescale_image(image)
    
    # Step 3: Normalize from [0, 1] to [-1, 1]
    image = normalize_image(image, mean=IMAGE_MEAN, std=IMAGE_STD)
    
    return image


class EagleImagePreprocessor(torch.nn.Module):
    """
    Traceable image preprocessor module for Eagle VLM.
    
    Usage:
        preprocessor = EagleImagePreprocessor(target_height=448, target_width=448)
        processed = preprocessor(image_tensor)
    """
    
    def __init__(
        self,
        target_height: int,
        target_width: int,
        mean: float = IMAGE_MEAN,
        std: float = IMAGE_STD,
    ):
        super().__init__()
        self.target_height = target_height
        self.target_width = target_width
        # Register as buffers so they're included in state_dict
        self.register_buffer('mean', torch.tensor(mean))
        self.register_buffer('std', torch.tensor(std))
    
    def forward(self, image: Tensor) -> Tensor:
        """
        Args:
            image: (C, H, W) or (N, C, H, W) uint8 tensor
            
        Returns:
            Processed tensor in [-1, 1] range
        """
        # Resize
        squeeze_batch = False
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze_batch = True
        
        # Resize with bicubic interpolation (torchvision v2 PIL-compatible kernel)
        image = tv_resize(
            image.float(),
            size=[self.target_height, self.target_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        
        # Round and clamp to [0, 255] to simulate PIL uint8 behavior
        image = image.round().clamp(0, 255)
        
        # Rescale and normalize: (x / 255 - 0.5) / 0.5 = x / 127.5 - 1
        image = image / 255.0
        image = (image - self.mean) / self.std
        
        if squeeze_batch:
            image = image.squeeze(0)
        
        return image
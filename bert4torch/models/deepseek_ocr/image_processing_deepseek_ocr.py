# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for DeepSeek-OCR."""

import math
from collections.abc import Iterable
from typing import List, Optional, Tuple, Union, Dict
import torch
from torch import nn
from PIL import Image, ImageOps
from ...processor.image_processing_backends import TorchvisionBackend
from ...processor.image_processing_utils import BatchFeature, register_image_processor
from ...processor.image_transforms import group_images_by_shape, reorder_images
from ...processor.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processor.processing_utils import ImagesKwargs, Unpack
from ...snippets import TensorType, auto_docstring, is_torchvision_available, safe_import


if is_torchvision_available():
    from torchvision import transforms
    from torchvision.transforms.v2 import functional as tvF


class DeepseekOcrImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    crop_mode (`bool`, *optional*, defaults to `True`):
        Whether to use crop mode for image processing.
    base_size (`int`, *optional*, defaults to `1024`):
        The base size for the global view image.
    crop_thread (`int`, *optional*, defaults to `768`):
        The threshold size for cropping images.
    dynamic_preprocess_max_num (`int`, *optional*, defaults to `6`):
        The maximum number of patches for dynamic preprocessing.
    image_size (`int`, *optional*, defaults to `768`):
        The target image size for processing.
    """

    crop_mode: bool
    base_size: int
    crop_thread: int
    dynamic_preprocess_max_num: int
    image_size: int


def normalize_transform(mean, std):
    """Create a normalization transform."""
    if mean is None and std is None:
        return None
    elif mean is None and std is not None:
        mean = [0.] * len(std)
    elif std is None and mean is not None:
        std = [1.] * len(mean)
    return transforms.Normalize(mean=mean, std=std)


class BasicImageTransform:
    def __init__(
        self, 
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True
    ):

        self.mean = mean
        self.std = std
    
        transform_pipelines = [
            transforms.ToTensor()
        ]

        normalize = normalize_transform(mean, std) if normalize else nn.Identity()
        if normalize is not None:
            transform_pipelines.append(normalize)

        self.transform = transforms.Compose(transform_pipelines)
    
    def __call__(self, x):
        x = self.transform(x)
        return x


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=6, image_size=768, use_thumbnail=False):
    """Dynamically preprocess image by splitting into multiple patches.

    Args:
        image: PIL image to preprocess.
        min_num: Minimum number of patches.
        max_num: Maximum number of patches.
        image_size: Target size for each patch.
        use_thumbnail: Whether to add a thumbnail.

    Returns:
        Tuple of (list of processed image patches, target aspect ratio).
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate target ratios
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and split
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images, target_aspect_ratio


@auto_docstring
@register_image_processor
class DeepseekOcrImageProcessor(TorchvisionBackend):
    """Image processor for DeepSeek-OCR using TorchvisionBackend."""

    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 768}
    default_to_square = False
    do_rescale = True
    do_normalize = True
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.5, 0.5, 0.5)
    do_convert_rgb = True
    patch_size = 16
    downsample_ratio = 4
    merge_size = 2
    valid_kwargs = DeepseekOcrImageProcessorKwargs
    model_input_names = ["images_ori", "images_crop", "images_spatial_crop"]

    # DeepSeek-OCR specific attributes
    crop_mode = True
    base_size = 1024
    crop_thread = 768
    image_size = 768
    dynamic_preprocess_max_num = 6
    image_token_id = 128815

    def __init__(self, **kwargs: Unpack[DeepseekOcrImageProcessorKwargs]):
        # Extract DeepSeek-OCR specific kwargs
        self.crop_mode = kwargs.pop("crop_mode", self.crop_mode)
        self.base_size = kwargs.pop("base_size", self.base_size)
        self.crop_thread = kwargs.pop("crop_thread", self.crop_thread)
        self.image_size = kwargs.pop("image_size", self.image_size)
        self.dynamic_preprocess_max_num = kwargs.pop("dynamic_preprocess_max_num", self.dynamic_preprocess_max_num)

        size = kwargs.pop("size", None)
        size = self.size if size is None else size

        super().__init__(size=size, **kwargs)


    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[DeepseekOcrImageProcessorKwargs],
    ) -> BatchFeature:
        """Preprocess images for DeepSeek-OCR.

        Args:
            images: PIL images or image paths to process.

        Returns:
            BatchFeature containing:
                - images_ori: Original/global view images
                - images_crop: Cropped/local view images
                - images_spatial_crop: Spatial crop information [width_crop_num, height_crop_num]
        """
        # Parse images input
        images = self._prepare_images_structure(images)

        images_list = []
        images_crop_list = []
        images_spatial_crop = []

        image_transform=BasicImageTransform(mean=self.image_mean, std=self.image_std, normalize=True)
        for image in images:
            image_draw = image.copy()
            w, h = image_draw.size
            ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))

            if self.crop_mode:
                if image.size[0] <= self.crop_thread and image.size[1] <= self.crop_thread:
                    crop_ratio = [1, 1]
                else:
                    images_crop_raw, crop_ratio = dynamic_preprocess(
                        image, max_num=self.dynamic_preprocess_max_num, image_size=self.image_size
                    )

                # Process global view
                global_view = ImageOps.pad(
                    image, (self.base_size, self.base_size),
                    color=tuple(int(x * 255) for x in image_transform.mean)
                )
                images_list.append(image_transform(global_view).to(torch.bfloat16))

                width_crop_num, height_crop_num = crop_ratio
                images_spatial_crop.append([width_crop_num, height_crop_num])

                # Process local views if needed
                if width_crop_num > 1 or height_crop_num > 1:
                    for i in range(len(images_crop_raw)):
                        images_crop_list.append(
                            image_transform(images_crop_raw[i]).to(torch.bfloat16)
                        )
            else:
                # Non-crop mode
                if self.image_size <= self.crop_thread:
                    image = image.resize((self.image_size, self.image_size))
                global_view = ImageOps.pad(
                    image, (self.image_size, self.image_size),
                    color=tuple(int(x * 255) for x in image_transform.mean)
                )
                images_list.append(image_transform(global_view).to(torch.bfloat16))
                images_spatial_crop.append([1, 1])

        # Stack results
        if len(images_list) == 0:
            images_ori = torch.zeros((1, 3, self.image_size, self.image_size))
            images_spatial_crop_tensor = torch.zeros((1, 2), dtype=torch.long)
            images_crop = torch.zeros((1, 3, self.base_size, self.base_size))
        else:
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((1, 3, self.base_size, self.base_size))

        return BatchFeature(data={
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop_tensor,
        })

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """Get number of image patches for given image size.

        Args:
            height: Image height.
            width: Image width.
            images_kwargs: Optional kwargs override.

        Returns:
            Number of image patches.
        """
        image_size = images_kwargs.get("image_size", self.image_size) if images_kwargs else self.image_size
        dynamic_preprocess_max_num = images_kwargs.get("dynamic_preprocess_max_num", self.dynamic_preprocess_max_num) if images_kwargs else self.dynamic_preprocess_max_num

        aspect_ratio = width / height
        target_ratios = set(
            (i, j) for n in range(2, dynamic_preprocess_max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if i * j <= dynamic_preprocess_max_num and i * j >= 2
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        crop_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size)
        return crop_ratio[0] * crop_ratio[1] + 1  # +1 for global view


__all__ = ["DeepseekOcrImageProcessor"]

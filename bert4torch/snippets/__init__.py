from torch4keras.snippets import *
from .data_process import *
from .misc import *
from .import_utils import *
from .hub import *
from .generic import *
from .openai_client import *


CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = "preprocessor_config.json"
VIDEO_PROCESSOR_NAME = "video_preprocessor_config.json"
AUDIO_TOKENIZER_NAME = "audio_tokenizer_config.json"
PROCESSOR_NAME = "processor_config.json"
GENERATION_CONFIG_NAME = "generation_config.json"

# 原utils.constant.py
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def auto_docstring(cls):
    """
    自动文档字符串装饰器（占位版本，无实际逻辑）
    """
    return cls


@lru_cache
def get_available_devices() -> frozenset[str]:
    """
    Returns a frozenset of devices available for the current PyTorch installation.
    """
    devices = {"cpu"}  # `cpu` is always supported as a device in PyTorch

    if is_torch_cuda_available():
        devices.add("cuda")

    if is_torch_mps_available():
        devices.add("mps")

    if is_torch_xpu_available():
        devices.add("xpu")

    if is_torch_npu_available():
        devices.add("npu")

    if is_torch_hpu_available():
        devices.add("hpu")

    if is_torch_mlu_available():
        devices.add("mlu")

    if is_torch_musa_available():
        devices.add("musa")

    return frozenset(devices)

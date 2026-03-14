import functools
import gc
import inspect
from typing import Optional

import torch

from .imports import (
    is_cuda_available,
    is_hpu_available,
    is_mlu_available,
    is_mps_available,
    is_musa_available,
    is_neuron_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)

def clear_device_cache(garbage_collection=False):
    """
    Clears the device cache by calling `torch.{backend}.empty_cache`. Can also run `gc.collect()`, but do note that
    this is a *considerable* slowdown and should be used sparingly.
    """
    if garbage_collection:
        gc.collect()

    if is_xpu_available():
        torch.xpu.empty_cache()
    elif is_mlu_available():
        torch.mlu.empty_cache()
    elif is_sdaa_available():
        torch.sdaa.empty_cache()
    elif is_musa_available():
        torch.musa.empty_cache()
    elif is_npu_available():
        torch.npu.empty_cache()
    elif is_mps_available(min_version="2.0"):
        torch.mps.empty_cache()
    elif is_cuda_available():
        torch.cuda.empty_cache()
    elif is_hpu_available():
        # torch.hpu.empty_cache() # not available on hpu as it reserves all device memory for the current process
        pass
    elif is_neuron_available():
        # Not sure it actually does something, but adding for consistency with other backends
        torch.neuron.empty_cache()

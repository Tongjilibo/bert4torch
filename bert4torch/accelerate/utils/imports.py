# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import importlib
import importlib.metadata
from functools import lru_cache, wraps
import torch
from packaging import version
from packaging.version import parse
from .environment import parse_flag_from_env, patch_environment
from .versions import compare_versions, is_torch_version


# Try to run Torch native job in an environment with TorchXLA installed by setting this value to 0.
USE_TORCH_XLA = parse_flag_from_env("USE_TORCH_XLA", default=True)

_torch_xla_available = False
if USE_TORCH_XLA:
    try:
        import torch_xla.core.xla_model as xm  # noqa: F401
        import torch_xla.runtime

        _torch_xla_available = True
    except ImportError:
        pass

# Keep it for is_tpu_available. It will be removed along with is_tpu_available.
_tpu_available = _torch_xla_available

# Cache this result has it's a C FFI call which can be pretty time-consuming
_torch_distributed_available = torch.distributed.is_available()


def _is_package_available(pkg_name, metadata_name=None):
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    if package_exists:
        try:
            # Some libraries have different names in the metadata
            _ = importlib.metadata.metadata(pkg_name if metadata_name is None else metadata_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False


def is_torch_distributed_available() -> bool:
    return _torch_distributed_available


def is_cuda_available():
    """
    Checks if `cuda` is available via an `nvml-based` check which won't trigger the drivers and leave cuda
    uninitialized.
    """
    with patch_environment(PYTORCH_NVML_BASED_CUDA_CHECK="1"):
        available = torch.cuda.is_available()

    return available


@lru_cache
def is_torch_xla_available(check_is_tpu=False, check_is_gpu=False):
    """
    Check if `torch_xla` is available. To train a native pytorch job in an environment with torch xla installed, set
    the USE_TORCH_XLA to false.
    """
    assert not (check_is_tpu and check_is_gpu), "The check_is_tpu and check_is_gpu cannot both be true."

    if not _torch_xla_available:
        return False
    elif check_is_gpu:
        return torch_xla.runtime.device_type() in ["GPU", "CUDA"]
    elif check_is_tpu:
        return torch_xla.runtime.device_type() == "TPU"

    return True


def is_deepspeed_available():
    return _is_package_available("deepspeed")


def is_bnb_available(min_version=None):
    package_exists = _is_package_available("bitsandbytes")
    if package_exists and min_version is not None:
        bnb_version = version.parse(importlib.metadata.version("bitsandbytes"))
        return compare_versions(bnb_version, ">=", min_version)
    else:
        return package_exists


def is_mps_available(min_version="1.12"):
    "Checks if MPS device is available. The minimum version required is 1.12."
    # With torch 1.12, you can use torch.backends.mps
    # With torch 2.0.0, you can use torch.mps
    return is_torch_version(">=", min_version) and torch.backends.mps.is_available() and torch.backends.mps.is_built()


@lru_cache
def is_mlu_available(check_device=False):
    """
    Checks if `mlu` is available via an `cndev-based` check which won't trigger the drivers and leave mlu
    uninitialized.
    """
    if importlib.util.find_spec("torch_mlu") is None:
        return False

    import torch_mlu  # noqa: F401

    with patch_environment(PYTORCH_CNDEV_BASED_MLU_CHECK="1"):
        available = torch.mlu.is_available()

    return available


@lru_cache
def is_musa_available(check_device=False):
    "Checks if `torch_musa` is installed and potentially if a MUSA is in the environment"
    if importlib.util.find_spec("torch_musa") is None:
        return False

    import torch_musa  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no MUSA is found
            _ = torch.musa.device_count()
            return torch.musa.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "musa") and torch.musa.is_available()


@lru_cache
def is_npu_available(check_device=False):
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if importlib.util.find_spec("torch_npu") is None:
        return False

    # NOTE: importing torch_npu may raise error in some envs
    # e.g. inside cpu-only container with torch_npu installed
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()


@lru_cache
def is_sdaa_available(check_device=False):
    "Checks if `torch_sdaa` is installed and potentially if a SDAA is in the environment"
    if importlib.util.find_spec("torch_sdaa") is None:
        return False

    import torch_sdaa  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            _ = torch.sdaa.device_count()
            return torch.sdaa.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "sdaa") and torch.sdaa.is_available()


@lru_cache
def is_hpu_available(init_hccl=False):
    "Checks if `torch.hpu` is installed and potentially if a HPU is in the environment"
    if (
        importlib.util.find_spec("habana_frameworks") is None
        or importlib.util.find_spec("habana_frameworks.torch") is None
    ):
        return False

    import habana_frameworks.torch  # noqa: F401

    if init_hccl:
        import habana_frameworks.torch.distributed.hccl as hccl  # noqa: F401

    return hasattr(torch, "hpu") and torch.hpu.is_available()


@lru_cache
def is_xpu_available(check_device=False):
    """
    Checks if XPU acceleration is available via stock PyTorch (>=2.7) and
    potentially if a XPU is in the environment
    """

    if is_torch_version("<=", "2.6"):
        return False

    if check_device:
        try:
            # Will raise a RuntimeError if no XPU is found
            _ = torch.xpu.device_count()
            return torch.xpu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "xpu") and torch.xpu.is_available()


@lru_cache
def is_neuron_available(check_device=False):
    if importlib.util.find_spec("torch_neuronx") is None:
        return False

    if check_device:
        try:
            import torch_neuronx  # noqa: F401

            # Will raise a RuntimeError if no Neuron is found
            _ = torch.neuron.device_count()
            return torch.neuron.is_available()
        except RuntimeError:
            return False

    return hasattr(torch, "neuron") and torch.neuron.is_available()

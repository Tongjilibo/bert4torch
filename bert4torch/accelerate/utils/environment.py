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

import logging
import os
import platform
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from shutil import which
from typing import Optional, Union

import torch
from packaging.version import parse


logger = logging.getLogger(__name__)


def str_to_bool(value, to_bool: bool = False) -> Union[int, bool]:
    """
    Converts a string representation of truth to `True` (1) or `False` (0).

    True values are `y`, `yes`, `t`, `true`, `on`, and `1`; False value are `n`, `no`, `f`, `false`, `off`, and `0`;
    """
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return 1 if not to_bool else True
    elif value in ("n", "no", "f", "false", "off", "0"):
        return 0 if not to_bool else False
    else:
        raise ValueError(f"invalid truth value {value}")


def _nvidia_smi():
    """
    Returns the right nvidia-smi command based on the system.
    """
    if platform.system() == "Windows":
        # If platform is Windows and nvidia-smi can't be found in path
        # try from systemd drive with default installation path
        command = which("nvidia-smi")
        if command is None:
            command = f"{os.environ['systemdrive']}\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"
    else:
        command = "nvidia-smi"
    return command


def get_gpu_info():
    """
    Gets GPU count and names using `nvidia-smi` instead of torch to not initialize CUDA.

    Largely based on the `gputil` library.
    """
    # Returns as list of `n` GPUs and their names
    output = subprocess.check_output(
        [_nvidia_smi(), "--query-gpu=count,name", "--format=csv,noheader"], universal_newlines=True
    )
    output = output.strip()
    gpus = output.split(os.linesep)
    # Get names from output
    gpu_count = len(gpus)
    gpu_names = [gpu.split(",")[1].strip() for gpu in gpus]
    return gpu_names, gpu_count


def get_driver_version():
    """
    Returns the driver version

    In the case of multiple GPUs, will return the first.
    """
    output = subprocess.check_output(
        [_nvidia_smi(), "--query-gpu=driver_version", "--format=csv,noheader"], universal_newlines=True
    )
    output = output.strip()
    return output.split(os.linesep)[0]


def check_cuda_p2p_ib_support():
    """
    Checks if the devices being used have issues with P2P and IB communications, namely any consumer GPU hardware after
    the 3090.

    Notably uses `nvidia-smi` instead of torch to not initialize CUDA.
    """
    try:
        device_names, device_count = get_gpu_info()
        # As new consumer GPUs get released, add them to `unsupported_devices``
        unsupported_devices = {"RTX 40"}
        if device_count > 1:
            if any(
                unsupported_device in device_name
                for device_name in device_names
                for unsupported_device in unsupported_devices
            ):
                # Check if they have the right driver version
                acceptable_driver_version = "550.40.07"
                current_driver_version = get_driver_version()
                if parse(current_driver_version) < parse(acceptable_driver_version):
                    return False
                return True
    except Exception:
        pass
    return True


def parse_flag_from_env(key, default=False):
    """Returns truthy value for `key` from the env if available else the default."""
    value = os.environ.get(key, str(default))
    return str_to_bool(value) == 1  # As its name indicates `str_to_bool` actually returns an int...


@contextmanager
def patch_environment(**kwargs):
    """
    A context manager that will add each keyword argument passed to `os.environ` and remove them when exiting.

    Will convert the values in `kwargs` to strings and upper-case all the keys.

    Example:

    ```python
    >>> import os
    >>> from accelerate.utils import patch_environment

    >>> with patch_environment(FOO="bar"):
    ...     print(os.environ["FOO"])  # prints "bar"
    >>> print(os.environ["FOO"])  # raises KeyError
    ```
    """
    existing_vars = {}
    for key, value in kwargs.items():
        key = key.upper()
        if key in os.environ:
            existing_vars[key] = os.environ[key]
        os.environ[key] = str(value)

    try:
        yield
    finally:
        for key in kwargs:
            key = key.upper()
            if key in existing_vars:
                # restore previous value
                os.environ[key] = existing_vars[key]
            else:
                os.environ.pop(key, None)

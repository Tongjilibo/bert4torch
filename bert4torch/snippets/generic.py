import functools
import types
from enum import Enum
import json
import os
import numpy as np
from collections import UserDict
from typing import Any
import re
from .import_utils import (
    is_torch_available,
    is_tf_available,
    is_flax_available,
    is_mlx_available
)


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"
    MLX = "mlx"


def copy_func(f):
    """Returns a copy of a function f."""
    # Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def _is_numpy(x):
    return isinstance(x, np.ndarray)


def is_numpy_array(x):
    """
    Tests if `x` is a numpy array or not.
    """
    return _is_numpy(x)


def _is_torch(x):
    import torch

    return isinstance(x, torch.Tensor)


def is_torch_tensor(x):
    """
    Tests if `x` is a torch tensor or not. Safe to call even if torch is not installed.
    """
    return False if not is_torch_available() else _is_torch(x)

	
def _is_torch_device(x):
    import torch

    return isinstance(x, torch.device)


def _is_jax(x):
    import jax.numpy as jnp  # noqa: F811

    return isinstance(x, jnp.ndarray)


def is_jax_tensor(x):
    """
    Tests if `x` is a Jax tensor or not. Safe to call even if jax is not installed.
    """
    return False if not is_flax_available() else _is_jax(x)


def is_torch_device(x):
    """
    Tests if `x` is a torch device or not. Safe to call even if torch is not installed.
    """
    return False if not is_torch_available() else _is_torch_device(x)


def _is_mlx(x):
    import mlx.core as mx

    return isinstance(x, mx.array)


def is_mlx_array(x):
    """
    Tests if `x` is a mlx array or not. Safe to call even when mlx is not installed.
    """
    return False if not is_mlx_available() else _is_mlx(x)


def infer_framework_from_repr(x):
    """
    Tries to guess the framework of an object `x` from its repr (brittle but will help in `is_tensor` to try the
    frameworks in a smart order, without the need to import the frameworks).
    """
    representation = str(type(x))
    if representation.startswith("<class 'torch."):
        return "pt"
    elif representation.startswith("<class 'tensorflow."):
        return "tf"
    elif representation.startswith("<class 'jax"):
        return "jax"
    elif representation.startswith("<class 'numpy."):
        return "np"
    elif representation.startswith("<class 'mlx."):
        return "mlx"

def _get_frameworks_and_test_func(x):
    """
    Returns an (ordered since we are in Python 3.7+) dictionary framework to test function, which places the framework
    we can guess from the repr first, then Numpy, then the others.
    """
    framework_to_test = {
        "pt": is_torch_tensor,
        "jax": is_jax_tensor,
        "np": is_numpy_array,
        "mlx": is_mlx_array,
    }
    preferred_framework = infer_framework_from_repr(x)
    # We will test this one first, then numpy, then the others.
    frameworks = [] if preferred_framework is None else [preferred_framework]
    if preferred_framework != "np":
        frameworks.append("np")
    frameworks.extend([f for f in framework_to_test if f not in [preferred_framework, "np"]])
    return {f: framework_to_test[f] for f in frameworks}


def to_py_obj(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list.
    """
    if isinstance(obj, (int, float)):
        return obj
    elif isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        try:
            arr = np.array(obj)
            if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating):
                return arr.tolist()
        except Exception:
            pass
        return [to_py_obj(o) for o in obj]

    framework_to_py_obj = {
        "pt": lambda obj: obj.tolist(),
        "tf": lambda obj: obj.numpy().tolist(),
        "jax": lambda obj: np.asarray(obj).tolist(),
        "np": lambda obj: obj.tolist(),
    }

    # This gives us a smart order to test the frameworks with the corresponding tests.
    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        if test_func(obj):
            return framework_to_py_obj[framework](obj)

    # tolist also works on 0d np arrays
    if isinstance(obj, np.number):
        return obj.tolist()
    else:
        return obj


def _is_torch_dtype(x):
    import torch

    if isinstance(x, str):
        if hasattr(torch, x):
            x = getattr(torch, x)
        else:
            return False
    return isinstance(x, torch.dtype)


def is_torch_dtype(x):
    """
    Tests if `x` is a torch dtype or not. Safe to call even if torch is not installed.
    """
    return False if not is_torch_available() else _is_torch_dtype(x)


def _is_tensorflow(x):
    import tensorflow as tf

    return isinstance(x, tf.Tensor)


def is_tf_tensor(x):
    """
    Tests if `x` is a tensorflow tensor or not. Safe to call even if tensorflow is not installed.
    """
    return False if not is_tf_available() else _is_tensorflow(x)


def is_timm_config_dict(config_dict: dict[str, Any]) -> bool:
    """Checks whether a config dict is a timm config dict."""
    return "pretrained_cfg" in config_dict


def is_timm_local_checkpoint(pretrained_model_path: str) -> bool:
    """
    Checks whether a checkpoint is a timm model checkpoint.
    """
    if pretrained_model_path is None:
        return False

    # in case it's Path, not str
    pretrained_model_path = str(pretrained_model_path)

    is_file = os.path.isfile(pretrained_model_path)
    is_dir = os.path.isdir(pretrained_model_path)

    # pretrained_model_path is a file
    if is_file and pretrained_model_path.endswith(".json"):
        with open(pretrained_model_path) as f:
            config_dict = json.load(f)
        return is_timm_config_dict(config_dict)

    # pretrained_model_path is a directory with a config.json
    if is_dir and os.path.exists(os.path.join(pretrained_model_path, "config.json")):
        with open(os.path.join(pretrained_model_path, "config.json")) as f:
            config_dict = json.load(f)
        return is_timm_config_dict(config_dict)

    return False


def to_numpy(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a Numpy array.
    """

    framework_to_numpy = {
        "pt": lambda obj: obj.detach().cpu().numpy(),
        "tf": lambda obj: obj.numpy(),
        "jax": lambda obj: np.asarray(obj),
        "np": lambda obj: obj,
    }

    if isinstance(obj, (dict, UserDict)):
        return {k: to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return np.array(obj)

    # This gives us a smart order to test the frameworks with the corresponding tests.
    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        if test_func(obj):
            return framework_to_numpy[framework](obj)

    return obj


def safe_load_json_file(json_file: str):
    "A helper to load safe config files and raise a proper error message if it wasn't serialized correctly"
    try:
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
    except json.JSONDecodeError:
        raise OSError(f"It looks like the config file at '{json_file}' is not a valid JSON file.")
    return config_dict


def is_flash_attention_requested(
    config=None, requested_attention_implementation: str | None = None, version: int | None = None
) -> bool:
    """
    Checks whether some flavor of flash attention is requested or not. Optionally, checks for a specific version of
    flash attention.

    This is checked against one of the two arguments, i.e. either the `config` or the directly passed value
    `requested_attention_implementation`. Otherwise, an error will be raised (ambiguity).

    The different versions of flash attention are usually
    - Implementations based on the original flash attention repo: https://github.com/Dao-AILab/flash-attention
    - Kernels implementations such as: https://huggingface.co/kernels-community/vllm-flash-attn3
    """
    if config is not None and requested_attention_implementation is not None:
        raise ValueError(
            "Requested attention implementation is ambiguous: "
            "Please pass either the config or the name of the attention implementation, not both."
        )

    if config is not None:
        checked_attention_implementation = config._attn_implementation
    else:
        checked_attention_implementation = requested_attention_implementation

    # theoretically can happen, equivalent to default implementation (sdpa/eager)
    if checked_attention_implementation is None:
        return False

    # If a specific version is requested, look for a pattern of type "flash...{version}"
    if version is not None:
        return re.match(r".*flash.*" + str(version), checked_attention_implementation) is not None
    # Otherwise, just check "flash" is in the attention implementation
    return "flash" in checked_attention_implementation


def torch_int(x):
    """
    Casts an input to a torch int64 tensor if we are in a tracing context, otherwise to a Python int.
    """
    if not is_torch_available():
        return int(x)
    import torch
    return x.to(torch.int64) if torch.jit.is_tracing() and isinstance(x, torch.Tensor) else int(x)
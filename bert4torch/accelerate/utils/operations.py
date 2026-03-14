from collections.abc import Mapping
from typing import Any
import torch
from .imports import is_npu_available


def is_torch_tensor(tensor):
    return isinstance(tensor, torch.Tensor)


def is_namedtuple(data):
    """
    Checks if `data` is a `namedtuple` or not. Can have false positives, but only if a user is trying to mimic a
    `namedtuple` perfectly.
    """
    return isinstance(data, tuple) and hasattr(data, "_asdict") and hasattr(data, "_fields")


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple, or namedtuple)
    """
    # Some objects may not be able to instantiate from a generator directly
    if is_namedtuple(obj):
        return type(obj)(*list(generator))
    else:
        return type(obj)(generator)

def find_device(data):
    """
    Finds the device on which a nested dict/list/tuple of tensors lies (assuming they are all on the same device).

    Args:
        (nested list/tuple/dictionary of `torch.Tensor`): The data we want to know the device of.
    """
    if isinstance(data, Mapping):
        for obj in data.values():
            device = find_device(obj)
            if device is not None:
                return device
    elif isinstance(data, (tuple, list)):
        for obj in data:
            device = find_device(obj)
            if device is not None:
                return device
    elif isinstance(data, torch.Tensor):
        return data.device


def send_to_device(tensor, device, non_blocking=False, skip_keys=None):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to send to a given device.
        device (`torch.device`):
            The device to send the data to.

    Returns:
        The same data structure as `tensor` with all tensors sent to the proper device.
    """
    if is_torch_tensor(tensor) or hasattr(tensor, "to"):
        # `torch.Tensor.to("npu")` could not find context when called for the first time (see this [issue](https://gitee.com/ascend/pytorch/issues/I8KECW?from=project-issue)).
        if device == "npu":
            device = "npu:0"
        try:
            return tensor.to(device, non_blocking=non_blocking)
        except TypeError:  # .to() doesn't accept non_blocking as kwarg
            return tensor.to(device)
        except AssertionError as error:
            # `torch.Tensor.to(<int num>)` is not supported by `torch_npu` (see this [issue](https://github.com/Ascend/pytorch/issues/16)).
            # This call is inside the try-block since is_npu_available is not supported by torch.compile.
            if is_npu_available():
                if isinstance(device, int):
                    device = f"npu:{device}"
            else:
                raise error
        try:
            return tensor.to(device, non_blocking=non_blocking)
        except TypeError:  # .to() doesn't accept non_blocking as kwarg
            return tensor.to(device)
    elif isinstance(tensor, (tuple, list)):
        return honor_type(
            tensor, (send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys) for t in tensor)
        )
    elif isinstance(tensor, Mapping):
        if isinstance(skip_keys, str):
            skip_keys = [skip_keys]
        elif skip_keys is None:
            skip_keys = []
        return type(tensor)(
            {
                k: t if k in skip_keys else send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys)
                for k, t in tensor.items()
            }
        )
    else:
        return tensor
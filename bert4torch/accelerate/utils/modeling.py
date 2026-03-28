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

import inspect
import logging
import re
import warnings
from collections import OrderedDict, defaultdict
from typing import Optional, Union
import torch
from torch import nn
import os
from .dataclasses import CustomDtype
from .offload import offload_weight
from .imports import (
    is_hpu_available,
    is_mlu_available,
    is_mps_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)
from .memory import clear_device_cache
from .versions import is_torch_version


MEMORY_USAGE_RATIO = os.environ.get("MEMORY_SHRINK_RATIO", 0.85)


logger = logging.getLogger(__name__)


def check_device_same(first_device, second_device):
    """
    Utility method to check if two `torch` devices are similar. When dealing torch accelerator devices(e.g. cuda, xpu),
    torch throws `False` for `torch.device("cuda") == torch.device("cuda:0")` whereas they should be the same

    Args:
        first_device (`torch.device`):
            First device to check
        second_device (`torch.device`):
            Second device to check
    """
    if first_device.type != second_device.type:
        return False

    if first_device.type != "cpu" and first_device.index is None:
        # In case the first_device is an torch accelerator device(e.g. cuda, xpu) and have
        # the index attribute set to `None`, default it to `0`
        first_device = torch.device(first_device.type, index=0)

    if second_device.type != "cpu" and second_device.index is None:
        # In case the second_device is an torch accelerator device(e.g. cuda, xpu) and have
        # the index attribute set to `None`, default it to `0`
        second_device = torch.device(second_device.type, index=0)

    return first_device == second_device


def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.

    Example:

    ```py
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    mem_size = -1
    err_msg = (
        f"`size` {size} is not in a valid format. Use an integer for bytes, or a string with an unit (like '5.0GB')."
    )
    try:
        if isinstance(size, int):
            mem_size = size
        elif size.upper().endswith("GIB"):
            mem_size = int(float(size[:-3]) * (2**30))
        elif size.upper().endswith("MIB"):
            mem_size = int(float(size[:-3]) * (2**20))
        elif size.upper().endswith("KIB"):
            mem_size = int(float(size[:-3]) * (2**10))
        elif size.upper().endswith("GB"):
            int_size = int(float(size[:-2]) * (10**9))
            mem_size = int_size // 8 if size.endswith("b") else int_size
        elif size.upper().endswith("MB"):
            int_size = int(float(size[:-2]) * (10**6))
            mem_size = int_size // 8 if size.endswith("b") else int_size
        elif size.upper().endswith("KB"):
            int_size = int(float(size[:-2]) * (10**3))
            mem_size = int_size // 8 if size.endswith("b") else int_size
    except ValueError:
        raise ValueError(err_msg)

    if mem_size < 0:
        raise ValueError(err_msg)
    return mem_size


def dtype_byte_size(dtype: torch.dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.

    Example:

    ```py
    >>> dtype_byte_size(torch.float32)
    4
    ```
    """
    if dtype == torch.bool:
        return 1 / 8
    elif dtype == CustomDtype.INT2:
        return 1 / 4
    elif dtype == CustomDtype.INT4:
        return 1 / 2
    elif dtype == CustomDtype.FP8:
        return 1
    elif is_torch_version(">=", "2.1.0") and dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        return 1
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def set_module_tensor_to_device(
    module: nn.Module,
    tensor_name: str,
    device: Union[int, str, torch.device],
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    fp16_statistics: Optional[torch.HalfTensor] = None,
    tied_params_map: Optional[dict[int, dict[torch.device, torch.Tensor]]] = None,
    non_blocking: bool = False,
    clear_cache: bool = True,
):
    """
    A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
    `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).

    Args:
        module (`torch.nn.Module`):
            The module in which the tensor we want to move lives.
        tensor_name (`str`):
            The full name of the parameter/buffer.
        device (`int`, `str` or `torch.device`):
            The device on which to set the tensor.
        value (`torch.Tensor`, *optional*):
            The value of the tensor (useful when going from the meta device to any other device).
        dtype (`torch.dtype`, *optional*):
            If passed along the value of the parameter will be cast to this `dtype`. Otherwise, `value` will be cast to
            the dtype of the existing parameter in the model.
        fp16_statistics (`torch.HalfTensor`, *optional*):
            The list of fp16 statistics to set on the module, used for 8 bit model serialization.
        tied_params_map (Dict[int, Dict[torch.device, torch.Tensor]], *optional*, defaults to `None`):
            A map of current data pointers to dictionaries of devices to already dispatched tied weights. For a given
            execution device, this parameter is useful to reuse the first available pointer of a shared weight on the
            device for all others, instead of duplicating memory.
        non_blocking (`bool`, *optional*, defaults to `False`):
            If `True`, the device transfer will be asynchronous with respect to the host, if possible.
        clear_cache (`bool`, *optional*, defaults to `True`):
            Whether or not to clear the device cache after setting the tensor on the device.
    """
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]

    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers
    old_value = getattr(module, tensor_name)

    # Treat the case where old_value (or a custom `value`, typically offloaded to RAM/disk) belongs to a tied group, and one of the weight
    # in the tied group has already been dispatched to the device, by avoiding reallocating memory on the device and just copying the pointer.
    if (
        value is not None
        and tied_params_map is not None
        and value.data_ptr() in tied_params_map
        and device in tied_params_map[value.data_ptr()]
    ):
        module._parameters[tensor_name] = tied_params_map[value.data_ptr()][device]
        return
    elif (
        tied_params_map is not None
        and old_value.data_ptr() in tied_params_map
        and device in tied_params_map[old_value.data_ptr()]
    ):
        module._parameters[tensor_name] = tied_params_map[old_value.data_ptr()][device]
        return

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")

    param = module._parameters[tensor_name] if tensor_name in module._parameters else None
    param_cls = type(param)

    if value is not None:
        # We can expect mismatches when using bnb 4bit since Params4bit will reshape and pack the weights.
        # In other cases, we want to make sure we're not loading checkpoints that do not match the config.
        if old_value.shape != value.shape and param_cls.__name__ != "Params4bit":
            raise ValueError(
                f'Trying to set a tensor of shape {value.shape} in "{tensor_name}" (which has shape {old_value.shape}), this looks incorrect.'
            )

        if dtype is None:
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
            value = value.to(old_value.dtype, non_blocking=non_blocking)
        elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            value = value.to(dtype, non_blocking=non_blocking)

    device_quantization = None
    with torch.no_grad():
        # leave it on cpu first before moving them to device
        # # fix the case where the device is meta, we don't want to put it on cpu because there is no data =0
        if (
            param is not None
            and param.device.type not in ("cuda", "xpu")
            and torch.device(device).type in ("cuda", "xpu")
            and param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]
        ):
            device_quantization = device
            device = "cpu"
        # `torch.Tensor.to(<int num>)` is not supported by `torch_npu` (see this [issue](https://github.com/Ascend/pytorch/issues/16)).
        if isinstance(device, int):
            if is_npu_available():
                device = f"npu:{device}"
            elif is_mlu_available():
                device = f"mlu:{device}"
            elif is_sdaa_available():
                device = f"sdaa:{device}"
            elif is_musa_available():
                device = f"musa:{device}"
            elif is_hpu_available():
                device = "hpu"
        if "xpu" in str(device) and not is_xpu_available():
            raise ValueError(f'{device} is not available, you should use device="cpu" instead')
        if value is None:
            new_value = old_value.to(device, non_blocking=non_blocking)
            if dtype is not None and device in ["meta", torch.device("meta")]:
                if not str(old_value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
                    new_value = new_value.to(dtype, non_blocking=non_blocking)

                if not is_buffer:
                    module._parameters[tensor_name] = param_cls(new_value, requires_grad=old_value.requires_grad)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device, non_blocking=non_blocking)
        else:
            new_value = torch.tensor(value, device=device)
        if device_quantization is not None:
            device = device_quantization
        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif value is not None or not check_device_same(torch.device(device), module._parameters[tensor_name].device):
            param_cls = type(module._parameters[tensor_name])
            kwargs = module._parameters[tensor_name].__dict__
            if param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]:
                if param_cls.__name__ == "Int8Params" and new_value.dtype == torch.float32:
                    # downcast to fp16 if any - needed for 8bit serialization
                    new_value = new_value.to(torch.float16, non_blocking=non_blocking)
                # quantize module that are going to stay on the cpu so that we offload quantized weights
                if device == "cpu" and param_cls.__name__ == "Int8Params":
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(0).to("cpu")
                    new_value.CB = new_value.CB.to("cpu")
                    new_value.SCB = new_value.SCB.to("cpu")
                else:
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(
                        device, non_blocking=non_blocking
                    )
            elif param_cls.__name__ in ["QTensor", "QBitsTensor"]:
                new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad).to(
                    device, non_blocking=non_blocking
                )
            elif param_cls.__name__ in ["AffineQuantizedTensor"]:
                new_value = new_value.to(device, non_blocking=non_blocking)
            else:
                new_value = param_cls(new_value, requires_grad=old_value.requires_grad).to(
                    device, non_blocking=non_blocking
                )

            module._parameters[tensor_name] = new_value
            if fp16_statistics is not None:
                module._parameters[tensor_name].SCB = fp16_statistics.to(device, non_blocking=non_blocking)
                del fp16_statistics
            # as we put the weight to meta, it doesn't have SCB attr anymore. make sure that it is not a meta weight
            if (
                module.__class__.__name__ == "Linear8bitLt"
                and getattr(module.weight, "SCB", None) is None
                and str(module.weight.device) != "meta"
            ):
                # quantize only if necessary
                device_index = torch.device(device).index if torch.device(device).type in ["cuda", "xpu"] else None
                if not getattr(module.weight, "SCB", None) and device_index is not None:
                    if module.bias is not None and module.bias.device.type != "meta":
                        # if a bias exists, we need to wait until the bias is set on the correct device
                        module = module.to(device_index)
                    elif module.bias is None:
                        # if no bias exists, we can quantize right away
                        module = module.to(device_index)
            elif (
                module.__class__.__name__ == "Linear4bit"
                and getattr(module.weight, "quant_state", None) is None
                and str(module.weight.device) != "meta"
            ):
                # quantize only if necessary
                device_index = torch.device(device).index if torch.device(device).type in ["cuda", "xpu"] else None
                if not getattr(module.weight, "quant_state", None) and device_index is not None:
                    module.weight = module.weight.to(device_index)

    # clean pre and post forward hook
    if clear_cache and device not in ("cpu", "meta"):
        clear_device_cache()

    # When handling tied weights, we update tied_params_map to keep track of the tied weights that have already been allocated on the device in
    # order to avoid duplicating memory, see above.
    if (
        tied_params_map is not None
        and old_value.data_ptr() in tied_params_map
        and device not in tied_params_map[old_value.data_ptr()]
    ):
        tied_params_map[old_value.data_ptr()][device] = new_value
    elif (
        value is not None
        and tied_params_map is not None
        and value.data_ptr() in tied_params_map
        and device not in tied_params_map[value.data_ptr()]
    ):
        tied_params_map[value.data_ptr()][device] = new_value


def named_module_tensors(
    module: nn.Module, include_buffers: bool = True, recurse: bool = False, remove_non_persistent: bool = False
):
    """
    A helper function that gathers all the tensors (parameters + buffers) of a given module. If `include_buffers=True`
    it's the same as doing `module.named_parameters(recurse=recurse) + module.named_buffers(recurse=recurse)`.

    Args:
        module (`torch.nn.Module`):
            The module we want the tensors on.
        include_buffer (`bool`, *optional*, defaults to `True`):
            Whether or not to include the buffers in the result.
        recurse (`bool`, *optional`, defaults to `False`):
            Whether or not to go look in every submodule or just return the direct parameters and buffers.
        remove_non_persistent (`bool`, *optional*, defaults to `False`):
            Whether or not to remove the non persistent buffer from the buffers. Useful only when include_buffers =
            True
    """
    yield from module.named_parameters(recurse=recurse)

    if include_buffers:
        non_persistent_buffers = set()
        if remove_non_persistent:
            non_persistent_buffers = get_non_persistent_buffers(module, recurse=recurse)
        for named_buffer in module.named_buffers(recurse=recurse):
            name, _ = named_buffer
            if name not in non_persistent_buffers:
                yield named_buffer


def get_non_persistent_buffers(module: nn.Module, recurse: bool = False, fqns: bool = False):
    """
    Gather all non persistent buffers of a given modules into a set

    Args:
        module (`nn.Module`):
            The module we want the non persistent buffers on.
        recurse (`bool`, *optional*, defaults to `False`):
            Whether or not to go look in every submodule or just return the direct non persistent buffers.
        fqns (`bool`, *optional*, defaults to `False`):
            Whether or not to return the fully-qualified names of the non persistent buffers.
    """

    non_persistent_buffers_set = module._non_persistent_buffers_set
    if recurse:
        for n, m in module.named_modules():
            if fqns:
                non_persistent_buffers_set |= {n + "." + b for b in m._non_persistent_buffers_set}
            else:
                non_persistent_buffers_set |= m._non_persistent_buffers_set

    return non_persistent_buffers_set


def check_tied_parameters_in_config(model: nn.Module):
    """
    Check if there is any indication in the given model that some weights should be tied.

    Args:
        model (`torch.nn.Module`): The model to inspect

    Returns:
        bool: True if the model needs to have tied weights
    """

    # based on model.tie_weights() method
    has_tied_word_embedding = False
    has_tied_encoder_decoder = False
    has_tied_module = False

    if "PreTrainedModel" in [c.__name__ for c in inspect.getmro(model.__class__)]:
        has_tied_word_embedding = False
        model_decoder_config = None
        if hasattr(model, "config"):
            model_decoder_config = (
                model.config.get_text_config(decoder=True)
                if hasattr(model.config, "get_text_config")
                else model.config
            )
        has_tied_word_embedding = (
            model_decoder_config is not None
            and getattr(model_decoder_config, "tie_word_embeddings", False)
            and model.get_output_embeddings()
        )

        has_tied_encoder_decoder = (
            hasattr(model, "config")
            and getattr(model.config, "is_encoder_decoder", False)
            and getattr(model.config, "tie_encoder_decoder", False)
        )
        has_tied_module = any(hasattr(module, "_tie_weights") for module in model.modules())
    return any([has_tied_word_embedding, has_tied_encoder_decoder, has_tied_module])


def _get_param_device(param, device_map):
    if param in device_map:
        return device_map[param]
    parent_param = ".".join(param.split(".")[:-1])
    if parent_param == param:
        raise ValueError(f"The `device_map` does not contain the module {param}.")
    else:
        return _get_param_device(parent_param, device_map)


def check_tied_parameters_on_same_device(tied_params, device_map):
    """
    Check if tied parameters are on the same device

    Args:
        tied_params (`List[List[str]]`):
            A list of lists of parameter names being all tied together.

        device_map (`Dict[str, Union[int, str, torch.device]]`):
            A map that specifies where each submodule should go.

    """
    for tie_param in tied_params:
        tie_param_devices = {}
        for param in tie_param:
            tie_param_devices[param] = _get_param_device(param, device_map)
        if len(set(tie_param_devices.values())) > 1:
            logger.warning(
                f"Tied parameters are on different devices: {tie_param_devices}. "
                "Please modify your custom device map or set `device_map='auto'`. "
            )


def find_tied_parameters(model: torch.nn.Module, **kwargs) -> list[list[str]]:
    """
    Find the tied parameters in a given model.

    <Tip warning={true}>

    The signature accepts keyword arguments, but they are for the recursive part of this function and you should ignore
    them.

    </Tip>

    Args:
        model (`torch.nn.Module`): The model to inspect.

    Returns:
        List[List[str]]: A list of lists of parameter names being all tied together.

    Example:

    ```py
    >>> from collections import OrderedDict
    >>> import torch.nn as nn

    >>> model = nn.Sequential(OrderedDict([("linear1", nn.Linear(4, 4)), ("linear2", nn.Linear(4, 4))]))
    >>> model.linear2.weight = model.linear1.weight
    >>> find_tied_parameters(model)
    [['linear1.weight', 'linear2.weight']]
    ```
    """

    # get ALL model parameters and their names
    all_named_parameters = {name: param for name, param in model.named_parameters(remove_duplicate=False)}

    # get ONLY unique named parameters,
    # if parameter is tied and have multiple names, it will be included only once
    no_duplicate_named_parameters = {name: param for name, param in model.named_parameters(remove_duplicate=True)}

    # the difference of the two sets will give us the tied parameters
    tied_param_names = set(all_named_parameters.keys()) - set(no_duplicate_named_parameters.keys())

    # 'tied_param_names' contains the names of parameters that are tied in the model, but we do not know
    # which names refer to the same parameter. To identify this, we need to group them together.
    tied_param_groups = {}
    for tied_param_name in tied_param_names:
        tied_param = all_named_parameters[tied_param_name]
        for param_name, param in no_duplicate_named_parameters.items():
            # compare if parameters are the same, if so, group their names together
            if param is tied_param:
                if param_name not in tied_param_groups:
                    tied_param_groups[param_name] = []
                tied_param_groups[param_name].append(tied_param_name)

    return [sorted([weight] + list(set(tied))) for weight, tied in tied_param_groups.items()]


def retie_parameters(model, tied_params):
    """
    Reties tied parameters in a given model if the link was broken (for instance when adding hooks).

    Args:
        model (`torch.nn.Module`):
            The model in which to retie parameters.
        tied_params (`List[List[str]]`):
            A mapping parameter name to tied parameter name as obtained by `find_tied_parameters`.
    """
    for tied_group in tied_params:
        param_to_tie = None
        # two loops : the first one to set param_to_tie , the second one to change the values of tied_group
        for param_name in tied_group:
            module = model
            splits = param_name.split(".")
            for split in splits[:-1]:
                module = getattr(module, split)
            param = getattr(module, splits[-1])
            if param_to_tie is None and param.device != torch.device("meta"):
                param_to_tie = param
                break
        if param_to_tie is not None:
            for param_name in tied_group:
                module = model
                splits = param_name.split(".")
                for split in splits[:-1]:
                    module = getattr(module, split)
                setattr(module, splits[-1], param_to_tie)


def _get_proper_dtype(dtype: Union[str, torch.device]) -> torch.dtype:
    """
    Just does torch.dtype(dtype) if necessary.
    """
    if isinstance(dtype, str):
        # We accept "torch.float16" or just "float16"
        dtype = dtype.replace("torch.", "")
        dtype = getattr(torch, dtype)
    return dtype


def compute_module_sizes(
    model: nn.Module,
    dtype: Optional[Union[str, torch.device]] = None,
    special_dtypes: Optional[dict[str, Union[str, torch.device]]] = None,
    buffers_only: bool = False,
):
    """
    Compute the size of each submodule of a given model.
    """
    if dtype is not None:
        dtype = _get_proper_dtype(dtype)
        dtype_size = dtype_byte_size(dtype)
    if special_dtypes is not None:
        special_dtypes = {key: _get_proper_dtype(dtyp) for key, dtyp in special_dtypes.items()}
        special_dtypes_size = {key: dtype_byte_size(dtyp) for key, dtyp in special_dtypes.items()}
    module_sizes = defaultdict(int)

    module_list = []

    if not buffers_only:
        module_list = named_module_tensors(model, recurse=True)
    else:
        module_list = model.named_buffers(recurse=True)

    for name, tensor in module_list:
        if special_dtypes is not None and name in special_dtypes:
            size = tensor.numel() * special_dtypes_size[name]
        elif dtype is None:
            size = tensor.numel() * dtype_byte_size(tensor.dtype)
        elif str(tensor.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            # According to the code in set_module_tensor_to_device, these types won't be converted
            # so use their original size here
            size = tensor.numel() * dtype_byte_size(tensor.dtype)
        else:
            size = tensor.numel() * min(dtype_size, dtype_byte_size(tensor.dtype))
        name_parts = name.split(".")
        for idx in range(len(name_parts) + 1):
            module_sizes[".".join(name_parts[:idx])] += size

    return module_sizes


def compute_module_total_buffer_size(
    model: nn.Module,
    dtype: Optional[Union[str, torch.device]] = None,
    special_dtypes: Optional[dict[str, Union[str, torch.device]]] = None,
):
    """
    Compute the total size of buffers in each submodule of a given model.
    """
    module_sizes = compute_module_sizes(model, dtype=dtype, special_dtypes=special_dtypes, buffers_only=True)
    return module_sizes.get("", 0)


def get_max_layer_size(
    modules: list[tuple[str, torch.nn.Module]], module_sizes: dict[str, int], no_split_module_classes: list[str]
):
    """
    Utility function that will scan a list of named modules and return the maximum size used by one full layer. The
    definition of a layer being:
    - a module with no direct children (just parameters and buffers)
    - a module whose class name is in the list `no_split_module_classes`

    Args:
        modules (`List[Tuple[str, torch.nn.Module]]`):
            The list of named modules where we want to determine the maximum layer size.
        module_sizes (`Dict[str, int]`):
            A dictionary mapping each layer name to its size (as generated by `compute_module_sizes`).
        no_split_module_classes (`List[str]`):
            A list of class names for layers we don't want to be split.

    Returns:
        `Tuple[int, List[str]]`: The maximum size of a layer with the list of layer names realizing that maximum size.
    """
    max_size = 0
    layer_names = []
    modules_to_treat = modules.copy()
    while len(modules_to_treat) > 0:
        module_name, module = modules_to_treat.pop(0)
        modules_children = list(module.named_children()) if isinstance(module, torch.nn.Module) else []
        if len(modules_children) == 0 or module.__class__.__name__ in no_split_module_classes:
            # No splitting this one so we compare to the max_size
            size = module_sizes[module_name]
            if size > max_size:
                max_size = size
                layer_names = [module_name]
            elif size == max_size:
                layer_names.append(module_name)
        else:
            modules_to_treat = [(f"{module_name}.{n}", v) for n, v in modules_children] + modules_to_treat
    return max_size, layer_names


def get_max_memory(max_memory: Optional[dict[Union[int, str], Union[int, str]]] = None):
    """
    Get the maximum memory available if nothing is passed, converts string to int otherwise.
    """
    import psutil

    if max_memory is None:
        max_memory = {}
        # Make sure device is initialized on each device to have the right memory info.
        if is_npu_available():
            for i in range(torch.npu.device_count()):
                try:
                    _ = torch.tensor(0, device=torch.device("npu", i))
                    max_memory[i] = torch.npu.mem_get_info(i)[0]
                except Exception:
                    logger.info(f"Device {i} seems unavailable, Proceeding to check subsequent devices.")
                    continue
        elif is_mlu_available():
            for i in range(torch.mlu.device_count()):
                try:
                    _ = torch.tensor(0, device=torch.device("mlu", i))
                    max_memory[i] = torch.mlu.mem_get_info(i)[0]
                except Exception:
                    logger.info(f"Device {i} seems unavailable, Proceeding to check subsequent devices.")
                    continue
        elif is_sdaa_available():
            for i in range(torch.sdaa.device_count()):
                try:
                    _ = torch.tensor(0, device=torch.device("sdaa", i))
                    max_memory[i] = torch.sdaa.mem_get_info(i)[0]
                except Exception:
                    logger.info(f"Device {i} seems unavailable, Proceeding to check subsequent devices.")
                    continue
        elif is_musa_available():
            for i in range(torch.musa.device_count()):
                try:
                    _ = torch.tensor(0, device=torch.device("musa", i))
                    max_memory[i] = torch.musa.mem_get_info(i)[0]
                except Exception:
                    logger.info(f"Device {i} seems unavailable, Proceeding to check subsequent devices.")
                    continue
        elif is_xpu_available():
            for i in range(torch.xpu.device_count()):
                try:
                    _ = torch.tensor(0, device=torch.device("xpu", i))
                    max_memory[i] = torch.xpu.mem_get_info(i)[0]
                except Exception:
                    logger.info(f"Device {i} seems unavailable, Proceeding to check subsequent devices.")
                    continue
        elif is_hpu_available():
            for i in range(torch.hpu.device_count()):
                try:
                    _ = torch.tensor(0, device=torch.device("hpu", i))
                    max_memory[i] = torch.hpu.mem_get_info(i)[0]
                except Exception:
                    logger.info(f"Device {i} seems unavailable, Proceeding to check subsequent devices.")
                    continue
        else:
            for i in range(torch.cuda.device_count()):
                try:
                    _ = torch.tensor([0], device=i)
                    max_memory[i] = torch.cuda.mem_get_info(i)[0]
                except Exception:
                    logger.info(f"Device {i} seems unavailable, Proceeding to check subsequent devices.")
                    continue
        # allocate everything in the mps device as the RAM is shared
        if is_mps_available():
            max_memory["mps"] = psutil.virtual_memory().available
        else:
            max_memory["cpu"] = psutil.virtual_memory().available
        return max_memory

    for key in max_memory:
        if isinstance(max_memory[key], str):
            max_memory[key] = convert_file_size_to_int(max_memory[key])

    # Need to sort the device by type to make sure that we allocate the gpu first.
    # As gpu/npu/xpu are represented by int, we need to sort them first.
    gpu_devices = [k for k in max_memory.keys() if isinstance(k, int)]
    gpu_devices.sort()
    # check if gpu/npu/xpu devices are available and if not, throw a warning
    if is_npu_available():
        num_devices = torch.npu.device_count()
    elif is_mlu_available():
        num_devices = torch.mlu.device_count()
    elif is_sdaa_available():
        num_devices = torch.sdaa.device_count()
    elif is_musa_available():
        num_devices = torch.musa.device_count()
    elif is_xpu_available():
        num_devices = torch.xpu.device_count()
    elif is_hpu_available():
        num_devices = torch.hpu.device_count()
    else:
        num_devices = torch.cuda.device_count()
    for device in gpu_devices:
        if device >= num_devices or device < 0:
            logger.warning(f"Device {device} is not available, available devices are {list(range(num_devices))}")
    # Add the other devices in the preset order if they are available
    all_devices = gpu_devices + [k for k in ["mps", "cpu", "disk"] if k in max_memory.keys()]
    # Raise an error if a device is not recognized
    for k in max_memory.keys():
        if k not in all_devices:
            raise ValueError(
                f"Device {k} is not recognized, available devices are integers(for GPU/XPU), 'mps', 'cpu' and 'disk'"
            )
    max_memory = {k: max_memory[k] for k in all_devices}

    return max_memory


def clean_device_map(device_map: dict[str, Union[int, str, torch.device]], module_name: str = ""):
    """
    Cleans a device_map by grouping all submodules that go on the same device together.
    """
    # Get the value of the current module and if there is only one split across several keys, regroup it.
    prefix = "" if module_name == "" else f"{module_name}."
    values = [v for k, v in device_map.items() if k.startswith(prefix)]
    if len(set(values)) == 1 and len(values) > 1:
        for k in [k for k in device_map if k.startswith(prefix)]:
            del device_map[k]
        device_map[module_name] = values[0]

    # Recurse over the children
    children_modules = [k for k in device_map.keys() if k.startswith(prefix) and len(k) > len(module_name)]
    idx = len(module_name.split(".")) + 1 if len(module_name) > 0 else 1
    children_modules = set(".".join(k.split(".")[:idx]) for k in children_modules)
    for child in children_modules:
        clean_device_map(device_map, module_name=child)

    return device_map

def get_module_leaves(module_sizes):
    module_children = {}
    for module in module_sizes:
        if module == "" or "." not in module:
            continue
        parent = module.rsplit(".", 1)[0]
        module_children[parent] = module_children.get(parent, 0) + 1
    leaves = [module for module in module_sizes if module_children.get(module, 0) == 0 and module != ""]
    return leaves


def get_balanced_memory(
    model: nn.Module,
    max_memory: Optional[dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[list[str]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    special_dtypes: Optional[dict[str, Union[str, torch.device]]] = None,
    low_zero: bool = False,
):
    """
    Compute a `max_memory` dictionary for [`infer_auto_device_map`] that will balance the use of each available GPU.

    <Tip>

    All computation is done analyzing sizes and dtypes of the model parameters. As a result, the model can be on the
    meta device (as it would if initialized within the `init_empty_weights` context manager).

    </Tip>

    Args:
        model (`torch.nn.Module`):
            The model to analyze.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available if unset.
            Example: `max_memory={0: "1GB"}`.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        special_dtypes (`Dict[str, Union[str, torch.device]]`, *optional*):
            If provided, special dtypes to consider for some specific weights (will override dtype used as default for
            all weights).
        low_zero (`bool`, *optional*):
            Minimizes the number of weights on GPU 0, which is convenient when it's used for other operations (like the
            Transformers generate function).
    """
    # Get default / clean up max_memory
    user_not_set_max_memory = max_memory is None
    max_memory = get_max_memory(max_memory)

    if is_npu_available():
        expected_device_type = "npu"
    elif is_mlu_available():
        expected_device_type = "mlu"
    elif is_sdaa_available():
        expected_device_type = "sdaa"
    elif is_musa_available():
        expected_device_type = "musa"
    elif is_xpu_available():
        expected_device_type = "xpu"
    elif is_hpu_available():
        expected_device_type = "hpu"
    elif is_mps_available():
        expected_device_type = "mps"
    else:
        expected_device_type = "cuda"
    num_devices = len([d for d in max_memory if torch.device(d).type == expected_device_type and max_memory[d] > 0])

    if num_devices == 0:
        return max_memory

    if num_devices == 1:
        # We cannot do low_zero on just one GPU, but we will still reserve some memory for the buffer
        low_zero = False
        # If user just asked us to handle memory usage, we should avoid OOM
        if user_not_set_max_memory:
            for key in max_memory.keys():
                if isinstance(key, int):
                    max_memory[key] *= MEMORY_USAGE_RATIO
                    logger.info(
                        f"We will use {MEMORY_USAGE_RATIO*100:.0f}% of the memory on device {key} for storing the model, and 10% for the buffer to avoid OOM. "
                        "You can set `max_memory` in to a higher value to use more memory (at your own risk)."
                    )
                    break  # only one device

    module_sizes = compute_module_sizes(model, dtype=dtype, special_dtypes=special_dtypes)
    per_gpu = module_sizes[""] // (num_devices - 1 if low_zero else num_devices)

    # We can't just set the memory to model_size // num_devices as it will end being too small: each GPU will get
    # slightly less layers and some layers will end up offload at the end. So this function computes a buffer size to
    # add which is the biggest of:
    # - the size of no split block (if applicable)
    # - the mean of the layer sizes
    if no_split_module_classes is None:
        no_split_module_classes = []
    elif not isinstance(no_split_module_classes, (list, tuple)):
        no_split_module_classes = [no_split_module_classes]

    # Identify the size of the no_split_block modules
    if len(no_split_module_classes) > 0:
        no_split_children = {}
        for name, size in module_sizes.items():
            if name == "":
                continue
            submodule = model
            for submodule_name in name.split("."):
                submodule = getattr(submodule, submodule_name)
            class_name = submodule.__class__.__name__
            if class_name in no_split_module_classes and class_name not in no_split_children:
                no_split_children[class_name] = size

            if set(no_split_children.keys()) == set(no_split_module_classes):
                break
        buffer = max(no_split_children.values()) if len(no_split_children) > 0 else 0
    else:
        buffer = 0

    # Compute mean of final modules. In the first dict of module sizes, leaves are the parameters
    leaves = get_module_leaves(module_sizes)
    leaves_set = set(leaves)  # Convert to set for O(1) membership testing
    module_sizes = {n: v for n, v in module_sizes.items() if n not in leaves_set}
    # Once removed, leaves are the final modules.
    leaves = get_module_leaves(module_sizes)
    mean_leaves = int(sum([module_sizes[n] for n in leaves]) / max(len(leaves), 1))
    buffer = int(1.25 * max(buffer, mean_leaves))
    per_gpu += buffer

    # Sorted list of GPUs id (we may have some gpu ids not included in the our max_memory list - let's ignore them)
    gpus_idx_list = list(
        sorted(
            device_id for device_id, device_mem in max_memory.items() if isinstance(device_id, int) and device_mem > 0
        )
    )
    # The last device is left with max_memory just in case the buffer is not enough.
    for idx in gpus_idx_list[:-1]:
        max_memory[idx] = min(max_memory[0] if low_zero and idx == 0 else per_gpu, max_memory[idx])

    if low_zero:
        min_zero = max(0, module_sizes[""] - sum([max_memory[i] for i in range(1, num_devices)]))
        max_memory[0] = min(min_zero, max_memory[0])

    return max_memory




def _init_infer_auto_device_map(
    model: nn.Module,
    max_memory: Optional[dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[list[str]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    special_dtypes: Optional[dict[str, Union[str, torch.device]]] = None,
) -> tuple[
    list[Union[int, str]],
    dict[Union[int, str], Union[int, str]],
    list[Union[int, str]],
    list[int],
    dict[str, int],
    list[list[str]],
    list[str],
    list[tuple[str, nn.Module]],
]:
    """
    Initialize variables required for computing the device map for model allocation.
    """
    max_memory = get_max_memory(max_memory)
    if no_split_module_classes is None:
        no_split_module_classes = []
    elif not isinstance(no_split_module_classes, (list, tuple)):
        no_split_module_classes = [no_split_module_classes]

    devices = list(max_memory.keys())
    if "disk" not in devices:
        devices.append("disk")
    gpus = [device for device in devices if device not in ["cpu", "disk"]]

    # Devices that need to keep space for a potential offloaded layer.
    if "mps" in gpus:
        main_devices = ["mps"]
    elif len(gpus) > 0:
        main_devices = [gpus[0], "cpu"]
    else:
        main_devices = ["cpu"]

    module_sizes = compute_module_sizes(model, dtype=dtype, special_dtypes=special_dtypes)
    tied_parameters = find_tied_parameters(model)
    if check_tied_parameters_in_config(model) and len(tied_parameters) == 0:
        logger.warning(
            "The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function."
        )

    # Direct submodules and parameters
    modules_to_treat = (
        list(model.named_parameters(recurse=False))
        + list(model.named_children())
        + list(model.named_buffers(recurse=False))
    )

    return (
        devices,
        max_memory,
        main_devices,
        gpus,
        module_sizes,
        tied_parameters,
        no_split_module_classes,
        modules_to_treat,
    )


def get_module_size_with_ties(
    tied_params,
    module_size,
    module_sizes,
    modules_to_treat,
) -> tuple[int, list[str], list[nn.Module]]:
    """
    Calculate the total size of a module, including its tied parameters.

    Args:
        tied_params (`List[str]`): The list of tied parameters.
        module_size (`int`): The size of the module without tied parameters.
        module_sizes (`Dict[str, int]`): A dictionary mapping each layer name to its size.
        modules_to_treat (`List[Tuple[str, nn.Module]]`): The list of named modules to treat.

    Returns:
        `Tuple[int, List[str], List[nn.Module]]`: The total size of the module, the names of the tied modules, and the
        tied modules.
    """
    if len(tied_params) < 1:
        return module_size, [], []
    tied_module_names = []
    tied_modules = []

    for tied_param in tied_params:
        tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if tied_param.startswith(n + ".")][0]
        tied_module_names.append(modules_to_treat[tied_module_index][0])
        tied_modules.append(modules_to_treat[tied_module_index][1])

    module_size_with_ties = module_size
    for tied_param, tied_module_name in zip(tied_params, tied_module_names):
        module_size_with_ties += module_sizes[tied_module_name] - module_sizes[tied_param]

    return module_size_with_ties, tied_module_names, tied_modules


def fallback_allocate(
    modules: list[tuple[str, nn.Module]],
    module_sizes: dict[str, int],
    size_limit: Union[int, str],
    no_split_module_classes: Optional[list[str]] = None,
    tied_parameters: Optional[list[list[str]]] = None,
) -> tuple[Optional[str], Optional[nn.Module], list[tuple[str, nn.Module]]]:
    """
    Find a module that fits in the size limit using BFS and return it with its name and the remaining modules.

    Args:
        modules (`List[Tuple[str, nn.Module]]`):
            The list of named modules to search in.
        module_sizes (`Dict[str, int]`):
            A dictionary mapping each layer name to its size (as generated by `compute_module_sizes`).
        size_limit (`Union[int, str]`):
            The maximum size a module can have.
        no_split_module_classes (`Optional[List[str]]`, *optional*):
            A list of class names for layers we don't want to be split.
        tied_parameters (`Optional[List[List[str]]`, *optional*):
            A list of lists of parameter names being all tied together.

    Returns:
        `Tuple[Optional[str], Optional[nn.Module], List[Tuple[str, nn.Module]]]`: A tuple containing:
        - The name of the module that fits within the size limit.
        - The module itself.
        - The list of remaining modules after the found module is removed.
    """
    try:
        size_limit = convert_file_size_to_int(size_limit)
    except ValueError:
        return None, None, modules

    if no_split_module_classes is None:
        no_split_module_classes = []

    if tied_parameters is None:
        tied_parameters = []

    modules_to_search = modules.copy()
    module_found = False

    while modules_to_search:
        name, module = modules_to_search.pop(0)

        tied_param_groups = [
            tied_group
            for tied_group in tied_parameters
            if any(name + "." in k + "." for k in tied_group) and not all(name + "." in k + "." for k in tied_group)
        ]

        tied_params = sum(
            [[p for p in tied_group if name + "." not in p + "."] for tied_group in tied_param_groups], []
        )

        module_size_with_ties, _, _ = get_module_size_with_ties(
            tied_params, module_sizes[name], module_sizes, modules_to_search
        )

        # If the module fits in the size limit, we found it.
        if module_size_with_ties <= size_limit:
            module_found = True
            break

        # The module is too big, we need to split it if possible.
        modules_children = (
            []
            if isinstance(module, nn.Parameter) or isinstance(module, torch.Tensor)
            else list(module.named_children())
        )

        # Split fails, move to the next module
        if len(modules_children) == 0 or module.__class__.__name__ in no_split_module_classes:
            continue

        # split is possible, add the children to the list of modules to search
        modules_children = list(module.named_parameters(recurse=False)) + modules_children
        modules_to_search = [(f"{name}.{n}", v) for n, v in modules_children] + modules_to_search

    if not module_found:
        return None, None, modules

    # Prepare the module list for removal of the found module
    current_names = [n for n, _ in modules]
    dot_idx = [i for i, c in enumerate(name) if c == "."]

    for dot_index in dot_idx:
        parent_name = name[:dot_index]
        if parent_name in current_names:
            parent_module_idx = current_names.index(parent_name)
            _, parent_module = modules[parent_module_idx]
            module_children = list(parent_module.named_parameters(recurse=False)) + list(
                parent_module.named_children()
            )
            modules = (
                modules[:parent_module_idx]
                + [(f"{parent_name}.{n}", v) for n, v in module_children]
                + modules[parent_module_idx + 1 :]
            )
            current_names = [n for n, _ in modules]

    # Now the target module should be directly in the list
    target_idx = current_names.index(name)
    name, module = modules.pop(target_idx)

    return name, module, modules


def infer_auto_device_map(
    model: nn.Module,
    max_memory: Optional[dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[list[str]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    special_dtypes: Optional[dict[str, Union[str, torch.dtype]]] = None,
    verbose: bool = False,
    clean_result: bool = True,
    offload_buffers: bool = False,
    fallback_allocation: bool = False,
):
    """
    Compute a device map for a given model giving priority to GPUs, then offload on CPU and finally offload to disk,
    such that:
    - we don't exceed the memory available of any of the GPU.
    - if offload to the CPU is needed, there is always room left on GPU 0 to put back the layer offloaded on CPU that
      has the largest size.
    - if offload to the CPU is needed,we don't exceed the RAM available on the CPU.
    - if offload to the disk is needed, there is always room left on the CPU to put back the layer offloaded on disk
      that has the largest size.

    <Tip>

    All computation is done analyzing sizes and dtypes of the model parameters. As a result, the model can be on the
    meta device (as it would if initialized within the `init_empty_weights` context manager).

    </Tip>

    Args:
        model (`torch.nn.Module`):
            The model to analyze.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available if unset.
            Example: `max_memory={0: "1GB"}`.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        special_dtypes (`Dict[str, Union[str, torch.device]]`, *optional*):
            If provided, special dtypes to consider for some specific weights (will override dtype used as default for
            all weights).
        verbose (`bool`, *optional*, defaults to `False`):
            Whether or not to provide debugging statements as the function builds the device_map.
        clean_result (`bool`, *optional*, defaults to `True`):
            Clean the resulting device_map by grouping all submodules that go on the same device together.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
        fallback_allocation (`bool`, *optional*, defaults to `False`):
            When regular allocation fails, try to allocate a module that fits in the size limit using BFS.
    """

    # Initialize the variables
    (
        devices,
        max_memory,
        main_devices,
        gpus,
        module_sizes,
        tied_parameters,
        no_split_module_classes,
        modules_to_treat,
    ) = _init_infer_auto_device_map(model, max_memory, no_split_module_classes, dtype, special_dtypes)

    device_map = OrderedDict()
    current_device = 0
    device_memory_used = {device: 0 for device in devices}
    device_buffer_sizes = {}
    device_minimum_assignment_memory = {}

    # Initialize maximum largest layer, to know which space to keep in memory
    max_layer_size, max_layer_names = get_max_layer_size(modules_to_treat, module_sizes, no_split_module_classes)

    # Ready ? This is going to be a bit messy.
    while len(modules_to_treat) > 0:
        name, module = modules_to_treat.pop(0)
        if verbose:
            print(f"\nTreating module {name}.")
        # Max size in the remaining layers may have changed since we took one, so we maybe update it.
        max_layer_names = [n for n in max_layer_names if n != name and not n.startswith(name + ".")]
        if len(max_layer_names) == 0:
            max_layer_size, max_layer_names = get_max_layer_size(
                [(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)],
                module_sizes,
                no_split_module_classes,
            )
        # Assess size needed
        module_size = module_sizes[name]

        # We keep relevant tied parameters only: one of the tied parameters in the group is inside the current module
        # and the other is not.
        # Note: If we are currently processing the name `compute.weight`, an other parameter named
        # e.g. `compute.weight_submodule.parameter`
        # needs to be considered outside the current module, hence the check with additional dots.
        tied_param_groups = [
            tied_group
            for tied_group in tied_parameters
            if any(name + "." in k + "." for k in tied_group) and not all(name + "." in k + "." for k in tied_group)
        ]

        if verbose and len(tied_param_groups) > 0:
            print(f"  Found the relevant tied param groups {tied_param_groups}")

        # Then we keep track of all the parameters that are tied to the current module, but not in the current module
        tied_params = sum(
            [[p for p in tied_group if name + "." not in p + "."] for tied_group in tied_param_groups], []
        )

        if verbose and len(tied_params) > 0:
            print(f"  So those parameters need to be taken into account {tied_params}")

        device = devices[current_device]
        current_max_size = max_memory[device] if device != "disk" else None
        current_memory_reserved = 0
        # Reduce max size available by the largest layer.
        if devices[current_device] in main_devices:
            current_max_size = current_max_size - max_layer_size
            current_memory_reserved = max_layer_size

        module_size_with_ties, tied_module_names, tied_modules = get_module_size_with_ties(
            tied_params, module_size, module_sizes, modules_to_treat
        )

        # The module and its tied modules fit on the current device.
        if current_max_size is None or device_memory_used[device] + module_size_with_ties <= current_max_size:
            if verbose:
                output = f"Putting {name}"

                if tied_module_names:
                    output += f" and {tied_module_names}"
                else:
                    output += f" (size={module_size})"

                if current_max_size is not None:
                    output += f" (available={current_max_size - device_memory_used[device]})"

                output += f" on {device}."
                print(output)

            device_memory_used[device] += module_size_with_ties

            # Assign the primary module to the device.
            device_map[name] = device

            # Assign tied modules if any.
            for tied_module_name in tied_module_names:
                if tied_module_name in [m[0] for m in modules_to_treat]:
                    # Find the index of the tied module in the list
                    tied_module_index = next(i for i, (n, _) in enumerate(modules_to_treat) if n == tied_module_name)
                    # Remove the tied module from the list to prevent reprocessing
                    modules_to_treat.pop(tied_module_index)

                # Assign the tied module to the device
                device_map[tied_module_name] = device

            # Buffer Handling
            if not offload_buffers and isinstance(module, nn.Module):
                # Compute the total buffer size for the module
                current_buffer_size = compute_module_total_buffer_size(
                    module, dtype=dtype, special_dtypes=special_dtypes
                )
                # Update the buffer size on the device
                device_buffer_sizes[device] = device_buffer_sizes.get(device, 0) + current_buffer_size

            continue

        # The current module itself fits, so we try to split the tied modules.
        if len(tied_params) > 0 and device_memory_used[device] + module_size <= current_max_size:
            # can we split one of the tied modules to make it smaller or do we need to go on the next device?
            if verbose:
                print(
                    f"Not enough space on {devices[current_device]} to put {name} and {tied_module_names} (space "
                    f"available {current_max_size - device_memory_used[device]}, needed size {module_size_with_ties})."
                )
            split_happened = False
            for tied_module_name, tied_module in zip(tied_module_names, tied_modules):
                tied_module_children = list(tied_module.named_children())
                if len(tied_module_children) == 0 or tied_module.__class__.__name__ in no_split_module_classes:
                    # can't break this one.
                    continue

                if verbose:
                    print(f"Splitting {tied_module_name}.")
                tied_module_children = list(tied_module.named_parameters(recurse=False)) + tied_module_children
                tied_module_children = [(f"{tied_module_name}.{n}", v) for n, v in tied_module_children]
                tied_module_index = [i for i, (n, _) in enumerate(modules_to_treat) if n == tied_module_name][0]

                modules_to_treat = (
                    [(name, module)]
                    + modules_to_treat[:tied_module_index]
                    + tied_module_children
                    + modules_to_treat[tied_module_index + 1 :]
                )
                # Update the max layer size.
                max_layer_size, max_layer_names = get_max_layer_size(
                    [(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)],
                    module_sizes,
                    no_split_module_classes,
                )
                split_happened = True
                break

            if split_happened:
                continue

            # If the tied module is not split, we go to the next device
            if verbose:
                print("None of the tied module can be split, going to the next device.")

        # The current module itself doesn't fit, so we have to split it or go to the next device.
        if device_memory_used[device] + module_size >= current_max_size:
            # Split or not split?
            modules_children = (
                []
                if isinstance(module, nn.Parameter) or isinstance(module, torch.Tensor)
                else list(module.named_children())
            )
            if verbose:
                print(
                    f"Not enough space on {devices[current_device]} to put {name} (space available "
                    f"{current_max_size - device_memory_used[device]}, module size {module_size})."
                )
            if len(modules_children) == 0 or module.__class__.__name__ in no_split_module_classes:
                # -> no split, we go to the next device
                if verbose:
                    print("This module cannot be split, going to the next device.")

            else:
                # -> split, we replace the module studied by its children + parameters
                if verbose:
                    print(f"Splitting {name}.")
                modules_children = list(module.named_parameters(recurse=False)) + modules_children
                modules_to_treat = [(f"{name}.{n}", v) for n, v in modules_children] + modules_to_treat
                # Update the max layer size.
                max_layer_size, max_layer_names = get_max_layer_size(
                    [(n, m) for n, m in modules_to_treat if isinstance(m, torch.nn.Module)],
                    module_sizes,
                    no_split_module_classes,
                )
                continue

        # If no module is assigned to the current device, we attempt to allocate a fallback module
        # if fallback_allocation is enabled.
        if device_memory_used[device] == 0 and fallback_allocation and device != "disk":
            # We try to allocate a module that fits in the size limit using BFS.
            # Recompute the current max size as we need to consider the current module as well.
            current_max_size = max_memory[device] - max(max_layer_size, module_size_with_ties)

            fallback_module_name, fallback_module, remaining_modules = fallback_allocate(
                modules_to_treat,
                module_sizes,
                current_max_size - device_memory_used[device],
                no_split_module_classes,
                tied_parameters,
            )
            # use the next iteration to put the fallback module on the next device to avoid code duplication
            if fallback_module is not None:
                modules_to_treat = [(fallback_module_name, fallback_module)] + [(name, module)] + remaining_modules
                continue

        if device_memory_used[device] == 0:
            device_minimum_assignment_memory[device] = module_size_with_ties + current_memory_reserved

        #  Neither the current module nor any tied modules can be split, so we move to the next device.
        device_memory_used[device] = device_memory_used[device] + current_memory_reserved
        current_device += 1
        modules_to_treat = [(name, module)] + modules_to_treat

    device_memory_used = {device: mem for device, mem in device_memory_used.items() if mem > 0}

    if clean_result:
        device_map = clean_device_map(device_map)

    non_gpu_buffer_size = device_buffer_sizes.get("cpu", 0) + device_buffer_sizes.get("disk", 0)
    if non_gpu_buffer_size > 0 and not offload_buffers:
        is_buffer_fit_any_gpu = False
        for gpu_device, gpu_max_memory in max_memory.items():
            if gpu_device == "cpu" or gpu_device == "disk":
                continue

            if not is_buffer_fit_any_gpu:
                gpu_memory_used = device_memory_used.get(gpu_device, 0)

                if gpu_max_memory >= non_gpu_buffer_size + gpu_memory_used:
                    is_buffer_fit_any_gpu = True

        if len(gpus) > 0 and not is_buffer_fit_any_gpu:
            warnings.warn(
                f"Current model requires {non_gpu_buffer_size} bytes of buffer for offloaded layers, which seems does "
                f"not fit any GPU's remaining memory. If you are experiencing a OOM later, please consider using "
                f"offload_buffers=True."
            )

    if device_minimum_assignment_memory:
        devices_info = "\n".join(
            f"  - {device}: {mem} bytes required" for device, mem in device_minimum_assignment_memory.items()
        )
        logger.info(
            f"Based on the current allocation process, no modules could be assigned to the following devices due to "
            f"insufficient memory:\n"
            f"{devices_info}\n"
            f"These minimum requirements are specific to this allocation attempt and may vary. Consider increasing "
            f"the available memory for these devices to at least the specified minimum, or adjusting the model config."
        )
    return device_map


def check_device_map(model: nn.Module, device_map: dict[str, Union[int, str, torch.device]]):
    """
    Checks a device map covers everything in a given model.

    Args:
        model (`torch.nn.Module`): The model to check the device map against.
        device_map (`Dict[str, Union[int, str, torch.device]]`): The device map to check.
    """
    all_module_names = dict(model.named_modules())
    invalid_keys = [k for k in device_map if k != "" and k not in all_module_names]

    if invalid_keys:
        warnings.warn(
            f"The following device_map keys do not match any submodules in the model: {invalid_keys}", UserWarning
        )

    all_model_tensors = [name for name, _ in model.state_dict().items()]
    for module_name in device_map.keys():
        if module_name == "":
            all_model_tensors.clear()
            break
        else:
            all_model_tensors = [
                name
                for name in all_model_tensors
                if not name == module_name and not name.startswith(module_name + ".")
            ]
    if len(all_model_tensors) > 0:
        non_covered_params = ", ".join(all_model_tensors)
        raise ValueError(
            f"The device_map provided does not give any device for the following parameters: {non_covered_params}"
        )

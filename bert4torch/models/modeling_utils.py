import torch
from torch import nn
import gc
import inspect
from torch.utils.checkpoint import CheckpointFunction
from torch4keras.snippets import log_info, log_warn, log_error, is_accelerate_available, find_tied_parameters, log_warn_once
from typing import Union, Optional
from functools import partial, wraps


if is_accelerate_available():
    from accelerate.utils.modeling import (
        infer_auto_device_map, 
        get_balanced_memory, 
        check_tied_parameters_on_same_device, 
        get_max_memory,
        set_module_tensor_to_device, 
        offload_weight
    )


def cal_ts_num(tensor_shape):
    '''查看某个tensor在gc中的数量'''
    cal_num = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj): # or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensor = obj
            else:
                continue
            if tensor.is_cuda and tensor.size() == tensor_shape:
                print(tensor.shape)
                cal_num+=1
        except Exception as e:
            print('A trivial exception occured: {}'.format(e))
    print(cal_num)


def get_state_dict_dtype(state_dict:dict):
    """
    Returns the first found floating dtype in `state_dict` if there is one, otherwise returns the first dtype.
    """
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype

    # if no floating dtype was found return whatever the first dtype is
    else:
        return next(state_dict.values()).dtype


def restore_default_torch_dtype(func):
    """
    Decorator to restore the default torch dtype
    at the end of the function. Serves
    as a backup in case calling the function raises
    an error after the function has changed the default dtype but before it could restore it.
    """

    @wraps(func)
    def _wrapper(*args, **kwargs):
        old_dtype = torch.get_default_dtype()
        try:
            return func(*args, **kwargs)
        finally:
            torch.set_default_dtype(old_dtype)

    return _wrapper


def set_default_torch_dtype(torch_dtype: Union[str, torch.dtype], model_name:str='model', model_config:dict=None) -> torch.dtype:
    """设置默认权重类型"""
    if not isinstance(model_name, str):
        model_name = 'model'

    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)

    if not torch_dtype.is_floating_point:
        raise ValueError(f"Can't instantiate {model_name} under dtype={torch_dtype} since it is not a floating point dtype")
    
    # 目前的量化模型，都是float16的
    if hasattr(model_config, "quantization_config"):
        torch_dtype = torch.float16
    
    torch.set_default_dtype(torch_dtype)
    return torch_dtype


@torch.no_grad()
def load_state_dict_into_meta_model(
        model:nn.Module,
        state_dict:dict, 
        device_map:dict=None, 
        dtype:Union[str, torch.dtype]=None, 
        offload_folder=None,
        state_dict_folder=None,
        state_dict_index=None,
        is_safetensors:bool=False
):
    """ 把state_dict导入meta_model
    为了代码简洁，这里device_map需要外部手动指定, 形式如{'embeddings.word_embeddings': 0, 'LayerNormFinal': 0, 'lm_head': 0}
    """

    for param_name, param in state_dict.items():
        module_name = param_name
        set_module_kwargs = {"value": param}
        if (device_map is None) or (device_map == 'cpu'):
            param_device = "cpu"
        else:
            while len(module_name) > 0 and module_name not in device_map:
                module_name = ".".join(module_name.split(".")[:-1])
            if module_name == "" and "" not in device_map:
                # TODO: group all errors and raise at the end.
                raise ValueError(f"{param_name} doesn't have any device set.")
            param_device = device_map[module_name]

        if param_device == "disk":
            if not is_safetensors:
                offload_index = offload_weight(param, param_name, offload_folder, offload_index)
        elif param_device == "cpu" and state_dict_index is not None:
            state_dict_index = offload_weight(param, param_name, state_dict_folder, state_dict_index)
        else:
            # For backward compatibility with older versions of `accelerate` and for non-quantized params
            set_module_kwargs["dtype"] = dtype or param.dtype
            try:
                set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs)
            except ValueError as e:
                # 如果报错，加上参数名称，方便debug
                e.args = (f'Parameter `{param_name}`: ' + e.args[-1], )
                raise e


def get_device_map(pretrained_model, device_map, torch_dtype, **kwargs):
    '''获取合适的device_map'''
    max_memory = kwargs.pop('max_memory', None)
    if isinstance(device_map, torch.device):
        device_map = {"": device_map}
    elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
        try:
            device_map = {"": torch.device(device_map)}
        except RuntimeError:
            raise ValueError(
                "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
            )
    elif isinstance(device_map, int):
        if device_map < 0:
            raise ValueError(
                "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
            )
        else:
            device_map = {"": device_map}

    if not is_accelerate_available():
        log_warn_once('Package `accelerate` not available, use `pip install accelerate`')
        return device_map

    if isinstance(device_map, str):
        special_dtypes = {}

        # TODO: keep_in_fp32_modules
        keep_in_fp32_modules = []
        special_dtypes.update(
            {
                name: torch.float32
                for name, _ in pretrained_model.named_parameters()
                if any(m in name for m in keep_in_fp32_modules)
            }
        )

        target_dtype = torch_dtype

        no_split_modules = pretrained_model._get_no_split_modules(device_map)
        if device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            raise ValueError(
                "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                "'sequential'."
            )

        device_map_kwargs = {"no_split_module_classes": no_split_modules}
        if "special_dtypes" in inspect.signature(infer_auto_device_map).parameters:
            device_map_kwargs["special_dtypes"] = special_dtypes
        elif len(special_dtypes) > 0:
            log_warn(
                "This model has some weights that should be kept in higher precision, you need to upgrade "
                "`accelerate` to properly deal with them (`pip install --upgrade accelerate`)."
            )
        if device_map != "sequential":
            max_memory = get_balanced_memory(
                pretrained_model,
                dtype=target_dtype,
                low_zero=(device_map == "balanced_low_0"),
                max_memory=max_memory,
                **device_map_kwargs,
            )
        else:
            max_memory = get_max_memory(max_memory)
        device_map_kwargs["max_memory"] = max_memory

        # Make sure tied weights are tied before creating the device map.
        pretrained_model.tie_weights()
        device_map = infer_auto_device_map(pretrained_model, dtype=target_dtype, **device_map_kwargs)

    elif device_map is not None:
        pretrained_model.tie_weights()
        tied_params = find_tied_parameters(pretrained_model)
        # check if we don't have tied param in different devices
        check_tied_parameters_on_same_device(tied_params, device_map)
    return device_map


def old_checkpoint(function, model_kwargs):
    ''' 兼容torch<1.11.0时仅允许输入输出是位置参数
    通过闭包来对返回参数进行控制
    '''

    def create_custom_forward(module):
        def custom_forward(*inputs):
            outputs = module(*inputs)
            if isinstance(outputs, dict):
                setattr(create_custom_forward, 'outputs_keys', [v for v in outputs.keys()])
                return tuple(outputs.values())
            else:
                return outputs
        return custom_forward
    
    args = []
    __args = inspect.getargspec(type(function).forward)
    arg_names, arg_defaults = __args[0][1:], __args[-1]
    for i, arg_name in enumerate(arg_names):
        args.append(model_kwargs.get(arg_name, arg_defaults[i]))

    preserve = model_kwargs.pop('preserve_rng_state', True)

    outputs = CheckpointFunction.apply(create_custom_forward(function), preserve, *args)
    if hasattr(create_custom_forward, 'outputs_keys'):
        return dict(zip(create_custom_forward.outputs_keys, outputs))
    else:
        return outputs



def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def peft_module_casting_to_bf16(model):
    from peft.tuners.tuners_utils import BaseTunerLayer

    for name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            module = module.to(torch.bfloat16)
        elif isinstance(module, torch.nn.LayerNorm) or "norm" in name:
            module = module.to(torch.float32)
        elif any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)


def get_layers(module: nn.Module, layers=[nn.Conv2d, nn.Linear], prefix: Optional[str] = None, name: str = ""):
    """
    Get all the layers with a specific prefix in the module
    Args:
        module (`nn.Module`):
            The module that contains our layers
        layers (`list`, defaults to `[Conv1D, nn.Conv2d, nn.Linear]`):
            Type of the layers that we want to get
        prefix (`Optional[str]`, defaults to `None`):
            Prefix of layers
        name (`str`, defaults to `""`):
            Used for recursion. Don't modify

    Returns:
        `Dict[str,Union[Conv1D, nn.Conv2d, nn.Linear]]`: Mapping of the name of the layer and the actual layer
    """
    for layer in layers:
        if isinstance(module, layer):
            if prefix is not None:
                if name.startswith(prefix):
                    return {name: module}
            else:
                return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(get_layers(child, layers=layers, prefix=prefix, name=name + "." + name1 if name != "" else name1))
    return res


def find_all_linear_names(peft_model, int4=False, int8=False, ignore_names=['lm_head', 'output_layer']):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if any([True if key in name else False for key in ignore_names]):
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)
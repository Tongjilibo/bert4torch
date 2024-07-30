#! -*- coding: utf-8 -*-
'''工具函数
'''
import torch
from torch import nn
import gc
import inspect
from torch.utils.checkpoint import CheckpointFunction
from torch4keras.snippets import log_info, log_warn, Timeit
from typing import Union
import re


def insert_arguments(**arguments):
    """装饰器，为类方法增加参数（主要用于类的__init__方法）"""
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k, v in arguments.items():
                if k in kwargs:
                    v = kwargs.pop(k)
                setattr(self, k, v)
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


def delete_arguments(*arguments):
    """装饰器，为类方法删除参数（主要用于类的__init__方法）"""
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k in arguments:
                if k in kwargs:
                    raise TypeError(
                        '%s got an unexpected keyword argument \'%s\'' %
                        (self.__class__.__name__, k)
                    )
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator


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


def set_default_torch_dtype(dtype: Union[str, torch.dtype], model_name:str='model') -> torch.dtype:
    """设置默认权重类型"""
    if not isinstance(model_name, str):
        model_name = 'model'
    mapping = {
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'bfloat16': torch.bfloat16
        }
    if isinstance(dtype, str):
        dtype = mapping[dtype]

    if not dtype.is_floating_point:
        raise ValueError(f"Can't instantiate {model_name} under dtype={dtype} since it is not a floating point dtype")
    dtype_orig = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    if dtype_orig != dtype:
        log_info(f"Instantiating {model_name} under default dtype {dtype}.")
    return dtype, dtype_orig


def load_state_dict_into_meta_model(model:nn.Module, state_dict:dict, device_map:dict=None, torch_dtype:Union[str, torch.dtype]=None):
    """ 把state_dict导入meta_model
    为了代码简洁，这里device_map需要外部手动指定, 形式如{'embeddings.word_embeddings': 0, 'LayerNormFinal': 0, 'lm_head': 0}
    """

    from accelerate.utils import set_module_tensor_to_device
    for param_name, param in state_dict.items():
        set_module_kwargs = {"value": param}
        if (device_map is None) or (device_map == 'cpu'):
            param_device = "cpu"
        elif device_map == 'auto':
            param_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device_map in {'gpu', 'cuda'}:
            param_device = 'cuda'
        elif isinstance(device_map, torch.device) or isinstance(device_map, int):
            param_device = device_map
        elif isinstance(device_map, dict):
            param_device = device_map[param_name]
        else:
            param_device = 'cpu'
            log_warn(f'Args `device_map`={device_map} has not been pre maintained')

        set_module_kwargs["dtype"] = torch_dtype or param.dtype
        set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs)


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


class GenerateSpeed(Timeit):
    '''上下文管理器，计算token生成的速度

    Examples:
    ```python
    >>> from bert4torch.snippets import GenerateSpeed
    >>> with GenerateSpeed() as gs:
    ...     response = model.generate(query, **generation_config)
    ...     tokens_len = len(tokenizer(response, return_tensors='pt')['input_ids'][0])
    ...     gs(tokens_len)
    ```
    '''
    def __enter__(self):
        super().__enter__()
        self.template = 'Generate speed: {:.2f} token/s'
        return self


def modify_variable_mapping(original_func, **new_dict):
    '''对variable_mapping的返回值（字典）进行修改
    '''
    def wrapper(*args, **kwargs):
        # 调用原始函数并获取结果
        result = original_func(*args, **kwargs)
        
        # 对返回值进行修改
        result.update(new_dict)
        return result
    
    return wrapper


def get_weight_decay_optim_groups(module:nn.Module, weight_decay:float) -> dict:
    '''获取weight_decay的参数列表'''
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in module.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    log_info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    log_info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    return optim_groups


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


def has_chinese_char(text:str) -> bool:
    '''判断一句话中是否包含中文'''
    if text is None:
        return False
    elif text.strip() == '':
        return False
    return re.search(r'[\u4e00-\u9fff]', text)

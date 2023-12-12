#! -*- coding: utf-8 -*-
'''工具函数
'''

import torch
import gc
import importlib
import inspect
from packaging import version
from torch4keras.snippets import *
from torch.utils.checkpoint import CheckpointFunction
import sys
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


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


def set_default_torch_dtype(dtype: torch.dtype, model_name='model') -> torch.dtype:
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
    log_info(f"Instantiating {model_name} under default dtype {dtype}.")
    dtype_orig = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    return dtype, dtype_orig


def is_accelerate_available(check_partial_state=False):
    accelerate_available = importlib.util.find_spec("accelerate") is not None
    if accelerate_available:
        if check_partial_state:
            return version.parse(importlib_metadata.version("accelerate")) >= version.parse("0.17.0")
        else:
            return True
    else:
        return False


def load_state_dict_into_meta_model(model, state_dict, device_map=None, torch_dtype=None):
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
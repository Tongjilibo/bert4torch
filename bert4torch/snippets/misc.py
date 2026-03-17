#! -*- coding: utf-8 -*-
'''工具函数
'''

from torch4keras.snippets import log_info, log_warn, log_error, TimeitContextManager
from typing import Union, Dict, Type, Callable
import re
from io import BytesIO
import requests
from PIL import Image
import base64
import numpy as np
import os
import functools
import types
from .import_utils import (
    is_torch_available,
    
)

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


class GenerateSpeed(TimeitContextManager):
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


def has_chinese_char(text:str) -> bool:
    '''判断一句话中是否包含中文'''
    if text is None:
        return False
    elif text.strip() == '':
        return False
    return re.search(r'[\u4e00-\u9fff]', text)


def load_image(image: Union[Image.Image, np.ndarray, str]) -> Image.Image:
    '''加载图片'''
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, np.ndarray):
        image_obj = Image.fromarray(image)
    elif isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):  # 网址
            image_obj = Image.open(requests.get(image, stream=True).raw)
        elif image.startswith("file://"):  # 网盘
            image_obj = Image.open(image[7:])
        elif image.startswith("data:image"):  # base64编码
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                image_obj = Image.open(BytesIO(data))
        elif os.path.isfile(image):  # 本地文件路径
            image_obj = Image.open(image)

    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64, np.ndarray and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    return image


def create_registrar(target_map: Dict[str, Type]) -> Callable:
    """
    注册器工厂函数：根据目标映射字典创建对应的注册装饰器
    
    参数:
        target_map: 要注册类的目标字典（如MLP_MAP、ATTN_MAP）
    
    返回:
        专用的注册装饰器函数
    """
    def register(cls=None, name: str = None):
        """
        为指定映射字典服务的注册装饰器
        
        参数:
            cls: 要注册的类（装饰器自动传入）
            name: 可选，自定义注册到字典中的key，默认使用类名
        """
        def decorator(cls):
            # 确定注册的key，优先使用自定义name，否则使用类名
            register_name = name or cls.__name__
            # 将类注册到指定的目标字典中
            target_map[register_name] = cls
            # 返回原类，不改变类的功能
            return cls
        
        # 处理两种使用方式：@register 或 @register(name="CustomName")
        if cls is not None:
            return decorator(cls)
        return decorator
    
    return register


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
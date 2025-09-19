#! -*- coding: utf-8 -*-
'''工具函数
'''

import torch
from torch import nn
from torch4keras.snippets import log_info, log_warn, log_error, TimeitContextManager
from typing import Union, Optional
import re
from packaging import version
from io import BytesIO
import requests
from PIL import Image
import base64
import numpy as np
import os


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


def has_meta_param(model:nn.Module, verbose:bool=False):
    '''是否有meta的param'''
    meta_names = [name_ for name_, para_ in model.named_parameters() if para_.device == torch.device('meta')]

    if len(meta_names) > 0:
        if verbose:
            log_error(f'Meta device not allowed: {meta_names}')
        return True
    return False


if version.parse(torch.__version__) >= version.parse("1.10.0"):
    inference_mode = torch.inference_mode
else:
    inference_mode = torch.no_grad
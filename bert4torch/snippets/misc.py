#! -*- coding: utf-8 -*-
'''工具函数
'''

import json
import torch
import gc
import inspect
import os
from torch.utils.checkpoint import CheckpointFunction
import shutil
import re
from torch4keras.snippets import log_info, log_warn, Timeit


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
    dtype_orig = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    if dtype_orig != dtype:
        log_info(f"Instantiating {model_name} under default dtype {dtype}.")
    return dtype, dtype_orig


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


def cuda_empty_cache(device=None):
    '''清理gpu显存'''
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            return
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class GenerateSpeed(Timeit):
    '''上下文管理器，计算token生成的速度

    Example
    -----------------------------------------------------
    >>> from bert4torch.snippets import GenerateSpeed
    >>> with GenerateSpeed() as gs:
    >>>     response = model.generate(query, **generation_config)
    >>>     tokens_len = len(tokenizer(response, return_tensors='pt')['input_ids'][0])
    >>>     gs(tokens_len)
    '''
    def __enter__(self):
        super().__enter__()
        self.template = 'Generate speed: {:.2f} token/s'
        return self


class WebServing(object):
    """简单的Web接口，基于bottlepy简单封装，仅作为临时测试使用，不保证性能。

    Example:
        >>> arguments = {'text': (None, True), 'n': (int, False)}
        >>> web = WebServing(port=8864)
        >>> web.route('/gen_synonyms', gen_synonyms, arguments)
        >>> web.start()
        >>> # 然后访问 http://127.0.0.1:8864/gen_synonyms?text=你好
    
    依赖（如果不用 server='paste' 的话，可以不装paste库）:
        >>> pip install bottle
        >>> pip install paste
    """
    def __init__(self, host='0.0.0.0', port=8000, server='paste'):

        import bottle

        self.host = host
        self.port = port
        self.server = server
        self.bottle = bottle

    def wraps(self, func, arguments, method='GET'):
        """封装为接口函数

        :param func: 要转换为接口的函数，需要保证输出可以json化，即需要保证 json.dumps(func(inputs)) 能被执行成功；
        :param arguments: 声明func所需参数，其中key为参数名，value[0]为对应的转换函数（接口获取到的参数值都是字符串型），value[1]为该参数是否必须；
        :param method: 'GET'或者'POST'。
        """
        def new_func():
            outputs = {'code': 0, 'desc': u'succeeded', 'data': {}}
            kwargs = {}
            for key, value in arguments.items():
                if method == 'GET':
                    result = self.bottle.request.GET.getunicode(key)
                else:
                    result = self.bottle.request.POST.getunicode(key)
                if result is None:
                    if value[1]:
                        outputs['code'] = 1
                        outputs['desc'] = 'lack of "%s" argument' % key
                        return json.dumps(outputs, ensure_ascii=False)
                else:
                    if value[0] is not None:
                        result = value[0](result)
                    kwargs[key] = result
            try:
                outputs['data'] = func(**kwargs)
            except Exception as e:
                outputs['code'] = 2
                outputs['desc'] = str(e)
            return json.dumps(outputs, ensure_ascii=False)

        return new_func

    def route(self, path, func, arguments, method='GET'):
        """添加接口"""
        func = self.wraps(func, arguments, method)
        self.bottle.route(path, method=method)(func)

    def start(self):
        """启动服务"""
        self.bottle.run(host=self.host, port=self.port, server=self.server)


class AnyClass:
    '''主要用于import某个包不存在时候，作为类的替代'''
    def __init__(self, *args, **kwargs) -> None:
        pass


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


def copytree(src:str, dst:str, ignore_copy_files:str=None, dirs_exist_ok=False):
    '''从一个文件夹copy到另一个文件夹
    
    :param src: str, copy from src
    :param dst: str, copy to dst
    '''
    def _ignore_copy_files(path, content):
        to_ignore = []
        if ignore_copy_files is None:
            return to_ignore
        
        for file_ in content:
            for pattern in ignore_copy_files:
                if re.search(pattern, file_):
                    to_ignore.append(file_)
        return to_ignore

    if src:
        os.makedirs(src, exist_ok=True)
    shutil.copytree(src, dst, ignore=_ignore_copy_files, dirs_exist_ok=dirs_exist_ok)


def get_weight_decay_optim_groups(self, weight_decay:float) -> dict:
    '''获取weight_decay的参数列表'''
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in self.named_parameters()}
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

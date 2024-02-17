#! -*- coding: utf-8 -*-
'''工具函数
'''

import json
import torch
import gc
import inspect
from torch4keras.snippets import *
from torch.utils.checkpoint import CheckpointFunction
import shutil
import re
from pathlib import Path

if os.environ.get('SAFETENSORS_FIRST', False):
    SAFETENSORS_BINS = ['.safetensors', '.bin']  # 优先查找safetensors格式权重
else:
    SAFETENSORS_BINS = ['.bin', '.safetensors']  # 优先查找bin格式权重

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


def get_state_dict_dtype(state_dict):
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
    from bert4torch.snippets import GenerateSpeed
    with GenerateSpeed() as gs:
        response = model.generate(query, **generation_config)
        tokens_len = len(tokenizer(response, return_tensors='pt')['input_ids'][0])
        gs(tokens_len)
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


def try_to_load_from_cache(
    repo_id: str,
    filename: str,
    cache_dir: Union[str, Path, None] = None,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
) -> Optional[str]:
    """
    Explores the cache to return the latest cached file for a given revision if found.

    This function will not raise any exception if the file in not cached.

    Args:
        cache_dir (`str` or `os.PathLike`):
            The folder where the cached files lie.
        repo_id (`str`):
            The ID of the repo on huggingface.co.
        filename (`str`):
            The filename to look for inside `repo_id`.
        revision (`str`, *optional*):
            The specific model version to use. Will default to `"main"` if it's not provided and no `commit_hash` is
            provided either.
        repo_type (`str`, *optional*):
            The type of the repo.

    Returns:
        `Optional[str]` or `_CACHED_NO_EXIST`:
            Will return `None` if the file was not cached. Otherwise:
            - The exact path to the cached file if it's found in the cache
            - A special value `_CACHED_NO_EXIST` if the file does not exist at the given commit hash and this fact was
              cached.
    """
    if revision is None:
        revision = "main"

    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE

    object_id = repo_id.replace("/", "--")
    if repo_type is None:
        repo_type = "model"
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    if not os.path.isdir(repo_cache):
        # No cache for this model
        return None
    for subfolder in ["refs", "snapshots"]:
        if not os.path.isdir(os.path.join(repo_cache, subfolder)):
            return None

    # Resolve refs (for instance to convert main to the associated commit sha)
    cached_refs = os.listdir(os.path.join(repo_cache, "refs"))
    if revision in cached_refs:
        with open(os.path.join(repo_cache, "refs", revision)) as f:
            revision = f.read()

    if os.path.isfile(os.path.join(repo_cache, ".no_exist", revision, filename)):
        return object()

    cached_shas = os.listdir(os.path.join(repo_cache, "snapshots"))
    if revision not in cached_shas:
        # No cache for this revision and we won't try to return a random revision
        return None

    cached_file = os.path.join(repo_cache, "snapshots", revision, filename)
    return cached_file if os.path.isfile(cached_file) else None


def snapshot_download(
    repo_id: str,
    filename: str = None,
    revision: str = None,
    cache_dir: Union[str, Path, None] = None,
    library_name: str = None,
    library_version: str = None,
    user_agent: Union[Dict, str, None] = None
) -> str:
    """
    Download pretrained model from https://huggingface.co/
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    repo_cache = os.path.join(cache_dir, f'models--{repo_id.replace("/", "--")}')

    storage_folder = None
    if filename is None:
        # 下载repo下所有文件
        bert4torch_filenames = os.path.join(repo_cache, 'bert4torch_filenames.json')
        if os.path.exists(bert4torch_filenames):
            file_names = json.load(open(bert4torch_filenames, "r", encoding='utf-8'))
        else:
            model_info = HfApi().model_info(repo_id=repo_id, revision=revision)
            file_names = []
            for model_file in model_info.siblings:
                file_name = model_file.rfilename
                if file_name.endswith(".h5") or file_name.endswith(".ot") or file_name.endswith(".msgpack"):
                    continue
                file_names.append(file_name)
            # 仅下载safetensors格式的
            if any([i.endswith('.safetensors') for i in file_names]) and is_safetensors_available():
                file_names = [i for i in file_names if not i.endswith('.bin')]
            else:  # 仅下载pytorch_model_*.bin
                file_names = [i for i in file_names if not i.endswith('.safetensors')]
            json.dump(file_names, open(bert4torch_filenames, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

        for file_name in file_names:           
            # 从cache中恢复
            resolved_file = try_to_load_from_cache(repo_id, file_name, cache_dir=cache_dir)
            if resolved_file is not None:
                if resolved_file is object():
                    raise EnvironmentError(f"Could not locate {file_name}.")
                elif resolved_file.endswith('config.json'):
                    storage_folder = os.path.dirname(resolved_file)
                    log_info_once(f'Resume {repo_id} from {storage_folder}')
            else:
                # 下载指定文件
                resolved_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_name,
                    cache_dir=cache_dir,
                    # force_filename=filename,
                    library_name=library_name,
                    library_version=library_version,
                    user_agent=user_agent,
                )
                if resolved_file.endswith('config.json'):
                    storage_folder = os.path.dirname(resolved_file)
                    log_info(f'Download {repo_id} to {storage_folder}')
            if os.path.exists(resolved_file + ".lock"):
                os.remove(resolved_file + ".lock")
        return storage_folder
    else:
        # 从cache中恢复
        resolved_file = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir)
        if resolved_file is not None:
            if resolved_file is object():
                raise EnvironmentError(f"Could not locate {filename}.")
            else:
                log_info(f'Resume {repo_id} from {resolved_file}')
        else:
            # 下载指定文件
            resolved_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                # force_filename=filename,
                library_name=library_name,
                library_version=library_version,
                user_agent=user_agent,
            )
            log_info(f'Download {repo_id} to {resolved_file}')
        return resolved_file


def get_local_config_path(model_dir:str, allow_none=False):
    '''获取local文件夹下的config文件路径'''
    config_path = None
    # 文件
    if os.path.isfile(model_dir):
        if model_dir.endswith('config.json'):
            return model_dir
        else:
            model_dir = os.path.dirname(model_dir)

    # 文件夹
    if os.path.isdir(model_dir):
        for _config in ['bert4torch_config.json', 'config.json']:
            config_path = os.path.join(model_dir, _config)
            if os.path.exists(config_path):
                break
        if (not allow_none) and (config_path is None):
            raise FileNotFoundError('bert4torch_config.json or config.json not found')
    return config_path


def get_local_checkpoint_path(model_dir:str, verbose=1) -> list:
    '''获取该local文件夹下的ckpt文件、文件列表'''
    # 文件
    if os.path.isfile(model_dir):
        if model_dir.endswith('.bin') or model_dir.endswith('.safetensors'):
            return model_dir
        else:
            model_dir = os.path.dirname(model_dir)

    # 文件夹
    if os.path.isdir(model_dir):
        for postfix in SAFETENSORS_BINS:
            ckpt_names = [i for i in os.listdir(model_dir) if i.endswith(postfix)]
            if len(ckpt_names) > 0:
                model_dir = [os.path.join(model_dir, i) for i in os.listdir(model_dir) if i.endswith(postfix)]
                break
        if len(ckpt_names) == 0:
            raise FileNotFoundError(f'No weights found in {model_dir}')
        if verbose:
            # 仅传入文件夹时候打印权重列表，因为如果指定单个文件或者文件列表，在外面已经可以查看了
            log_info(f"Load model weights from {ckpt_names}")
    return model_dir


def check_checkpoint_config_path(checkpoint_path, config_path):
    '''修正checkpint_path和config_path
    1. model_name: 从hf下载
    2. local_file且config_path为None: 重新在local_file所在目录找对应的config_path
    3. local_dir且config_path为None: 重新在local_dir找对应的config_path
    '''
    if checkpoint_path is None:
        pass

    elif isinstance(checkpoint_path, str):
        if os.path.isfile(checkpoint_path):
            # 本地文件
            config_dir = os.path.dirname(checkpoint_path)
        elif os.path.isdir(checkpoint_path):
            # 本地文件夹
            config_dir = checkpoint_path
        else:  # model_name
            # 从hf下载bert4torch_config.json文件
            filename = checkpoint_path.split('/')[-1] + '/bert4torch_config.json'
            config_dir = snapshot_download('Tongjilibo/bert4torch_config', filename=filename)
            config_dir = config_dir or checkpoint_path  # 如果未维护使用config.json即可
            # 从hf下载模型
            checkpoint_path = snapshot_download(checkpoint_path)
        config_path = get_local_config_path(config_dir, allow_none=True) if config_path is None else None

    elif isinstance(checkpoint_path, (tuple,list)):
        config_dir = os.path.dirname(checkpoint_path[0])
        config_path = get_local_config_path(config_dir, allow_none=True) if config_path is None else None

    return checkpoint_path, config_path
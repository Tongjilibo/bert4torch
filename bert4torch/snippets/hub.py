#! -*- coding: utf-8 -*-
'''下载权重和bert4torch_config.json
'''

import os
import json
from pathlib import Path
from typing import Union, Optional, Dict
import re
from torch4keras.snippets import (
    log_error_once, 
    log_error, 
    log_info_once, 
    log_warn,
    is_safetensors_available, 
    check_file_modified,
    check_url_available_cached
)


if os.environ.get('SAFETENSORS_FIRST', False):
    SAFETENSORS_BINS = ['.safetensors', '.bin']  # 优先查找safetensors格式权重
else:
    SAFETENSORS_BINS = ['.bin', '.safetensors']  # 优先查找bin格式权重
_CACHED_NO_EXIST = object()


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
        return _CACHED_NO_EXIST

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
    user_agent: Union[Dict, str, None] = None,
    **kwargs
) -> str:
    """
    Download pretrained model from huggingface
    """
    _commit_hash = kwargs.get('_commit_hash', None)
    force_download = kwargs.get('force_download', False)
    local_files_only = kwargs.get('local_files_only', False)

    # 设置下载的镜像url
    default_endpoint, alternate_endpoint = "https://huggingface.co", "https://hf-mirror.com"
    if os.environ.get('HF_ENDPOINT') is not None:
        # 用户指定
        endpoint = os.environ.get('HF_ENDPOINT')
    elif check_url_available_cached(default_endpoint):
        # https://huggingface.co 可以访问
        endpoint = default_endpoint
    elif check_url_available_cached(alternate_endpoint):
        # https://hf-mirror.com 可以访问
        endpoint = alternate_endpoint
    else:
        log_error_once(f'Check your network can access "{default_endpoint}" or "{alternate_endpoint}"')
        endpoint = default_endpoint
    os.environ['HF_ENDPOINT'] = endpoint

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    repo_cache = os.path.join(cache_dir, f'models--{repo_id.replace("/", "--")}')
    
    storage_folder = None
    # 下载repo下所有文件
    if filename is None:
        b4t_filenames_path = os.path.join(repo_cache, 'bert4torch_filenames.json')
        if os.path.exists(b4t_filenames_path) and local_files_only:
            # 从本地bert4torch_filenames.json加载所需要的文件列表
            file_names = json.load(open(b4t_filenames_path, "r", encoding='utf-8'))
        else:
            # web获取需要的文件列表
            model_info = HfApi(endpoint=endpoint).model_info(repo_id=repo_id, revision=revision)
            file_names = []
            for model_file in model_info.siblings:
                file_name = model_file.rfilename
                if file_name.endswith(".h5") or file_name.endswith(".ot") or file_name.endswith(".msgpack"):
                    continue
                file_names.append(file_name)
            
            # 仅下载safetensors格式的
            if any([i.endswith('.safetensors') for i in file_names]) and is_safetensors_available():
                file_names = [i for i in file_names if not i.endswith('.bin')]
            # 仅下载pytorch_model_*.bin
            else:
                file_names = [i for i in file_names if not i.endswith('.safetensors')]
            os.makedirs(os.path.dirname(b4t_filenames_path), exist_ok=True)
            json.dump(file_names, open(b4t_filenames_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

        for file_name in file_names:
            # 从cache中恢复
            if (_commit_hash is not None and not force_download) or local_files_only:
                resolved_file = try_to_load_from_cache(repo_id, file_name, cache_dir=cache_dir, revision=_commit_hash)
                if resolved_file is not None:
                    if resolved_file is _CACHED_NO_EXIST:
                        log_error_once(f"Could not locate {filename} inside https://huggingface.co/{repo_id}/tree/main")
                    elif resolved_file.endswith('config.json'):
                        storage_folder = os.path.dirname(resolved_file)
                        log_info_once(f'Resume {repo_id} from {storage_folder}')
            else:
                # 下载指定文件
                try:
                    resolved_file = hf_hub_download(
                        repo_id = repo_id,
                        filename = file_name,
                        cache_dir = cache_dir,
                        revision = revision,
                        force_download = force_download,
                        # force_filename = filename,
                        library_name = library_name,
                        library_version = library_version,
                        user_agent = user_agent,
                        endpoint = endpoint 
                    )
                except EntryNotFoundError as e:
                    log_error(f'Download {repo_id} {file_name} failed')
                    raise EntryNotFoundError(e)
                
                if resolved_file.endswith('config.json'):
                    storage_folder = os.path.dirname(resolved_file)
                    if check_file_modified(resolved_file, duration=2):
                        # 如果文件在2s内下载的，则不打印
                        log_info_once(f'Download {repo_id} to {storage_folder}')
            if os.path.exists(resolved_file + ".lock"):
                os.remove(resolved_file + ".lock")
        return storage_folder

    # 下载指定文件
    else:
        # 从cache中恢复
        if (_commit_hash is not None and not force_download) or local_files_only:
            resolved_file = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir, revision=_commit_hash)
            if resolved_file is not None:
                if resolved_file is _CACHED_NO_EXIST:
                    log_error_once(f"Could not locate {filename} inside https://huggingface.co/{repo_id}/tree/main")
                    resolved_file = None
                else:
                    log_info_once(f'Resume {repo_id} from {resolved_file}')
        else:
            # 下载指定文件
            try:
                resolved_file = hf_hub_download(
                    repo_id = repo_id,
                    filename = filename,
                    cache_dir = cache_dir,
                    revision = revision,
                    force_download = force_download,
                    # force_filename = filename,
                    library_name = library_name,
                    library_version = library_version,
                    user_agent = user_agent,
                    endpoint = endpoint 
                )
                if check_file_modified(resolved_file, duration=2):
                    # 如果文件在2s内下载的，则不打印
                    log_info_once(f'Download {repo_id} to {resolved_file}')
            except EntryNotFoundError:
                log_error_once(f'Check your network can access "{default_endpoint}" or "{alternate_endpoint}"')
                log_error(f"Check '{filename}' exists in 'https://huggingface.co/{repo_id}/tree/main'.")
                resolved_file = None
        return resolved_file


def get_config_path(pretrained_model_name_or_path:str, allow_none=False, **kwargs) -> str:
    '''获取local文件夹下的config文件路径
    
    :param pretrained_model_name_or_path: str, 预训练权重的本地路径，或者是model_name
        1. model_name: 从hf下载
        2. local_file且config_path为None: 重新在local_file所在目录找对应的config_path
        3. local_dir且config_path为None: 重新在local_dir找对应的config_path
    :param allow_none: bool, 是否允许本地找不到config文件
    '''
    if pretrained_model_name_or_path is None:
        return pretrained_model_name_or_path
    elif isinstance(pretrained_model_name_or_path, (tuple,list)):
        pretrained_model_name_or_path = os.path.dirname(pretrained_model_name_or_path[0])

    config_path = None       
    # 传入bert4torch_config路径
    if pretrained_model_name_or_path.endswith('bert4torch_config.json'):
        if os.path.isfile(pretrained_model_name_or_path):
            # 本地存在bert4torch_config.json
            config_path = pretrained_model_name_or_path
        elif not allow_none:
            raise FileNotFoundError(f'{pretrained_model_name_or_path} not exists, please check your local path or model_name.')

    # 传入文件夹路径
    elif os.path.isdir(pretrained_model_name_or_path):
        config_path_tmp = os.path.join(pretrained_model_name_or_path, 'bert4torch_config.json')
        if os.path.isfile(config_path_tmp):
            # bert4torch_config.json存在
            config_path = config_path_tmp
        elif not allow_none:
            raise FileNotFoundError(f'{config_path_tmp} not exists, please check your local path or model_name.')

    # model_name: 从hf下载bert4torch_config.json文件
    elif len(re.findall('/', pretrained_model_name_or_path)) <= 1 and not re.search(r'\\', pretrained_model_name_or_path):
        if pretrained_model_name_or_path.startswith('Tongjilibo/'):
            # 独立的repo
            config_path = snapshot_download(pretrained_model_name_or_path, filename='bert4torch_config.json', **kwargs)
        else:
            # 单独下载bert4torch_config.json文件
            filename = '/'.join(pretrained_model_name_or_path.split('/')[-2:]) + '/bert4torch_config.json'
            config_path = snapshot_download('Tongjilibo/bert4torch_config', filename=filename, **kwargs)

    return config_path


def get_checkpoint_path(pretrained_model_name_or_path:Union[str,list], **kwargs) -> Union[str,list]:
    '''获取该local文件夹下的ckpt文件、文件列表
    :param pretrained_model_name_or_path: str, 预训练权重的本地路径，或者是model_name
        1. model_name: 从hf下载
        2. local_file且config_path为None: 重新在local_file所在目录找对应的config_path
        3. local_dir且config_path为None: 重新在local_dir找对应的config_path
    '''
    if pretrained_model_name_or_path is None:
        return pretrained_model_name_or_path
    
    # 文件列表
    elif isinstance(pretrained_model_name_or_path, (tuple,list)):
        return pretrained_model_name_or_path

    # 文件
    elif os.path.isfile(pretrained_model_name_or_path):
        if pretrained_model_name_or_path.endswith('.bin') or pretrained_model_name_or_path.endswith('.safetensors'):
            return pretrained_model_name_or_path
        else:
            pretrained_model_name_or_path = os.path.dirname(pretrained_model_name_or_path)

    # model_name: 从hf下载模型
    elif not os.path.isdir(pretrained_model_name_or_path):
        pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path, **kwargs)
    
    # 文件夹
    if os.path.isdir(pretrained_model_name_or_path):
        for postfix in SAFETENSORS_BINS:
            ckpt_names = [i for i in os.listdir(pretrained_model_name_or_path) if i.endswith(postfix)]
            if len(ckpt_names) > 0:
                pretrained_model_name_or_path = [os.path.join(pretrained_model_name_or_path, i) \
                                                 for i in os.listdir(pretrained_model_name_or_path) if i.endswith(postfix)]
                break
        if len(ckpt_names) == 0:
            raise FileNotFoundError(f'No weights found in {pretrained_model_name_or_path}')    
    return pretrained_model_name_or_path


def set_hf_endpoint(url="https://hf-mirror.com"):
    '''使用huggingface国内镜像'''
    os.environ['HF_ENDPOINT'] = url

#! -*- coding: utf-8 -*-
'''下载权重和bert4torch_config.json
'''

import os
import json
from pathlib import Path
from typing import Union, Optional, Dict, Iterable
import re
from . import logging
from requests.exceptions import HTTPError
import warnings
from torch4keras.snippets import (
    log_error_once, 
    log_error, 
    log_info_once, 
    log_warn,
    is_safetensors_available, 
    check_file_modified,
    check_url_available_cached
)
import tempfile
from .import_utils import ENV_VARS_TRUE_VALUES


LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE = "chat_template.json"
CHAT_TEMPLATE_FILE = "chat_template.jinja"
CHAT_TEMPLATE_DIR = "additional_chat_templates"
BERT4TORCH_CONFIG_NAME = "bert4torch_config.json"


if os.environ.get('SAFETENSORS_FIRST', False):
    SAFETENSORS_BINS = ['.safetensors', '.bin']  # 优先查找safetensors格式权重
else:
    SAFETENSORS_BINS = ['.bin', '.safetensors']  # 优先查找bin格式权重
_CACHED_NO_EXIST = object()


# huggingface_hub的一些设置
default_home = os.path.join(os.path.expanduser("~"), ".cache")
HF_HOME = os.path.expandvars(
    os.path.expanduser(
        os.getenv(
            "HF_HOME",
            os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "huggingface"),
        )
    )
)

HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(HF_HOME, "modules"))
TRANSFORMERS_DYNAMIC_MODULE_NAME = "transformers_modules"


default_cache_path = os.path.join(HF_HOME, "hub")
default_assets_cache_path = os.path.join(HF_HOME, "assets")
# Legacy env variables
HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", default_cache_path)


def _is_true(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.upper() in ENV_VARS_TRUE_VALUES

HF_HUB_CACHE = os.path.expandvars(
    os.path.expanduser(
        os.getenv(
            "HF_HUB_CACHE",
            HUGGINGFACE_HUB_CACHE,
        )
    )
)
HF_HUB_OFFLINE = _is_true(os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"))

PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", HF_HUB_CACHE)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def is_offline_mode():
    return HF_HUB_OFFLINE

REGEX_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")


def _get_cache_file_to_return(
    path_or_repo_id: str,
    full_filename: str,
    cache_dir: Union[str, Path, None] = None,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
):
    # We try to see if we have a cached version (not up to date):
    resolved_file = try_to_load_from_cache(
        path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision, repo_type=repo_type
    )
    if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
        return resolved_file
    return None


def extract_commit_hash(resolved_file: Optional[str], commit_hash: Optional[str]) -> Optional[str]:
    """
    Extracts the commit hash from a resolved filename toward a cache file.
    """
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    resolved_file = str(Path(resolved_file).as_posix())
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None


def cached_file(
    path_or_repo_id: Union[str, os.PathLike],
    filename: str,
    **kwargs,
) -> Optional[str]:
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:
            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
        proxies (`dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `hf auth login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).

    Examples:

    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("google-bert/bert-base-uncased", "pytorch_model.bin")
    ```
    """
    file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    file = file[0] if file is not None else file
    return file



def cached_files(
    path_or_repo_id: Union[str, os.PathLike],
    filenames: list[str],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    repo_type: Optional[str] = None,
    user_agent: Optional[Union[str, dict[str, str]]] = None,
    _raise_exceptions_for_gated_repo: bool = True,
    _raise_exceptions_for_missing_entries: bool = True,
    _raise_exceptions_for_connection_errors: bool = True,
    _commit_hash: Optional[str] = None,
    **deprecated_kwargs,
) -> Optional[str]:
    """
    Tries to locate several files in a local folder and repo, downloads and cache them if necessary.

    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:
            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filenames (`list[str]`):
            The name of all the files to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
        proxies (`dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `hf auth login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    Private args:
        _raise_exceptions_for_gated_repo (`bool`):
            if False, do not raise an exception for gated repo error but return None.
        _raise_exceptions_for_missing_entries (`bool`):
            if False, do not raise an exception for missing entries but return None.
        _raise_exceptions_for_connection_errors (`bool`):
            if False, do not raise an exception for connection errors but return None.
        _commit_hash (`str`, *optional*):
            passed when we are chaining several calls to various files (e.g. when loading a tokenizer or
            a pipeline). If files are cached for this commit hash, avoid calls to head and get from the cache.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).

    Examples:

    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("google-bert/bert-base-uncased", "pytorch_model.bin")
    ```
    """
    use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True
    if subfolder is None:
        subfolder = ""

    # Add folder to filenames
    full_filenames = [os.path.join(subfolder, file) for file in filenames]

    path_or_repo_id = str(path_or_repo_id)
    existing_files = []
    for filename in full_filenames:
        if os.path.isdir(path_or_repo_id):
            resolved_file = os.path.join(path_or_repo_id, filename)
            if not os.path.isfile(resolved_file):
                if _raise_exceptions_for_missing_entries and filename != os.path.join(subfolder, "config.json"):
                    revision_ = "main" if revision is None else revision
                    raise OSError(
                        f"{path_or_repo_id} does not appear to have a file named {filename}. Checkout "
                        f"'https://huggingface.co/{path_or_repo_id}/tree/{revision_}' for available files."
                    )
                else:
                    return None
            existing_files.append(resolved_file)

    # All files exist
    if len(existing_files) == len(full_filenames):
        return existing_files

    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    existing_files = []
    file_counter = 0
    if _commit_hash is not None and not force_download:
        for filename in full_filenames:
            # If the file is cached under that commit hash, we return it directly.
            resolved_file = try_to_load_from_cache(
                path_or_repo_id, filename, cache_dir=cache_dir, revision=_commit_hash, repo_type=repo_type
            )
            if resolved_file is not None:
                if resolved_file is not _CACHED_NO_EXIST:
                    file_counter += 1
                    existing_files.append(resolved_file)
                elif not _raise_exceptions_for_missing_entries:
                    file_counter += 1
                else:
                    raise OSError(f"Could not locate {filename} inside {path_or_repo_id}.")

    # Either all the files were found, or some were _CACHED_NO_EXIST but we do not raise for missing entries
    if file_counter == len(full_filenames):
        return existing_files if len(existing_files) > 0 else None

    # download the files if needed
    from huggingface_hub import HfApi, hf_hub_download
    if len(full_filenames) == 1:
        # This is slightly better for only 1 file
        hf_hub_download(
            path_or_repo_id,
            filenames[0],
            subfolder=None if len(subfolder) == 0 else subfolder,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            user_agent=user_agent,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )
    else:
        snapshot_download(
            path_or_repo_id,
            allow_patterns=full_filenames,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            user_agent=user_agent,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )



    resolved_files = [
        _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    ]
    # If there are any missing file and the flag is active, raise
    if any(file is None for file in resolved_files) and _raise_exceptions_for_missing_entries:
        missing_entries = [original for original, resolved in zip(full_filenames, resolved_files) if resolved is None]
        # Last escape
        if len(resolved_files) == 1 and missing_entries[0] == os.path.join(subfolder, "config.json"):
            return None
        # Now we raise for missing entries
        revision_ = "main" if revision is None else revision
        msg = (
            f"a file named {missing_entries[0]}" if len(missing_entries) == 1 else f"files named {(*missing_entries,)}"
        )
        raise OSError(
            f"{path_or_repo_id} does not appear to have {msg}. Checkout 'https://huggingface.co/{path_or_repo_id}/tree/{revision_}'"
            " for available files."
        )

    # Remove potential missing entries (we can silently remove them at this point based on the flags)
    resolved_files = [file for file in resolved_files if file is not None]
    # Return `None` if the list is empty, coherent with other Exception when the flag is not active
    resolved_files = None if len(resolved_files) == 0 else resolved_files

    return resolved_files


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
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
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
    from huggingface_hub.utils import EntryNotFoundError, filter_repo_objects
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
        
        # 过滤文件
        file_names: Iterable[str] = filter_repo_objects(
            items=file_names,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

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


def get_config_path(pretrained_model_name_or_path:Union[str, os.PathLike], allow_none=False, **kwargs) -> str:
    '''获取local文件夹下的config文件路径
    
    :param pretrained_model_name_or_path: str, 预训练权重的本地路径，或者是model_name
        1. model_name: 从hf下载
        2. local_file且config_path为None: 重新在local_file所在目录找对应的config_path
        3. local_dir且config_path为None: 重新在local_dir找对应的config_path
    :param allow_none: bool, 是否允许本地找不到config文件
    '''
    if pretrained_model_name_or_path is None:
        return pretrained_model_name_or_path

    config_path = None       
    # 传入bert4torch_config.json路径
    if pretrained_model_name_or_path.endswith(BERT4TORCH_CONFIG_NAME):
        if os.path.isfile(pretrained_model_name_or_path):
            # 文件存在
            config_path = pretrained_model_name_or_path
        elif not allow_none:
            raise FileNotFoundError(f'{pretrained_model_name_or_path} not exists, please check your local path or model_name.')

    # 传入文件夹路径
    elif os.path.isdir(pretrained_model_name_or_path):
        config_path_tmp = os.path.join(pretrained_model_name_or_path, BERT4TORCH_CONFIG_NAME)
        if os.path.isfile(config_path_tmp):
            # 文件存在
            config_path = config_path_tmp
        elif not allow_none:
            raise FileNotFoundError(f'{config_path_tmp} not exists, please check your local path or model_name.')

    # model_name: 从hf下载
    elif len(re.findall('/', pretrained_model_name_or_path)) <= 1 and not re.search(r'\\', pretrained_model_name_or_path):
        if pretrained_model_name_or_path.startswith('Tongjilibo/'):
            # 独立的repo
            config_path = snapshot_download(pretrained_model_name_or_path, filename=BERT4TORCH_CONFIG_NAME, **kwargs)
        else:
            # 单独下载bert4torch_config.json文件
            filename = '/'.join(pretrained_model_name_or_path.split('/')[-2:]) + f'/{BERT4TORCH_CONFIG_NAME}'
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


class PushToHubMixin:
    """
    A Mixin containing the functionality to push a model or tokenizer to the hub.
    """

    def _create_repo(
        self,
        repo_id: str,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        repo_url: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> str:
        """
        Create the repo if needed, cleans up repo_id with deprecated kwargs `repo_url` and `organization`, retrieves
        the token.
        """
        if repo_url is not None:
            warnings.warn(
                "The `repo_url` argument is deprecated and will be removed in v5 of Transformers. Use `repo_id` "
                "instead."
            )
            if repo_id is not None:
                raise ValueError(
                    "`repo_id` and `repo_url` are both specified. Please set only the argument `repo_id`."
                )
            repo_id = repo_url.replace(f"{HUGGINGFACE_CO_RESOLVE_ENDPOINT}/", "")
        if organization is not None:
            warnings.warn(
                "The `organization` argument is deprecated and will be removed in v5 of Transformers. Set your "
                "organization directly in the `repo_id` passed instead (`repo_id={organization}/{model_id}`)."
            )
            if not repo_id.startswith(organization):
                if "/" in repo_id:
                    repo_id = repo_id.split("/")[-1]
                repo_id = f"{organization}/{repo_id}"
        from huggingface_hub import create_repo
        url = create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
        return url.repo_id

    def _get_files_timestamps(self, working_dir: Union[str, os.PathLike]):
        """
        Returns the list of files with their last modification timestamp.
        """
        return {f: os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}

    def _upload_modified_files(
        self,
        working_dir: Union[str, os.PathLike],
        repo_id: str,
        files_timestamps: dict[str, float],
        commit_message: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
        revision: Optional[str] = None,
        commit_description: Optional[str] = None,
    ):
        """
        Uploads all modified files in `working_dir` to `repo_id`, based on `files_timestamps`.
        """
        if commit_message is None:
            if "Model" in self.__class__.__name__:
                commit_message = "Upload model"
            elif "Config" in self.__class__.__name__:
                commit_message = "Upload config"
            elif "Tokenizer" in self.__class__.__name__:
                commit_message = "Upload tokenizer"
            elif "FeatureExtractor" in self.__class__.__name__:
                commit_message = "Upload feature extractor"
            elif "Processor" in self.__class__.__name__:
                commit_message = "Upload processor"
            else:
                commit_message = f"Upload {self.__class__.__name__}"
        modified_files = [
            f
            for f in os.listdir(working_dir)
            if f not in files_timestamps or os.path.getmtime(os.path.join(working_dir, f)) > files_timestamps[f]
        ]

        # filter for actual files + folders at the root level
        modified_files = [
            f
            for f in modified_files
            if os.path.isfile(os.path.join(working_dir, f)) or os.path.isdir(os.path.join(working_dir, f))
        ]

        from huggingface_hub import CommitOperationAdd, create_branch, create_commit
        from huggingface_hub.utils import HfHubHTTPError
        operations = []
        # upload standalone files
        for file in modified_files:
            if os.path.isdir(os.path.join(working_dir, file)):
                # go over individual files of folder
                for f in os.listdir(os.path.join(working_dir, file)):
                    operations.append(
                        CommitOperationAdd(
                            path_or_fileobj=os.path.join(working_dir, file, f), path_in_repo=os.path.join(file, f)
                        )
                    )
            else:
                operations.append(
                    CommitOperationAdd(path_or_fileobj=os.path.join(working_dir, file), path_in_repo=file)
                )

        if revision is not None and not revision.startswith("refs/pr"):
            try:
                create_branch(repo_id=repo_id, branch=revision, token=token, exist_ok=True)
            except HfHubHTTPError as e:
                if e.response.status_code == 403 and create_pr:
                    # If we are creating a PR on a repo we don't have access to, we can't create the branch.
                    # so let's assume the branch already exists. If it's not the case, an error will be raised when
                    # calling `create_commit` below.
                    pass
                else:
                    raise

        logger.info(f"Uploading the following files to {repo_id}: {','.join(modified_files)}")
        return create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            commit_description=commit_description,
            token=token,
            create_pr=create_pr,
            revision=revision,
        )

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        max_shard_size: Optional[Union[int, str]] = "5GB",
        create_pr: bool = False,
        safe_serialization: bool = True,
        revision: Optional[str] = None,
        commit_description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        **deprecated_kwargs,
    ) -> str:
        """
        Upload the {object_files} to the 🤗 Model Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your {object} to. It should contain your organization name
                when pushing to a given organization.
            use_temp_dir (`bool`, *optional*):
                Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
                Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload {object}"`.
            private (`bool`, *optional*):
                Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `hf auth login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
                will then be each of size lower than this size. If expressed as a string, needs to be digits followed
                by a unit (like `"5MB"`). We default it to `"5GB"` so that users can easily load models on free-tier
                Google Colab instances without any CPU OOM issues.
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether or not to convert the model weights in safetensors format for safer serialization.
            revision (`str`, *optional*):
                Branch to push the uploaded files to.
            commit_description (`str`, *optional*):
                The description of the commit that will be created
            tags (`list[str]`, *optional*):
                List of tags to push on the Hub.

        Examples:

        ```python
        from bert4torch import {object_class}

        {object} = {object_class}.from_pretrained("google-bert/bert-base-cased")

        # Push the {object} to your namespace with the name "my-finetuned-bert".
        {object}.push_to_hub("my-finetuned-bert")

        # Push the {object} to an organization with the name "my-finetuned-bert".
        {object}.push_to_hub("huggingface/my-finetuned-bert")
        ```
        """
        use_auth_token = deprecated_kwargs.pop("use_auth_token", None)
        ignore_metadata_errors = deprecated_kwargs.pop("ignore_metadata_errors", False)
        save_jinja_files = deprecated_kwargs.pop(
            "save_jinja_files", None
        )  # TODO: This is only used for testing and should be removed once save_jinja_files becomes the default
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        repo_path_or_name = deprecated_kwargs.pop("repo_path_or_name", None)
        if repo_path_or_name is not None:
            # Should use `repo_id` instead of `repo_path_or_name`. When using `repo_path_or_name`, we try to infer
            # repo_id from the folder path, if it exists.
            warnings.warn(
                "The `repo_path_or_name` argument is deprecated and will be removed in v5 of Transformers. Use "
                "`repo_id` instead.",
                FutureWarning,
            )
            if repo_id is not None:
                raise ValueError(
                    "`repo_id` and `repo_path_or_name` are both specified. Please set only the argument `repo_id`."
                )
            if os.path.isdir(repo_path_or_name):
                # repo_path: infer repo_id from the path
                repo_id = repo_path_or_name.split(os.path.sep)[-1]
                working_dir = repo_id
            else:
                # repo_name: use it as repo_id
                repo_id = repo_path_or_name
                working_dir = repo_id.split("/")[-1]
        else:
            # Repo_id is passed correctly: infer working_dir from it
            working_dir = repo_id.split("/")[-1]

        # Deprecation warning will be sent after for repo_url and organization
        repo_url = deprecated_kwargs.pop("repo_url", None)
        organization = deprecated_kwargs.pop("organization", None)

        repo_id = self._create_repo(
            repo_id, private=private, token=token, repo_url=repo_url, organization=organization
        )

        # Create a new empty model card and eventually tag it
        model_card = create_and_tag_model_card(
            repo_id, tags, token=token, ignore_metadata_errors=ignore_metadata_errors
        )

        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)

        with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)

            # Save all files.
            if save_jinja_files:
                self.save_pretrained(
                    work_dir,
                    max_shard_size=max_shard_size,
                    safe_serialization=safe_serialization,
                    save_jinja_files=True,
                )
            else:
                self.save_pretrained(work_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)

            # Update model card if needed:
            model_card.save(os.path.join(work_dir, "README.md"))

            return self._upload_modified_files(
                work_dir,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
                revision=revision,
                commit_description=commit_description,
            )


def download_url(url, proxies=None):
    """
    Downloads a given url in a temporary file. This function is not safe to use in multiple processes. Its only use is
    for deprecated behavior allowing to download config/models with a single url instead of using the Hub.

    Args:
        url (`str`): The url of the file to download.
        proxies (`dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.

    Returns:
        `str`: The location of the temporary file where the url was downloaded.
    """
    warnings.warn(
        f"Using `from_pretrained` with the url of a file (here {url}) is deprecated and won't be possible anymore in"
        " v5 of Transformers. You should host your file on the Hub (hf.co) instead and use the repository ID. Note"
        " that this is not compatible with the caching system (your file will be downloaded at each execution) or"
        " multiple processes (each process will download the file in a different temporary file).",
        FutureWarning,
    )
    tmp_fd, tmp_file = tempfile.mkstemp()
    from huggingface_hub import http_get
    with os.fdopen(tmp_fd, "wb") as f:
        http_get(url, f, proxies=proxies)
    return tmp_file


def is_remote_url(url_or_filename):
    from urllib.parse import urlparse
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def list_repo_templates(
    repo_id: str,
    *,
    local_files_only: bool,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list[str]:
    """List template files from a repo.

    A template is a jinja file located under the `additional_chat_templates/` folder.
    If working in offline mode or if internet is down, the method will list jinja template from the local cache - if any.
    """
    from huggingface_hub import list_repo_tree
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError, LocalEntryNotFoundError
    if not local_files_only:
        try:
            return [
                entry.path.removeprefix(f"{CHAT_TEMPLATE_DIR}/")
                for entry in list_repo_tree(
                    repo_id=repo_id,
                    revision=revision,
                    path_in_repo=CHAT_TEMPLATE_DIR,
                    recursive=False,
                )
                if entry.path.endswith(".jinja")
            ]
        except (GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError):
            raise  # valid errors => do not catch
        except (ConnectionError, HTTPError):
            pass  # offline mode, internet down, etc. => try local files

    # check local files
    try:
        snapshot_dir = snapshot_download(
            repo_id=repo_id, revision=revision, cache_dir=cache_dir, local_files_only=True
        )
    except LocalEntryNotFoundError:  # No local repo means no local files
        return []
    templates_dir = Path(snapshot_dir, CHAT_TEMPLATE_DIR)
    if not templates_dir.is_dir():
        return []
    return [entry.stem for entry in templates_dir.iterdir() if entry.is_file() and entry.name.endswith(".jinja")]


def create_repo(*args, **kwargs):
    from huggingface_hub import create_repo
    return create_repo(*args, **kwargs)


def is_offline_mode(*args, **kwargs):
    try:
        from huggingface_hub import is_offline_mode
        return is_offline_mode(*args, **kwargs)
    except ImportError:  # Huggingface hub not installed
        return False


def validate_typed_dict(*args, **kwargs):
    try:
        from huggingface_hub.dataclasses import validate_typed_dict
        return validate_typed_dict(*args, **kwargs)
    except ImportError:  # Huggingface hub not installed
        return None


def as_validated_field(*args, **kwargs):
    from huggingface_hub.dataclasses import as_validated_field
    return as_validated_field(*args, **kwargs)


def model_info(*args, **kwargs):
    from huggingface_hub import model_info
    return model_info(*args, **kwargs)

def list_repo_tree(*args, **kwargs):
    from huggingface_hub import list_repo_tree
    return list_repo_tree(*args, **kwargs)


def list_repo_files(*args, **kwargs):
    from huggingface_hub import list_repo_files
    return list_repo_files(*args, **kwargs)
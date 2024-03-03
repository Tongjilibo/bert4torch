import importlib.util
import torch
from packaging import version
from torch4keras.snippets.import_utils import is_package_available
import sys
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


def is_accelerate_available(check_partial_state=False):
    '''是否可以使用accelerate'''
    accelerate_available = importlib.util.find_spec("accelerate") is not None
    if accelerate_available:
        if check_partial_state:
            return version.parse(importlib_metadata.version("accelerate")) >= version.parse("0.17.0")
        else:
            return True
    else:
        return False


def is_flash_attn_available():
    '''是否可以使用包flash_attn'''
    _flash_attn_available = is_package_available("flash_attn") and \
        version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")
    return _flash_attn_available and torch.cuda.is_available()


def is_xformers_available():
    '''是否可以使用xformers加速'''
    return is_package_available("xformers")


def is_fastapi_available():
    '''是否可以使用包fastapi'''
    return is_package_available('fastapi')


def is_pydantic_available():
    return is_package_available('pydantic')


def is_trl_available():
    return is_package_available("trl")


def is_sseclient_available():
    return importlib.util.find_spec("sseclient")


def is_streamlit_available():
    return is_package_available('streamlit')
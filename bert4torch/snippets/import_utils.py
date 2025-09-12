import importlib.util
import torch
from packaging import version
from torch4keras.snippets.import_utils import *
from importlib.util import find_spec
import sys
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


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


def is_peft_available() -> bool:
    return find_spec("peft") is not None


def is_torch_sdpa_available() -> bool:
    return version.parse(torch.__version__) >= version.parse("2.1.1")


def is_transformers_available(return_version:bool=False) -> bool:
    return is_package_available('transformers', return_version)


def is_auto_gptq_available() -> bool:
    return is_package_available("auto_gptq")


def is_gptqmodel_available() -> bool:
    return is_package_available("gptqmodel")


def is_optimum_available() -> bool:
    return is_package_available("optimum")


def is_auto_awq_available() -> bool:
    return importlib.util.find_spec("awq") is not None
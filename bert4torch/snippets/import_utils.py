import importlib.util
from typing import Any, Tuple, Union
import torch
from packaging import version
import sys
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.) Talk to Sylvain to see how to do with it better.
def is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
        # print(f"Detected {pkg_name} version {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


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


def is_safetensors_available():
    return is_package_available("safetensors")


def is_trl_available():
    return is_package_available("trl")


def is_sseclient_available():
    return is_package_available("sseclient")
import importlib.util
import torch
from packaging import version
from torch4keras.snippets.import_utils import *
from importlib.util import find_spec
from typing import List
import sys
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata
from . import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
FORCE_TF_AVAILABLE = os.environ.get("FORCE_TF_AVAILABLE", "AUTO").upper()


_tf_version = "N/A"
_tf_available = False
if FORCE_TF_AVAILABLE in ENV_VARS_TRUE_VALUES:
    _tf_available = True
else:
    if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
        # Note: _is_package_available("tensorflow") fails for tensorflow-cpu. Please test any changes to the line below
        # with tensorflow-cpu to make sure it still works!
        _tf_available = importlib.util.find_spec("tensorflow") is not None
        if _tf_available:
            candidates = (
                "tensorflow",
                "tensorflow-cpu",
                "tensorflow-gpu",
                "tf-nightly",
                "tf-nightly-cpu",
                "tf-nightly-gpu",
                "tf-nightly-rocm",
                "intel-tensorflow",
                "intel-tensorflow-avx512",
                "tensorflow-rocm",
                "tensorflow-macos",
                "tensorflow-aarch64",
            )
            _tf_version = None
            # For the metadata, we have to look for both tensorflow and tensorflow-cpu
            for pkg in candidates:
                try:
                    _tf_version = importlib.metadata.version(pkg)
                    break
                except importlib.metadata.PackageNotFoundError:
                    pass
            _tf_available = _tf_version is not None
        if _tf_available:
            if version.parse(_tf_version) < version.parse("2"):
                logger.info(
                    f"TensorFlow found but with version {_tf_version}. Transformers requires version 2 minimum."
                )
                _tf_available = False
    else:
        logger.info("Disabling Tensorflow because USE_TORCH is set")


def is_tf_available():
    return _tf_available


def is_jinja_available():
    return is_package_available("jinja2")


@lru_cache
def is_vision_available():
    _pil_available = importlib.util.find_spec("PIL") is not None
    if _pil_available:
        try:
            package_version = importlib.metadata.version("Pillow")
        except importlib.metadata.PackageNotFoundError:
            try:
                package_version = importlib.metadata.version("Pillow-SIMD")
            except importlib.metadata.PackageNotFoundError:
                return False
        logger.debug(f"Detected PIL version {package_version}")
    return _pil_available


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


def is_protobuf_available():
    if importlib.util.find_spec("google") is None:
        return False
    return importlib.util.find_spec("google.protobuf") is not None


def is_tokenizers_available():
    return is_package_available('tokenizers')


def is_sentencepiece_available():
    return is_package_available("sentencepiece")


def get_valid_subdirs(root_dir: str) -> List[str]:
    """获取所有包含 __init__.py 的有效子目录"""
    subdirs = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        # 是目录 + 不在排除列表 + 包含 __init__.py
        if (
            os.path.isdir(item_path)
            and os.path.exists(os.path.join(item_path, "__init__.py"))
        ):
            subdirs.append(item)
    return subdirs


def import_submodels(root_dir: str, package_prefix='') -> None:
    """从指定子模块中导入所有模型类"""
    
    # 确定要扫描的目标目录
    target_dirs = get_valid_subdirs(root_dir)

    # 遍历所有目标目录，导入模型
    for subdir in target_dirs:
        try:
            sub_module = importlib.import_module(f"{package_prefix}.{subdir}", package=__name__)
        except ImportError as e:
            log_error_once(f"import {package_prefix}.{subdir} failed - {e}")
            return


# docstyle-ignore
PROTOBUF_IMPORT_ERROR = """
{0} requires the protobuf library but it was not found in your environment. Check out the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""

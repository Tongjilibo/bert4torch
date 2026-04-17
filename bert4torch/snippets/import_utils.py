import importlib.util
import torch
from packaging import version
from torch4keras.snippets.import_utils import *
from importlib.util import find_spec
from typing import List
import operator
import re
from enum import Enum
from types import ModuleType
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
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()
BITSANDBYTES_MIN_VERSION = "0.46.1"

FORCE_TF_AVAILABLE = os.environ.get("FORCE_TF_AVAILABLE", "AUTO").upper()

_torchvision_available, _torchvision_version = is_package_available("torchvision", return_version=True)


_tf_version = "N/A"
_tf_available = False
if FORCE_TF_AVAILABLE in ENV_VARS_TRUE_VALUES:
    _tf_available = True
else:
    if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
        # Note: is_package_available("tensorflow") fails for tensorflow-cpu. Please test any changes to the line below
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


_flax_available = False
if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _flax_available, _flax_version = is_package_available("flax", return_version=True)
    if _flax_available:
        _jax_available, _jax_version = is_package_available("jax", return_version=True)
        if _jax_available:
            logger.info(f"JAX version {_jax_version}, Flax version {_flax_version} available.")
        else:
            _flax_available = _jax_available = False
            _jax_version = _flax_version = "N/A"


# docstyle-ignore
PROTOBUF_IMPORT_ERROR = """
{0} requires the protobuf library but it was not found in your environment. Check out the instructions on the
installation page of its repo: https://github.com/protocolbuffers/protobuf/tree/master/python#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


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


def is_librosa_available():
    return is_package_available("librosa")


def is_flax_available():
    return _flax_available


def is_soundfile_available():
    return is_package_available("soundfile")


def is_rocm_platform():
    if is_torch_available():
        import torch

        return torch.version.hip is not None
    else:
        return False


def is_torchvision_available():
    return _torchvision_available


def is_torchvision_v2_available():
    if not is_torchvision_available():
        return False

    # NOTE: We require torchvision>=0.15 as v2 transforms are available from this version: https://pytorch.org/vision/stable/transforms.html#v1-or-v2-which-one-should-i-use
    return version.parse(_torchvision_version) >= version.parse("0.15")


def is_av_available():
    return importlib.util.find_spec("av") is not None


def is_cv2_available():
    return importlib.util.find_spec("cv2") is not None


def is_decord_available():
    return importlib.util.find_spec("decord") is not None


def is_torchcodec_available():
    return importlib.util.find_spec("torchcodec") is not None


def is_yt_dlp_available():
    return importlib.util.find_spec("yt_dlp") is not None


def is_mlx_available():
    return is_package_available("mlx")


def is_openai_available():
    return is_package_available("openai")


@lru_cache
def is_torch_mlu_available(check_device=False):
    """
    Checks if `mlu` is available via an `cndev-based` check which won't trigger the drivers and leave mlu
    uninitialized.
    """
    if not is_torch_available() or importlib.util.find_spec("torch_mlu") is None:
        return False

    import torch
    import torch_mlu  # noqa: F401

    pytorch_cndev_based_mlu_check_previous_value = os.environ.get("PYTORCH_CNDEV_BASED_MLU_CHECK")
    try:
        os.environ["PYTORCH_CNDEV_BASED_MLU_CHECK"] = str(1)
        available = torch.mlu.is_available()
    finally:
        if pytorch_cndev_based_mlu_check_previous_value:
            os.environ["PYTORCH_CNDEV_BASED_MLU_CHECK"] = pytorch_cndev_based_mlu_check_previous_value
        else:
            os.environ.pop("PYTORCH_CNDEV_BASED_MLU_CHECK", None)

    return available


def is_flash_attn_2_available():
    if not is_torch_available():
        return False

    if not is_package_available("flash_attn"):
        return False

    # Let's add an extra check to see if cuda is available
    import torch

    if not (torch.cuda.is_available() or is_torch_mlu_available()):
        return False

    if torch.version.cuda:
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.1.0")
    elif torch.version.hip:
        # TODO: Bump the requirement to 2.1.0 once released in https://github.com/ROCmSoftwarePlatform/flash-attention
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.0.4")
    elif is_torch_mlu_available():
        return version.parse(importlib.metadata.version("flash_attn")) >= version.parse("2.3.3")
    else:
        return False
    

@lru_cache
def is_gptqmodel_available() -> bool:
    return is_package_available("gptqmodel")[0]


def is_kernels_available():
    return is_package_available("kernels")


def is_torchdynamo_compiling() -> bool:
    # Importing torch._dynamo causes issues with PyTorch profiler (https://github.com/pytorch/pytorch/issues/130622)
    # hence rather relying on `torch.compiler.is_compiling()` when possible (torch>=2.3)
    try:
        import torch

        if hasattr(torch, "compiler"):
            return torch.compiler.is_compiling()
        return False
    except Exception:
        return False
    

@lru_cache
def get_torch_version() -> str:
    _, torch_version = is_package_available("torch", return_version=True)
    return torch_version


@lru_cache
def is_bitsandbytes_available(min_version: str = BITSANDBYTES_MIN_VERSION) -> bool:
    is_available, bitsandbytes_version = is_package_available("bitsandbytes", return_version=True)
    return is_available and version.parse(bitsandbytes_version) >= version.parse(min_version)


@lru_cache
def is_torch_cuda_available() -> bool:
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    return False


@lru_cache
def is_torch_mps_available(min_version: str | None = None) -> bool:
    if is_torch_available():
        import torch

        backend_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        if min_version is not None:
            flag = version.parse(get_torch_version()) >= version.parse(min_version)
            backend_available = backend_available and flag
        return backend_available
    return False


@lru_cache
def is_torch_mps_available(min_version: str | None = None) -> bool:
    if is_torch_available():
        import torch

        backend_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        if min_version is not None:
            flag = version.parse(get_torch_version()) >= version.parse(min_version)
            backend_available = backend_available and flag
        return backend_available
    return False


@lru_cache
def is_torch_npu_available(check_device=False) -> bool:
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if not is_torch_available() or not is_package_available("torch_npu")[0]:
        return False

    import torch
    import torch_npu  # noqa: F401

    if check_device:
        try:
            # Will raise a RuntimeError if no NPU is found
            if hasattr(torch, "npu"):
                _ = torch.npu.device_count()
                return torch.npu.is_available()
            return False
        except RuntimeError:
            return False
    return hasattr(torch, "npu") and torch.npu.is_available()


@lru_cache
def is_torch_xpu_available(check_device: bool = False) -> bool:
    """
    Checks if XPU acceleration is available via stock PyTorch (>=2.6) and
    potentially if a XPU is in the environment.
    """
    if not is_torch_available():
        return False

    torch_version = version.parse(get_torch_version())
    if torch_version.major == 2 and torch_version.minor < 6:
        return False

    import torch

    if check_device:
        try:
            # Will raise a RuntimeError if no XPU is found
            _ = torch.xpu.device_count()
            return torch.xpu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, "xpu") and torch.xpu.is_available()


@lru_cache
def is_torch_mlu_available() -> bool:
    """
    Checks if `mlu` is available via an `cndev-based` check which won't trigger the drivers and leave mlu
    uninitialized.
    """
    if not is_torch_available() or not is_package_available("torch_mlu")[0]:
        return False

    import torch
    import torch_mlu  # noqa: F401

    pytorch_cndev_based_mlu_check_previous_value = os.environ.get("PYTORCH_CNDEV_BASED_MLU_CHECK")
    try:
        os.environ["PYTORCH_CNDEV_BASED_MLU_CHECK"] = str(1)
        available = torch.mlu.is_available() if hasattr(torch, "mlu") else False
    finally:
        if pytorch_cndev_based_mlu_check_previous_value:
            os.environ["PYTORCH_CNDEV_BASED_MLU_CHECK"] = pytorch_cndev_based_mlu_check_previous_value
        else:
            os.environ.pop("PYTORCH_CNDEV_BASED_MLU_CHECK", None)

    return available


@lru_cache
def is_torch_musa_available(check_device=False) -> bool:
    "Checks if `torch_musa` is installed and potentially if a MUSA is in the environment"
    if not is_torch_available() or not is_package_available("torch_musa")[0]:
        return False

    import torch
    import torch_musa  # noqa: F401

    torch_musa_min_version = "0.33.0"
    accelerate_available, accelerate_version = is_package_available("accelerate", return_version=True)
    if accelerate_available and version.parse(accelerate_version) < version.parse(torch_musa_min_version):
        return False

    if check_device:
        try:
            # Will raise a RuntimeError if no MUSA is found
            if hasattr(torch, "musa"):
                _ = torch.musa.device_count()
                return torch.musa.is_available()
            return False
        except RuntimeError:
            return False
    return hasattr(torch, "musa") and torch.musa.is_available()


@lru_cache
def is_torch_hpu_available() -> bool:
    "Checks if `torch.hpu` is available and potentially if a HPU is in the environment"
    if (
        not is_torch_available()
        or not is_package_available("habana_frameworks")[0]
        or not is_package_available("habana_frameworks.torch")[0]
    ):
        return False

    torch_hpu_min_accelerate_version = "1.5.0"
    accelerate_available, accelerate_version = is_package_available("accelerate", return_version=True)
    if accelerate_available and version.parse(accelerate_version) < version.parse(torch_hpu_min_accelerate_version):
        return False

    import torch

    if os.environ.get("PT_HPU_LAZY_MODE", "1") == "1":
        # import habana_frameworks.torch in case of lazy mode to patch torch with torch.hpu
        import habana_frameworks.torch  # noqa: F401

    if not hasattr(torch, "hpu") or not torch.hpu.is_available():
        return False

    # We patch torch.gather for int64 tensors to avoid a bug on Gaudi
    # Graph compile failed with synStatus 26 [Generic failure]
    # This can be removed once bug is fixed but for now we need it.
    original_gather = torch.gather

    def patched_gather(input: torch.Tensor, dim: int, index: torch.LongTensor) -> torch.Tensor:
        if input.dtype == torch.int64 and input.device.type == "hpu":
            return original_gather(input.to(torch.int32), dim, index).to(torch.int64)
        else:
            return original_gather(input, dim, index)

    torch.gather = patched_gather
    torch.Tensor.gather = patched_gather

    original_take_along_dim = torch.take_along_dim

    def patched_take_along_dim(input: torch.Tensor, indices: torch.LongTensor, dim: int | None = None) -> torch.Tensor:
        if input.dtype == torch.int64 and input.device.type == "hpu":
            return original_take_along_dim(input.to(torch.int32), indices, dim).to(torch.int64)
        else:
            return original_take_along_dim(input, indices, dim)

    torch.take_along_dim = patched_take_along_dim

    original_cholesky = torch.linalg.cholesky

    def safe_cholesky(A, *args, **kwargs):
        output = original_cholesky(A, *args, **kwargs)

        if torch.isnan(output).any():
            jitter_value = 1e-9
            diag_jitter = torch.eye(A.size(-1), dtype=A.dtype, device=A.device) * jitter_value
            output = original_cholesky(A + diag_jitter, *args, **kwargs)

        return output

    torch.linalg.cholesky = safe_cholesky

    original_scatter = torch.scatter

    def patched_scatter(
        input: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        if input.device.type == "hpu" and input is src:
            return original_scatter(input, dim, index, src.clone(), *args, **kwargs)
        else:
            return original_scatter(input, dim, index, src, *args, **kwargs)

    torch.scatter = patched_scatter
    torch.Tensor.scatter = patched_scatter

    # IlyasMoutawwakil: we patch torch.compile to use the HPU backend by default
    # https://github.com/huggingface/transformers/pull/38790#discussion_r2157043944
    # This is necessary for cases where torch.compile is used as a decorator (defaulting to inductor)
    # https://github.com/huggingface/transformers/blob/af6120b3eb2470b994c21421bb6eaa76576128b0/src/transformers/models/modernbert/modeling_modernbert.py#L204
    original_compile = torch.compile

    def hpu_backend_compile(*args, **kwargs):
        if kwargs.get("backend") not in ["hpu_backend", "eager"]:
            logger.warning(
                f"Calling torch.compile with backend={kwargs.get('backend')} on a Gaudi device is not supported. "
                "We will override the backend with 'hpu_backend' to avoid errors."
            )
            kwargs["backend"] = "hpu_backend"

        return original_compile(*args, **kwargs)

    torch.compile = hpu_backend_compile

    return True


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


def direct_bert4torch_import(path: str, file="__init__.py") -> ModuleType:
    """Imports transformers directly

    Args:
        path (`str`): The path to the source file
        file (`str`, *optional*): The file to join with the path. Defaults to "__init__.py".

    Returns:
        `ModuleType`: The resulting imported module
    """
    name = "bert4torch"
    location = os.path.join(path, file)
    spec = importlib.util.spec_from_file_location(name, location, submodule_search_locations=[path])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module = sys.modules[name]
    return module


class VersionComparison(Enum):
    EQUAL = operator.eq
    NOT_EQUAL = operator.ne
    GREATER_THAN = operator.gt
    LESS_THAN = operator.lt
    GREATER_THAN_OR_EQUAL = operator.ge
    LESS_THAN_OR_EQUAL = operator.le

    @staticmethod
    def from_string(version_string: str) -> "VersionComparison":
        string_to_operator = {
            "=": VersionComparison.EQUAL.value,
            "==": VersionComparison.EQUAL.value,
            "!=": VersionComparison.NOT_EQUAL.value,
            ">": VersionComparison.GREATER_THAN.value,
            "<": VersionComparison.LESS_THAN.value,
            ">=": VersionComparison.GREATER_THAN_OR_EQUAL.value,
            "<=": VersionComparison.LESS_THAN_OR_EQUAL.value,
        }

        return string_to_operator[version_string]


@lru_cache
def split_package_version(package_version_str) -> tuple[str, str, str]:
    pattern = r"([a-zA-Z0-9_-]+)([!<>=~]+)([0-9.]+)"
    match = re.match(pattern, package_version_str)
    if match:
        return (match.group(1), match.group(2), match.group(3))
    else:
        raise ValueError(f"Invalid package version string: {package_version_str}")


class Backend:
    def __init__(self, backend_requirement: str):
        self.package_name, self.version_comparison, self.version = split_package_version(backend_requirement)

    def is_satisfied(self) -> bool:
        return VersionComparison.from_string(self.version_comparison)(
            version.parse(importlib.metadata.version(self.package_name)), version.parse(self.version)
        )

    def __repr__(self) -> str:
        return f'Backend("{self.package_name}", {VersionComparison[self.version_comparison]}, "{self.version}")'

    @property
    def error_message(self):
        return (
            f"{{0}} requires the {self.package_name} library version {self.version_comparison}{self.version}. That"
            f" library was not found with this version in your environment."
        )


def requires(*, backends=()):
    """
    This decorator enables two things:
    - Attaching a `__backends` tuple to an object to see what are the necessary backends for it
      to execute correctly without instantiating it
    - The '@requires' string is used to dynamically import objects
    """

    if not isinstance(backends, tuple):
        raise TypeError("Backends should be a tuple.")

    applied_backends = []
    for backend in backends:
        if 'is_' + backend + '_available' in globals() or is_package_available(backend)[0]:
            applied_backends.append(backend)
        else:
            if any(key in backend for key in ["=", "<", ">"]):
                applied_backends.append(Backend(backend))
            else:
                raise ValueError(f"Backend should be defined in the BACKENDS_MAPPING. Offending backend: {backend}")

    def inner_fn(fun):
        fun.__backends = applied_backends
        return fun

    return inner_fn


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__

    failed = []
    for backend in backends:
        if isinstance(backend, Backend):
            available, msg = backend.is_satisfied, backend.error_message
        else:
            func_name = 'is_' + backend + '_available'
            if func_name in globals():
                # 存在这个函数
                available = globals()[func_name]
            else:
                available = lambda x: is_package_available(backend)[0]
            msg = "{0} requires the " + f"{backend} library but it was not found in your environment."

        if not available():
            failed.append(msg.format(name))

    if failed:
        raise ImportError("".join(failed))

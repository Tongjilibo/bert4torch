import importlib.metadata
from typing import TYPE_CHECKING, List, Optional
from packaging import version
from .base import QuantizerBase
from bert4torch.snippets.import_utils import is_torch_available, is_auto_awq_available, is_accelerate_available
from bert4torch.snippets import log_warn, log_warn_once, log_info
from enum import Enum
import re


if is_torch_available():
    import torch


class AWQLinearVersion(str, Enum):
    GEMM = "gemm"
    GEMV = "gemv"
    EXLLAMA = "exllama"
    IPEX = "ipex"

    @staticmethod
    def from_str(version: str):
        version = version.lower()
        if version == "gemm":
            return AWQLinearVersion.GEMM
        elif version == "gemv":
            return AWQLinearVersion.GEMV
        elif version == "exllama":
            return AWQLinearVersion.EXLLAMA
        elif version == "ipex":
            return AWQLinearVersion.IPEX
        else:
            raise ValueError(f"Unknown AWQLinearVersion {version}")


class AwqQuantizer(QuantizerBase):
    """
    4-bit quantization for Activation-aware Weight Quantization(AWQ) (https://arxiv.org/abs/2306.00978)
    """

    # AWQ requires data callibration - we support only inference
    requires_calibration = True

    required_packages = ["awq", "accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, device_map, **kwargs):
        if not is_auto_awq_available():
            raise ImportError("Loading an AWQ quantized model requires auto-awq library (`pip install autoawq`)")

        if not is_accelerate_available():
            raise ImportError("Loading an AWQ quantized model requires accelerate (`pip install accelerate`)")

        if self.quantization_config.version == AWQLinearVersion.GEMM and not torch.cuda.is_available():
            log_warn_once("No CUDA found, replace GEMM with IPEX version to support non-cuda AWQ model.")
            self.quantization_config.version = AWQLinearVersion.IPEX

        if self.quantization_config.version == AWQLinearVersion.IPEX:
            if version.parse(importlib.metadata.version("autoawq")) < version.parse("0.2.6"):
                raise RuntimeError(
                    "To use IPEX backend, you need autoawq>0.2.6. Please install the latest version or from source."
                )
            if device_map is None:
                log_warn_once(
                    "You have loaded an AWQ model without setting device_map, please set 'cpu' or 'xpu' or 'auto'"
                )
            elif isinstance(device_map, dict) and "disk" in device_map.values():
                raise ValueError(
                    "You are attempting to load an IPEX version AWQ model with a device_map that contains disk device."
                    " This is not supported. Please make sure only cpu and xpu in the device_map."
                )
        else:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "GPU is required to run AWQ quantized model. You can use IPEX version AWQ if you have an Intel CPU"
                )

            if device_map is None:
                log_warn_once(
                    "You have loaded an AWQ model on CPU and have a CUDA device available, make sure to set "
                    "your model on a GPU device in order to run your model."
                )
            elif device_map is not None:
                if isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
                    raise ValueError(
                        "You are attempting to load an AWQ model with a device_map that contains a CPU or disk device."
                        " This is not supported. Please remove the CPU or disk device from the device_map."
                    )

    def update_torch_dtype(self, torch_dtype):
        if torch_dtype is None:
            torch_dtype = torch.float16
            log_info("Loading the model in `torch.float16`. To overwrite it, set `torch_dtype` manually.")
        elif torch_dtype != torch.float16:
            log_warn("We suggest you to set `torch_dtype=torch.float16` for better efficiency with AWQ.")
        return torch_dtype

    def _process_model_before_weight_loading(self, model, keep_in_fp32_modules: Optional[List[str]] = None, **kwargs):
        from transformers.integrations import replace_quantization_scales, replace_with_awq_linear

        # 修改
        self.modules_to_not_convert = self.quantization_config.modules_to_not_convert
        new_mapping = modify_variable_mapping(model, self.modules_to_not_convert)
        model.variable_mapping = lambda: new_mapping

        model, has_been_replaced = replace_with_awq_linear(
            model, quantization_config=self.quantization_config, modules_to_not_convert=self.modules_to_not_convert
        )
        model = replace_quantization_scales(model, model.config.model)

        if not has_been_replaced:
            log_warn(
                "You are loading an AWQ model but no linear modules were found in your model."
                " Please double check your model architecture, or submit an issue on github if you think this is a bug."
            )

    def _process_model_after_weight_loading(self, model, **kwargs):
        if self.quantization_config.do_fuse:
            from transformers.integrations import fuse_awq_modules

            model = fuse_awq_modules(model, self.quantization_config)
            model._awq_is_fused = True  # TODO: consider storing this flag in model.config instead

        if self.quantization_config.version == AWQLinearVersion.EXLLAMA:
            from transformers.integrations import post_init_awq_exllama_modules

            model = post_init_awq_exllama_modules(model, self.quantization_config.exllama_config)

        if self.quantization_config.version == AWQLinearVersion.IPEX:
            from transformers.integrations import post_init_awq_ipex_modules

            model = post_init_awq_ipex_modules(model)

    def is_serializable(self, safe_serialization=None):
        # AWQ through auto-awq has been always serializable, except if the model is fused.
        if self.quantization_config.do_fuse:
            log_warn("You cannot save an AWQ model that uses fused modules!")
            return False

        if self.quantization_config.version == AWQLinearVersion.EXLLAMA:
            log_warn("You cannot save an AWQ model that uses Exllama backend!")
            return False

        return True

    @property
    def is_trainable(self):
        # AWQ supports PEFT fine-tuning from version 0.2.0
        MIN_AWQ_VERSION_FOR_PEFT = "0.2.0"
        return version.parse(importlib.metadata.version("autoawq")) >= version.parse(MIN_AWQ_VERSION_FOR_PEFT)


def modify_variable_mapping(model, modules_to_not_convert:str):
    '''量化会修改模型的结构，因此也需要修改variable_mapping'''
    old_mapping = model.variable_mapping()
    new_mapping = {}
    for o, n in old_mapping.items():
        eval_str = 'model.' + re.sub(r'\.(\d+)\.', r'[\1].', o).replace('.weight', '').replace('.weight', '')
        module = eval(eval_str)
        if not isinstance(module, torch.nn.Linear):
            new_mapping[o] = n
            continue
        elif any([i in o for i in modules_to_not_convert]):
            new_mapping[o] = n
            continue
        o = o.replace('.weight', '').replace('.weight', '')
        n = n.replace('.weight', '').replace('.weight', '')
        new_mapping.update({
            f"{o}.qweight": f"{n}.qweight",
            f"{o}.qzeros": f"{n}.qzeros",
            f"{o}.scales": f"{n}.scales"
        })

    return new_mapping
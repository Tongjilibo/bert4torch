import importlib.metadata
from typing import TYPE_CHECKING, List, Optional
from packaging import version
from .base import QuantizerBase, register_quantizer, should_convert_module
from .quantization_config import AwqBackend, AwqConfig
from bert4torch.snippets.import_utils import is_torch_available, is_gptqmodel_available, is_accelerate_available
from bert4torch.snippets import log_warn, log_warn_once, log_info, logging
from enum import Enum
import re


if is_torch_available():
    import torch
    import torch.nn as nn

logger = logging.get_logger(__name__)


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


AWQ_SCALES_MAPPINGS = {
    "starcoder2": {"act": "act", "layer_before_act": "c_fc"},
    "RefinedWebModel": {"act": "act", "layer_before_act": "dense_h_to_4h"},
    "falcon": {"act": "act", "layer_before_act": "dense_h_to_4h"},
    "mpt": {"act": "act", "layer_before_act": "up_proj"},
    "gptj": {"act": "act", "layer_before_act": "fc_in"},
    "gpt_neox": {"act": "act", "layer_before_act": "dense_h_to_4h"},
    "gpt_bigcode": {"act": "act", "layer_before_act": "c_fc"},
    "bloom": {"act": "gelu_impl", "layer_before_act": "dense_h_to_4h"},
}


def replace_quantization_scales(model, model_type):
    from gptqmodel.quantization.awq.modules.act import ScaledActivation

    if model_type not in AWQ_SCALES_MAPPINGS:
        return model
    for name, module in model.named_children():
        act_name = AWQ_SCALES_MAPPINGS[model_type]["act"]
        layer_before_act_name = AWQ_SCALES_MAPPINGS[model_type]["layer_before_act"]
        if name == act_name and hasattr(model, layer_before_act_name):
            layer_before_act = getattr(model, AWQ_SCALES_MAPPINGS[model_type]["layer_before_act"])
            size = layer_before_act.out_features
            scale_like = torch.ones(size)
            model._modules[name] = ScaledActivation(module, scale_like)
        _ = replace_quantization_scales(module, model_type)
    return model


def replace_with_awq_linear(
    model,
    modules_to_not_convert=None,
    quantization_config=None,
    device_map: str | dict | None = None,
) -> bool:
    """
    Public method that replaces the linear layers of the given model with awq quantized layers.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`AwqConfig`):
            The quantization config object that contains the quantization parameters.
        modules_to_not_convert (`list[str]`, *optional*, defaults to `None`):
            A list of nn.Linear weights to not convert. If a parameter path is in the list (e.g. `lm_head.weight`), the corresponding module will not be
            converted.
        device_map (`Union[str, dict]`, *optional*, defaults to `None`):
            The device map that maps the parameters to the device
    """
    from gptqmodel.quantization import METHOD
    from gptqmodel.utils.importer import hf_select_quant_linear_v2

    target_cls = hf_select_quant_linear_v2(
        bits=quantization_config.bits,
        group_size=quantization_config.group_size,
        desc_act=False,
        sym=False,
        format=quantization_config.format,
        backend=quantization_config.backend,
        device_map=device_map,
        quant_method=METHOD.AWQ,
        zero_point=quantization_config.zero_point,
        pack=False,
    )

    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        with torch.device("meta"):
            if isinstance(module, nn.Linear):
                new_module = target_cls(
                    bits=quantization_config.bits,
                    sym=quantization_config.sym,
                    desc_act=quantization_config.desc_act,
                    group_size=quantization_config.group_size,
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    dev=module.weight.device,
                    register_buffers=True,
                )
                new_module.requires_grad_(False)
                model.set_submodule(module_name, new_module)
                has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using eetq but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model


@register_quantizer(name='awq')
class AwqQuantizer(QuantizerBase):
    """
    4-bit quantization for Activation-aware Weight Quantization(AWQ) (https://huggingface.co/papers/2306.00978)
    """

    # AWQ requires data calibration - we support only inference
    requires_calibration = True
    quantization_config: "AwqConfig"

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, **kwargs):
        if not is_gptqmodel_available():
            raise ImportError(
                "Loading an AWQ quantized model requires gptqmodel. Please install it with `pip install gptqmodel`"
            )

        if not is_accelerate_available():
            raise ImportError("Loading an AWQ quantized model requires accelerate (`pip install accelerate`)")

    def update_dtype(self, dtype):
        if dtype == torch.bfloat16 and (torch.cuda.is_available() or torch.xpu.is_available()):
            logger.warning(
                "`torch.bfloat16` is not supported for AWQ CUDA/XPU kernels yet. Casting to `torch.float16`."
            )
            dtype = torch.float16
        elif dtype != torch.float16 and (torch.cuda.is_available() or torch.xpu.is_available()):
            logger.warning("We suggest you to set `dtype=torch.float16` for better efficiency on CUDA/XPU with AWQ.")
        return dtype

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):

        # 修改
        self.modules_to_not_convert = self.quantization_config.modules_to_not_convert
        new_mapping = modify_variable_mapping(model, self.modules_to_not_convert)
        model.variable_mapping = lambda: new_mapping

        # self.modules_to_not_convert = self.get_modules_to_not_convert(
        #     model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules, add_default_skips=True
        # )

        model = replace_with_awq_linear(
            model,
            quantization_config=self.quantization_config,
            modules_to_not_convert=self.modules_to_not_convert,
            device_map=kwargs.get("device_map"),
        )

        model = replace_quantization_scales(model, model.config.model_type)

    def _process_model_after_weight_loading(self, model, **kwargs):
        from gptqmodel.utils.model import hf_gptqmodel_post_init

        hf_gptqmodel_post_init(model, use_act_order=self.quantization_config.desc_act)

    def is_serializable(self):
        if self.quantization_config.backend in [AwqBackend.EXLLAMA_V1, AwqBackend.EXLLAMA_V2]:
            logger.warning("You cannot save an AWQ model that uses Exllama backend!")
            return False

        return True

    @property
    def is_trainable(self):
        return version.parse(importlib.metadata.version("gptqmodel")) >= version.parse("5.0.0")


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
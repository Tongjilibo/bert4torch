
import importlib
from typing import TYPE_CHECKING, Optional
from packaging import version
from .base import QuantizerBase
from bert4torch.snippets.import_utils import is_auto_gptq_available, is_gptqmodel_available, is_optimum_available, is_torch_available
from bert4torch.snippets import log_warn, log_info
from bert4torch.models.modeling_utils import get_layers


if is_torch_available():
    import torch


class GptqQuantizer(QuantizerBase):
    """
    Quantizer of the GPTQ method - for GPTQ the quantizer support calibration of the model through
    `auto_gptq` or `gptqmodel` package. Quantization is done under the hood for users if they load a non-prequantized model.
    """

    requires_calibration = False
    required_packages = ["optimum", "auto_gptq", "gptqmodel"]
    optimum_quantizer = None

    def __init__(self, quantization_config: dict, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if not is_optimum_available():
            raise ImportError("Loading a GPTQ quantized model requires optimum (`pip install optimum`)")
        torch.set_default_dtype(torch.float32)  # 否则下一行会报错
        from optimum.gptq import GPTQQuantizer

        self.optimum_quantizer = GPTQQuantizer.from_dict(self.quantization_config)

    def validate_environment(self, *args, **kwargs):
        if not is_optimum_available():
            raise ImportError("Loading a GPTQ quantized model requires optimum (`pip install optimum`)")
        if is_auto_gptq_available() and is_gptqmodel_available():
            log_warn("Detected gptqmodel and auto-gptq, will use gptqmodel")

        gptq_supports_cpu = (
            is_auto_gptq_available()
            and version.parse(importlib.metadata.version("auto-gptq")) > version.parse("0.4.2")
        ) or is_gptqmodel_available()
        if not gptq_supports_cpu and not torch.cuda.is_available():
            raise RuntimeError("GPU is required to quantize or run quantize model.")
        elif not (is_auto_gptq_available() or is_gptqmodel_available()):
            raise ImportError(
                "Loading a GPTQ quantized model requires gptqmodel (`pip install gptqmodel`) or auto-gptq (`pip install auto-gptq`) library. "
            )
        elif is_auto_gptq_available() and version.parse(importlib.metadata.version("auto_gptq")) < version.parse(
            "0.4.2"
        ):
            raise ImportError(
                "You need a version of auto_gptq >= 0.4.2 to use GPTQ: `pip install --upgrade auto-gptq` or use gptqmodel by `pip install gptqmodel>=1.4.3`."
            )
        elif is_gptqmodel_available() and (
            version.parse(importlib.metadata.version("gptqmodel")) < version.parse("1.4.3")
            or version.parse(importlib.metadata.version("optimum")) < version.parse("1.23.99")
        ):
            raise ImportError("The gptqmodel version should be >= 1.4.3, optimum version should >= 1.24.0")

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = torch.float16
            log_info("Loading the model in `torch.float16`. To overwrite it, set `torch_dtype` manually.")
        elif torch_dtype != torch.float16:
            log_info("We suggest you to set `torch_dtype=torch.float16` for better efficiency with GPTQ.")
        return torch_dtype

    def update_device_map(self, device_map):
        if device_map is None:
            device_map = {"": torch.device("cpu")}
        # Only with auto-gptq do not support CPU, we should move the model to cuda if available.
        if not is_gptqmodel_available() and device_map in ("cpu", {"": torch.device("cpu")}):
            device_map == {"": 0}
        return device_map

    def _process_model_before_weight_loading(self, model, **kwargs):
        # if model.__class__.main_input_name != "input_ids":
        #     raise RuntimeError("We can only quantize pure text model.")

        if self.pre_quantized:
            new_mapping = modify_variable_mapping(model, kwargs['quantization_config'].get('block_name_to_quantize'))
            model.variable_mapping = lambda: new_mapping

            # compat: latest optimum has gptqmodel refactor
            if version.parse(importlib.metadata.version("optimum")) <= version.parse("1.23.99"):
                model = self.optimum_quantizer.convert_model(model)
            else:
                model = self.optimum_quantizer.convert_model(model, **kwargs)

    def _process_model_after_weight_loading(self, model, **kwargs):
        if self.pre_quantized:
            model = self.optimum_quantizer.post_init_model(model)
        else:
            if self.quantization_config.tokenizer is None:
                self.quantization_config.tokenizer = model.name_or_path

            self.optimum_quantizer.quantize_model(model, self.quantization_config.tokenizer)
            model.config.quantization_config = self.optimum_quantizer.to_dict()

    @property
    def is_trainable(self, model):
        return True

    def is_serializable(self, safe_serialization=None):
        return True


def modify_variable_mapping(model, block_name_to_quantize:str):
    '''量化会修改模型的结构，因此也需要修改variable_mapping'''
    old_mapping = model.variable_mapping()
    layers_to_be_replaced = get_layers(model, prefix=block_name_to_quantize)
    new_mapping = {}
    for k in layers_to_be_replaced.keys():
        for o, n in old_mapping.items():
            if not o.startswith(k+'.'):
                new_mapping[o] = n
                continue
            n = n.replace('.weight', '').replace('.bias', '')
            new_mapping.update({
                f"{k}.g_idx": f"{n}.g_idx",
                f"{k}.qweight": f"{n}.qweight",
                f"{k}.qzeros": f"{n}.qzeros",
                f"{k}.scales": f"{n}.scales"
            })

    return new_mapping
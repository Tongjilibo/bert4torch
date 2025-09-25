import inspect
from torch import nn
from typing import List, Dict, Literal
import torch
from torch4keras.snippets import log_warn
from bert4torch.snippets import has_meta_param
from .base import QuantizerBase
import importlib
from packaging import version


class BnbkBitHfQuantizer(QuantizerBase):
    def _process_model_before_weight_loading(self, model:nn.Module, **kwargs):
        '''transformer的load_in_8bit, 源自transformer源代码'''
        quantization_config = kwargs['quantization_config']
        quant_method:Literal['load_in_8bit', 'load_in_4bit'] = quantization_config['quant_method']
        keep_in_fp32_modules:List = quantization_config.get('keep_in_fp32_modules')
        llm_int8_skip_modules:List = quantization_config.get('llm_int8_skip_modules')
        bits_and_bytes_config:Dict = quantization_config.get('bits_and_bytes_config')


        # 兼容transformers新旧版本
        try:
            from transformers.integrations import replace_with_bnb_linear, set_module_quantized_tensor_to_device
        except:
            from transformers.utils.bitsandbytes import replace_with_bnb_linear, set_module_quantized_tensor_to_device
        from transformers.utils.quantization_config import BitsAndBytesConfig
        load_in_8bit = True if quant_method == 'load_in_8bit' else False
        load_in_4bit = True if quant_method == 'load_in_4bit' else False

        # 把meta权重to_empty(device='cpu'), 执行后就不是meta了
        if has_meta_param(model):
            model.apply(model.init_meta_weights)

        if bits_and_bytes_config is None:
            bits_and_bytes_config, kwargs = BitsAndBytesConfig.from_dict(
                config_dict={"load_in_8bit": load_in_8bit, "load_in_4bit": load_in_4bit}, return_unused_kwargs=True, **quantization_config
            )
        elif bits_and_bytes_config is not None:
            load_in_8bit = bits_and_bytes_config.load_in_8bit
            load_in_4bit = bits_and_bytes_config.load_in_4bit
            quantization_config_kwargs = {
                k: v for k, v in kwargs.items() if k in inspect.signature(BitsAndBytesConfig).parameters
            }

            if len(quantization_config_kwargs) > 0:
                raise ValueError(
                    "You can't pass `load_in_8bit` or any other `BitsAndBytesConfig` argument as a kwarg when passing "
                    "`quantization_config` argument at the same time."
                )

        load_in_8bit_skip_modules = bits_and_bytes_config.llm_int8_skip_modules or []

        # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
        modules_to_not_convert = load_in_8bit_skip_modules
        if not isinstance(modules_to_not_convert, list):
            modules_to_not_convert = [modules_to_not_convert]

        modules_to_not_convert.extend([] if keep_in_fp32_modules is None else keep_in_fp32_modules)
        modules_to_not_convert.extend([] if llm_int8_skip_modules is None else llm_int8_skip_modules)

        state_dict = model.state_dict()
        model = replace_with_bnb_linear(model, modules_to_not_convert=modules_to_not_convert, quantization_config=bits_and_bytes_config)

        for key, param in model.named_parameters():
            if param.device == torch.device("meta"):
                set_module_quantized_tensor_to_device(model, key, 'cpu', value=state_dict[key])

        model.is_loaded_in_8bit = self.load_in_8bit = load_in_8bit
        model.is_loaded_in_4bit = self.load_in_4bit = load_in_4bit
        return model

    def _process_model_after_weight_loading(self, model, **kwargs):
        if model.is_loaded_in_8bit:
            model.is_8bit_serializable = self.is_serializable()
        elif model.is_loaded_in_4bit:
            model.is_4bit_serializable = self.is_serializable()
        return model

    def is_serializable(self, safe_serialization=None):
        if self.load_in_4bit:
            _is_4bit_serializable = version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.41.3")

            if not _is_4bit_serializable:
                log_warn(
                    "You are calling `save_pretrained` to a 4-bit converted model, but your `bitsandbytes` version doesn't support it. "
                    "If you want to save 4-bit models, make sure to have `bitsandbytes>=0.41.3` installed."
                )
                return False
            
        elif self.load_in_8bit:
            _bnb_supports_8bit_serialization = version.parse(importlib.metadata.version("bitsandbytes")) > version.parse(
                "0.37.2"
            )

            if not _bnb_supports_8bit_serialization:
                log_warn(
                    "You are calling `save_pretrained` to a 8-bit converted model, but your `bitsandbytes` version doesn't support it. "
                    "If you want to save 8-bit models, make sure to have `bitsandbytes>0.37.2` installed. You will most likely face errors or"
                    " unexpected behaviours."
                )
                return False

        return True

    @property
    def is_trainable(self) -> bool:
        if self.load_in_8bit:
            return version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.37.0")
        elif self.load_in_4bit:    
            return True
        return False
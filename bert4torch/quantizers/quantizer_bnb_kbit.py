import inspect
from torch import nn
from typing import List, Dict, Literal
import torch
from .base import QuantizerBase


class BnbkBitHfQuantizer(QuantizerBase):
    def _process_model_before_weight_loading(
            model:nn.Module, 
            quant_method:Literal['load_in_8bit', 'load_in_4bit'], 
            keep_in_fp32_modules:List=None, 
            llm_int8_skip_modules:List=None, 
            quantization_config:Dict=None, 
            **kwargs
        ):
        '''transformer的load_in_8bit, 源自transformer源代码'''
        # 兼容transformers新旧版本
        try:
            from transformers.integrations import replace_with_bnb_linear, set_module_quantized_tensor_to_device
        except:
            from transformers.utils.bitsandbytes import replace_with_bnb_linear, set_module_quantized_tensor_to_device
        from transformers.utils.quantization_config import BitsAndBytesConfig
        load_in_8bit = True if quant_method == 'load_in_8bit' else False
        load_in_4bit = True if quant_method == 'load_in_4bit' else False

        # 把meta权重to_empty(device='cpu'), 执行后就不是meta了
        if str(next(model.parameters()).device) == 'meta':
            model.apply(model.init_meta_weights)

        if quantization_config is None:
            quantization_config, kwargs = BitsAndBytesConfig.from_dict(
                config_dict={"load_in_8bit": load_in_8bit, "load_in_4bit": load_in_4bit}, return_unused_kwargs=True, **kwargs
            )
        elif quantization_config is not None:
            load_in_8bit = quantization_config.load_in_8bit
            load_in_4bit = quantization_config.load_in_4bit
            quantization_config_kwargs = {
                k: v for k, v in kwargs.items() if k in inspect.signature(BitsAndBytesConfig).parameters
            }

            if len(quantization_config_kwargs) > 0:
                raise ValueError(
                    "You can't pass `load_in_8bit` or any other `BitsAndBytesConfig` argument as a kwarg when passing "
                    "`quantization_config` argument at the same time."
                )

        load_in_8bit_skip_modules = quantization_config.llm_int8_skip_modules or []

        # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
        modules_to_not_convert = load_in_8bit_skip_modules
        if not isinstance(modules_to_not_convert, list):
            modules_to_not_convert = [modules_to_not_convert]

        modules_to_not_convert.extend([] if keep_in_fp32_modules is None else keep_in_fp32_modules)
        modules_to_not_convert.extend([] if llm_int8_skip_modules is None else llm_int8_skip_modules)

        state_dict = model.state_dict()
        model = replace_with_bnb_linear(model, modules_to_not_convert=modules_to_not_convert, quantization_config=quantization_config)

        for key, param in model.named_parameters():
            if param.device == torch.device("meta"):
                set_module_quantized_tensor_to_device(model, key, 'cpu', value=state_dict[key])

        model.is_loaded_in_8bit = load_in_8bit
        model.is_loaded_in_4bit = load_in_4bit
        return model
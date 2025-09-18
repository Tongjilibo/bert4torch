from typing import List, Union, Dict, Literal
import os
import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
import gc
from bert4torch.snippets import log_free, cuda_empty_cache
import time


class PipeLineBase:
    '''基类
    '''
    def __init__(self, checkpoint_path:str, config_path:str=None, device:str=None, 
                 torch_dtype:Literal['double', 'float', 'half', 'float16', 'bfloat16', None]=None, 
                 quantization_config:dict=None, tokenizer_type:Literal['b4t', 'hf']='b4t', **kwargs) -> None:        
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path or checkpoint_path
        self.torch_dtype = torch_dtype
        self.quantization_config = quantization_config
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        if (tokenizer_type == 'b4t') and os.path.exists(os.path.join(self.checkpoint_path, 'vocab.txt')):
            self.tokenizer_type = 'b4t'
        else:
            self.tokenizer_type = 'hf'
        self.model = self.build_model(return_dict=True, **kwargs)
        self.tokenizer = self.build_tokenizer(**kwargs)
        self.config = self.model.config
    
    def build_tokenizer(self, **kwargs):
        # TODO: 默认优先使用默认的Tokenizer，如果没有vocab文件，则使用AutoTokenizer，后续可能修改
        if self.tokenizer_type == 'b4t':
            tokenizer = Tokenizer(os.path.join(self.checkpoint_path, 'vocab.txt'), do_lower_case=True)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        return tokenizer

    def build_model(self, **model_init_config):
        '''初始化model, 方便外部继承'''
        if (not hasattr(self, 'model')) or (self.model is None):
            # 初始化
            model = build_transformer_model(config_path=self.config_path, checkpoint_path=self.checkpoint_path, **model_init_config)
            model.eval()

            # 精度
            if self.torch_dtype == 'double':
                model = model.double()
            elif self.torch_dtype == 'float':
                model = model.float()
            elif self.torch_dtype in {'half', 'float16'}:
                model = model.half()
            elif self.torch_dtype == 'bfloat16':
                model = model.bfloat16()

            # 后量化 post_quantize
            if self.quantization_config is not None:
                model = model.quantize(**self.quantization_config)
            
            if model_init_config.get('device_map') is None:
                # 如果是None, 则说明用户未指定device_map，则需要切换device
                model.to(self.device)
            else:
                # 使用device_map对应的device
                self.device = model.device
            
            return model
        
        if str(self.device) not in str(self.model.device):
            # 切换device到cuda上
            cur = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            log_free(f'{cur} - Moving model from {str(self.model.device)} to {self.device}', prefix='[LOAD]', prefix_color='cyan')
            self.model.to(self.device)
            gc.collect()
            cuda_empty_cache()
            
        return self.model
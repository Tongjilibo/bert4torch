from typing import List, Union, Dict, Literal
import os
import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer


class PipeLineBase:
    '''基类
    '''
    def __init__(self, checkpoint_path:str, config_path:str=None, device:str=None, 
                 tokenizer_type:Literal['b4t', 'hf']='b4t', **kwargs) -> None:        
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path or checkpoint_path
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        if (tokenizer_type == 'b4t') and os.path.exists(os.path.join(self.checkpoint_path, 'vocab.txt')):
            self.tokenizer_type = 'b4t'
        else:
            self.tokenizer_type = 'hf'
        self.tokenizer = self.build_tokenizer()
        self.model = self.build_model(kwargs)
        self.config = self.model.config
    
    def build_tokenizer(self):
        # TODO: 默认优先使用默认的Tokenizer，如果没有vocab文件，则使用AutoTokenizer，后续可能修改
        if self.tokenizer_type == 'b4t':
            tokenizer = Tokenizer(os.path.join(self.checkpoint_path, 'vocab.txt'), do_lower_case=True)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        return tokenizer

    def build_model(self, model_config):
        model = build_transformer_model(config_path=self.config_path, checkpoint_path=self.checkpoint_path, 
                                        return_dict=True, **model_config).to(self.device)
        model.eval()
        return model
from typing import List, Union, Dict
import numpy as np
import os
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding
from bert4torch.tokenizers import Tokenizer
from tqdm.autonotebook import trange


class PipeLineBase:
    '''基类
    '''
    def __init__(self, checkpoint_path:str, device:str=None, **kwargs) -> None:        
        self.checkpoint_path = checkpoint_path
        self.config_path = kwargs.get('config_path') or checkpoint_path
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.tokenizer = self.build_tokenizer()
        self.model = self.build_model(kwargs)
        self.config = self.model.config
    
    def build_tokenizer(self):
        vocab_path = os.path.join(self.checkpoint_path, 'vocab.txt')
        if os.path.exists(vocab_path):
            tokenizer = Tokenizer(vocab_path, do_lower_case=True)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        return tokenizer

    def build_model(self, model_config):
        model = build_transformer_model(config_path=self.config_path, checkpoint_path=self.checkpoint_path, return_dict=True, **model_config).to(self.device)
        model.eval()
        return model
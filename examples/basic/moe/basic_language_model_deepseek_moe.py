#! -*- coding: utf-8 -*-
"""
基本测试：deepseek_moe模型的测试

"""
import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer


dir_path = 'E:/pretrain_ckpt/moe/deepseek-ai@deepseek-moe-16b-chat'
checkpoint_path = dir_path + '/pytorch_model.bin'
config_path = dir_path + '/bert4torch_config.json'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path)
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path)
model = model.to(device)

generation_config = {
    'tokenizer': tokenizer,
    'tokenizer_config': {'skip_special_tokens': True},
    'start_id': None, 
    'end_id': tokenizer.eos_token_id
}

model.generate('你好', **generation_config)
#! -*- coding: utf-8 -*-
"""
基本测试：deepseek_moe模型的测试

"""
import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer


dir_path = 'E:/pretrain_ckpt/moe/deepseek-ai@deepseek-moe-16b-chat'
config_path = dir_path + '/bert4torch_config.json'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=True)
model = build_transformer_model(config_path=config_path, checkpoint_path=dir_path)
model = model.to(device)

generation_config = {
    'tokenizer': tokenizer,
    'tokenizer_config': {"add_special_tokens": False, 'skip_special_tokens': True},
    'bos_token_id':  None, 
    'eos_token_id': tokenizer.eos_token_id,
    'max_new_tokens': 100,
    'top_k': 1
}

query = '你好'
messages = [{"role": "user", "content": "Who are you?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", tokenize=False, add_special_tokens=False)
res = model.generate(prompt, **generation_config)
print(res)
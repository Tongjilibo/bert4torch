#! -*- coding: utf-8 -*-
"""
基本测试：deepseek_moe模型的测试

"""
import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer


model_dir = '/data/pretrain_ckpt/deepseek/deepseek-ai@deepseek-moe-16b-chat'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir)
model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8)
model = model.to(device)

generation_config = {
    'tokenizer': tokenizer, 
    'max_new_tokens': 100, 
    'top_k': 1,
    "tokenizer_config": {"skip_special_tokens": True},
    # TODO: eos_token_id是否需要添加[/INST]标记
    }

while True:
    query = input('\nUser:')
    messages = [{"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", tokenize=False, add_special_tokens=False)
    res = model.generate(prompt, **generation_config)
    print(f'\nAssistant: {res}')
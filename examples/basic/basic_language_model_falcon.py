#! -*- coding: utf-8 -*-
"""
基本测试：falcon模型的测试：还在debug中
使用前需要进行权重转换 https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_falcon.py

falcon-rw-1b:   https://huggingface.co/tiiuae/falcon-rw-1b
"""

import torch
from bert4torch.models import build_transformer_model
from bert4torch.generation import SeqGeneration
from transformers import AutoTokenizer
import platform
import os

choice = 'falcon-rw-1b'
if choice == 'falcon-rw-1b':
    dir_path = '/Users/lb/Documents/pretrain_ckpt/falcon/falcon-rw-1b/'
    include_input = True
else:
    raise ValueError(f'{choice} not in pre maintained choices')

config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = dir_path + '/bert4torch_pytorch_model.bin'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path, use_fast=False)
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path)
# model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8)
model = model.to(device)

tokenizer_config = {'skip_special_tokens': True}
generation = SeqGeneration(model, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, mode='random_sample', 
                           tokenizer_config=tokenizer_config, maxlen=64, default_rtype='logits', use_states=True)


if __name__ == '__main__':
    os_name = platform.system()
    print("Welcome to use falcon model，type `clear` to clear history，type `stop` to stop program")
    while True:
        query = input("\nUser：")
        if query == "stop":
            break
        if query == "clear":
            command = 'cls' if os_name == 'Windows' else 'clear'
            os.system(command)
            print("Welcome to use falcon model，type `clear` to clear history，type `stop` to stop program")
            continue
        response = generation.generate(query, topk=1, include_input=include_input)      
        torch.cuda.empty_cache()  # 清理显存
        print(f"\nfalcon：{response}")
#! -*- coding: utf-8 -*-
"""
基本测试：bloom模型的测试
使用前需要进行权重转换 https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_bloom.py

bloom-560m: https://huggingface.co/bigscience/bloom-560m
bloomz-560m:  https://huggingface.co/bigscience/bloomz-560m
"""

import torch
from bert4torch.models import build_transformer_model
from bert4torch.generation import SeqGeneration
from transformers import AutoTokenizer
import platform
import os

choice = 'bloom-560m'
if choice == 'bloom-560m':
    dir_path = 'E:/pretrain_ckpt/bloom/bloom-560m'
    include_input = True
elif choice == 'bloomz-560m':
    dir_path = 'E:/pretrain_ckpt/bloom/bloomz-560m'
    include_input = False
else:
    raise ValueError(f'{choice} not in pre maintained choices')

config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = dir_path + '/bert4torch_pytorch_model.bin'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path, use_fast=False)
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='bloom')
# model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8)
model = model.to(device)

tokenizer_config = {'skip_special_tokens': True}
generation = SeqGeneration(model, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, mode='random_sample', 
                           tokenizer_config=tokenizer_config, maxlen=20, default_rtype='logits', use_states=True)


if __name__ == '__main__':
    os_name = platform.system()
    print("Welcome to use bloom model，type `clear` to clear history，type `stop` to stop program")
    while True:
        query = input("\nUser：")
        if query == "stop":
            break
        if query == "clear":
            command = 'cls' if os_name == 'Windows' else 'clear'
            os.system(command)
            print("Welcome to use bloom model，type `clear` to clear history，type `stop` to stop program")
            continue
        response = generation.generate(query, topk=1, include_input=include_input)      
        torch.cuda.empty_cache()  # 清理显存
        print(f"\nbloom：{response}")
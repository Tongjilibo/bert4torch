#! -*- coding: utf-8 -*-
"""
基本测试：原生llama模型的测试 https://github.com/facebookresearch/llama
使用前需要进行权重转换 https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_llama_hf.py
"""

import torch
from bert4torch.models import build_transformer_model
from bert4torch.generation import SeqGeneration
from transformers import AutoTokenizer, LlamaTokenizer
import platform
import os

choice = 'llama-2-7b-chat'
if choice == 'llama-2-7b':
    dir_path = 'E:/pretrain_ckpt/llama-2/llama-2-7b'
    with_prompt = False
elif choice == 'llama-2-7b-chat':
    dir_path = 'E:/pretrain_ckpt/llama-2/llama-2-7b-chat'
    with_prompt = True
elif choice == 'llama-2-13b':
    dir_path = 'E:/pretrain_ckpt/llama-2/llama-2-13b'
    with_prompt = False
elif choice == 'llama-2-13b-chat':
    dir_path = 'E:/pretrain_ckpt/llama-2/llama-2-13b-chat'
    with_prompt = True        
else:
    raise ValueError(f'{choice} not in pre maintained choices')

include_input = not with_prompt

config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = dir_path + '/bert4torch_pytorch_model.bin'
spm_path = dir_path + '/tokenizer.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path, use_fast=False)
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='llama').half()
model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8)
model = model.to(device)

tokenizer_config = {'skip_special_tokens': True, 'add_special_tokens': False}
article_completion = SeqGeneration(model, tokenizer, start_id=None, end_id=2, mode='random_sample', tokenizer_config=tokenizer_config,
                                   maxlen=512, default_rtype='logits', use_states=True)

def generate_prompt(query):
    return f'<s>Human: {query}\n</s><s>Assistant: '

if __name__ == '__main__':
    os_name = platform.system()
    print("Welcome to use llama model，type `clear` to clear history，type `stop` to stop program")
    while True:
        query = input("\nUser：")
        if query == "stop":
            break
        if query == "clear":
            command = 'cls' if os_name == 'Windows' else 'clear'
            os.system(command)
            print("Welcome to use llama model，type `clear` to clear history，type `stop` to stop program")
            continue
        if with_prompt:
            query = generate_prompt(query)
        response = article_completion.generate(query, topp=0.95, temperature=0.3, repetition_penalty=1.3, include_input=include_input)      
        torch.cuda.empty_cache()  # 清理显存
        print(f"\nllama：{response}")
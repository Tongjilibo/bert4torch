#! -*- coding: utf-8 -*-
"""
基本测试：原生llama模型的测试 https://github.com/facebookresearch/llama
    权重下载：[Github](https://github.com/facebookresearch/llama)
    [huggingface](https://huggingface.co/huggyllama)
    [torrent](https://pan.baidu.com/s/1yBaYZK5LHIbJyCCbtFLW3A?pwd=phhd)
"""

import torch
from bert4torch.models import build_transformer_model
from bert4torch.generation import SeqGeneration
from transformers import AutoTokenizer, LlamaTokenizer
import platform
import os

model_dir = '/data/pretrain_ckpt/llama/llama-7b'  # llama-7b, llama-13b
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir)
# model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8)
model = model.to(device)

article_completion = SeqGeneration(model, tokenizer, bos_token_id=None, eos_token_id=tokenizer.eos_token_id, 
                                   mode='random_sample', tokenizer_config={'skip_special_tokens': True},
                                   max_length=256, default_rtype='logits', use_states=True)


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
        response = article_completion.generate(query, include_input=True)      
        torch.cuda.empty_cache()  # 清理显存
        print(f"\nllama：{response}")
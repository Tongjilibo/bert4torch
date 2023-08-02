#! -*- coding: utf-8 -*-
"""
基本测试：原生llama模型的测试 https://github.com/facebookresearch/llama
使用前需要进行权重转换 https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_llama_pth.py
"""

import torch
from bert4torch.models import build_transformer_model
from bert4torch.generation import SeqGeneration
from transformers import AutoTokenizer, LlamaTokenizer
import platform
import os

# 原生llama
dir_path = 'E:/pretrain_ckpt/llama/7B'
config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = dir_path + '/bert4torch_pytorch_model.bin'
spm_path = dir_path + '/tokenizer.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path, use_fast=False)
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='llama').half()
# model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8)
model = model.to(device)

tokenizer_config = {'skip_special_tokens': True}
article_completion = SeqGeneration(model, tokenizer, start_id=None, end_id=2, mode='random_sample', tokenizer_config=tokenizer_config,
                                   maxlen=256, default_rtype='logits', use_states=True)


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
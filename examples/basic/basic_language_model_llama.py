#! -*- coding: utf-8 -*-
"""
基本测试：llama系列模型的测试, 7b的fp32精度的单卡占用约27g，fp16的显存占用约14g
使用前需要进行权重转换 https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_llama_facebook.py

[1]. llama模型：https://github.com/facebookresearch/llama
[2]. chinese_llama: https://github.com/ymcui/Chinese-LLaMA-Alpaca
[3]. chinese_alpaca: https://github.com/ymcui/Chinese-LLaMA-Alpaca
[4]. Ziya-LLaMA-13B_v1.1: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1
[5]. Ziya-LLaMA-13B_v1: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
[6]. Ziya-LLaMA-13B_pretrain: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1
"""

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import SpTokenizer
from bert4torch.generation import SeqGeneration
from transformers import AutoTokenizer
import platform
import os

choice = 'chinese_alpaca_plus_7b'
with_prompt = True
include_input = not with_prompt
generate_prompt = lambda instruction: instruction

if choice == 'llama-7b':
    # 原生llama
    dir_path = 'E:/pretrain_ckpt/llama/7B'
    topk, topp, temperature, repetition_penalty = 50, 1, 1, 1.0

elif choice in {'chinese_llama_plus_7b', 'chinese_alpaca_plus_7b'}:
    # chinese-llama-alpaca
    if choice == 'chinese_llama_plus_7b':
        dir_path = 'E:/pretrain_ckpt/llama/chinese-llama/chinese_llama_plus_7b'
    else:
        dir_path = 'E:/pretrain_ckpt/llama/chinese-alpaca/chinese_alpaca_plus_7b'
    topk, topp, temperature, repetition_penalty = 40, 0.9, 0.2, 1.3

    prompt_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n\n{instruction}\n\n### Response:\n\n"
    )
    def generate_prompt(instruction, input=None):
        if input:
            instruction = instruction + '\n' + input
        return prompt_input.format_map({'instruction': instruction})
    
elif choice == 'Ziya-LLaMA-13B_v1.1':
    # ziya模型
    dir_path = 'E:/pretrain_ckpt/llama/[IDEA-CCNL]--Ziya-LLaMA-13B-v1.1'
    topk, topp, temperature, repetition_penalty = 50, 1, 0.8, 1.0

    def generate_prompt(query):
        return f"<human>:{query.strip()}\n<bot>:"


config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = dir_path + '/bert4torch_pytorch_model.bin'
spm_path = dir_path + '/tokenizer.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='llama').half().to(device)
# model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8).to(device)  # 建立模型，加载权重

tokenizer = AutoTokenizer.from_pretrained(dir_path)
article_completion = SeqGeneration(model, tokenizer, start_id=None, end_id=2, mode='random_sample',
                                   maxlen=256, default_rtype='logits', use_states=True)
print('Loading tokenizer done...')


if __name__ == '__main__':
    os_name = platform.system()
    print("Welcome to use llama model，type `clear` to clear history，type `stop` to stop program")
    while True:
        query = input("\nUser：")
        if with_prompt:
            query = generate_prompt(query)
        if query == "stop":
            break
        if query == "clear":
            command = 'cls' if os_name == 'Windows' else 'clear'
            os.system(command)
            print("Welcome to use llama model，type `clear` to clear history，type `stop` to stop program")
            continue
        response = article_completion.generate(query, topk=topk, topp=topp, temperature=temperature, repetition_penalty=repetition_penalty, include_input=include_input)      
        torch.cuda.empty_cache()  # 清理显存
        print(f"\nllama：{response}")
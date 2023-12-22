#! -*- coding: utf-8 -*-
"""
基本测试：ziya系列模型的测试, bert4torch_config.json见readme

Ziya-LLaMA-13B_v1.1: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1
Ziya-LLaMA-13B_v1: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
Ziya-LLaMA-13B_pretrain: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1
"""


choice = 'Ziya-LLaMA-13B_v1.1'
if choice == 'Ziya-LLaMA-13B_v1.1':
    dir_path = 'E:/pretrain_ckpt/llama/IDEA-CCNL@Ziya-LLaMA-13B-v1.1'
    with_prompt = True
elif choice == 'Ziya-LLaMA-13B_v1':
    dir_path = 'E:/pretrain_ckpt/llama/IDEA-CCNL@Ziya-LLaMA-13B-v1'
    with_prompt = True
elif choice == 'Ziya-LLaMA-13B_pretrain':
    dir_path = 'E:/pretrain_ckpt/llama/IDEA-CCNL@Ziya-LLaMA-13B_pretrain'
    with_prompt = False
else:
    raise ValueError(f'{choice} not in pre maintained choices')
include_input = not with_prompt


'''
# 旧实现
import torch
from bert4torch.models import build_transformer_model
from bert4torch.generation import SeqGeneration
from transformers import AutoTokenizer, LlamaTokenizer
import platform
import os


config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('.bin')]
spm_path = dir_path + '/tokenizer.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path, use_fast=False)
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path).half()
model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8)
model = model.to(device)

tokenizer_config = {'skip_special_tokens': True}
article_completion = SeqGeneration(model, tokenizer, start_id=None, end_id=2, mode='random_sample', tokenizer_config=tokenizer_config,
                                   maxlen=256, default_rtype='logits', use_states=True)

def generate_prompt(query):
    return f"<human>:{query.strip()}\n<bot>:"

if __name__ == '__main__':
    os_name = platform.system()
    print("Welcome to use ziya model，type `clear` to clear history，type `stop` to stop program")
    while True:
        query = input("\nUser：")
        if query == "stop":
            break
        if query == "clear":
            command = 'cls' if os_name == 'Windows' else 'clear'
            os.system(command)
            print("Welcome to use ziya model，type `clear` to clear history，type `stop` to stop program")
            continue
        if with_prompt:
            query = generate_prompt(query)
        response = article_completion.generate(query, topk=50, topp=1, temperature=0.8, repetition_penalty=1.0, include_input=include_input)      
        torch.cuda.empty_cache()  # 清理显存
        print(f"\nZiya：{response}")
'''

# 新实现
from bert4torch.chat import CliDemoZiya
tokenizer_config = {'skip_special_tokens': True}
generation_config = {
    'end_id': 2, 
    'mode': 'random_sample', 
    'tokenizer_config': tokenizer_config,
    'maxlen': 256, 
    'default_rtype': 'logits', 
    'use_states': True,
    'include_input': include_input,
}


cli_demo = CliDemoZiya(
    dir_path, generation_config=generation_config,
    quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
    )

if __name__ == '__main__':
    if with_prompt:
        # chat模型
        cli_demo.run()
    else:
        # 预训练模型
        while True:
            query = input('\n输入:')
            response = cli_demo.model.generate(query, **generation_config)
            print(f'续写: {response}')

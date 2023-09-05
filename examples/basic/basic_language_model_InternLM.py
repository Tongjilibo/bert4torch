#! -*- coding: utf-8 -*-
"""书生浦语InternLM的测试
Github: https://github.com/InternLM/InternLM
使用前需要进行权重转换 https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_InternLM.py
"""

import torch
from bert4torch.models import build_transformer_model
from typing import Tuple, List, Union, Iterable
from bert4torch.generation import SeqGeneration
from transformers import AutoTokenizer
import platform
import os

choice = 'internlm-chat-7b'
if choice == 'internlm-chat-7b':
    dir_path = '/Users/lb/Documents/pretrain_ckpt/internlm/internlm-chat-7b'
    with_prompt = True
else:
    raise ValueError(f'{choice} not in pre maintained choices')
include_input = not with_prompt

config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = [f'{dir_path}/bert4torch_pytorch_model-0000{i}-of-00008.bin' for i in range(1, 9)]  # 多文件
spm_path = dir_path + '/tokenizer.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=True)

model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path)
# model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8)  # 解开注释使用量化
model = model.to(device)

def build_inputs(query: str, history: List[Tuple[str, str]] = [], replace=False):
    prompt = ""
    for record in history:
        prompt += f"""<s><|User|>:{record[0]}<eoh>\n<|Bot|>:{record[1]}<eoa>\n"""
    if len(prompt) == 0:
        prompt += "<s>"
    if query is not None:
        prompt += f"""<|User|>:{query}<eoh>\n<|Bot|>:"""
    
    if replace:
        for reg in ['<s>', '</s>', '<eoh>', '<eoa>']:
            prompt = prompt.replace(reg, '')
    return prompt

tokenizer_config = {'skip_special_tokens': True}
chat = SeqGeneration(model, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, mode='random_sample', 
                     tokenizer_config=tokenizer_config, maxlen=256, default_rtype='logits', use_states=True)


if __name__ == '__main__':
    history = []
    os_name = platform.system()
    clear_command = 'cls' if os_name == 'Windows' else 'clear'
    print("Welcome to use InternLM model，type `clear` to clear history，type `stop` to stop program")
    while True:
        query_input = query = input("\nUser：")
        if query == "stop":
            break
        if query == "clear":
            history = []
            os.system(clear_command)
            print("Welcome to use InternLM model，type `clear` to clear history，type `stop` to stop program")
            continue
        if with_prompt:
            query = build_inputs(query, history)
        else:
            query = build_inputs(query, [])

        for response in chat.stream_generate(query, include_input=include_input):
            os.system(clear_command)
            print(build_inputs(None, history+[(query_input, response)], replace=True), flush=True)

        if with_prompt:
            history.append((query_input, response))
        torch.cuda.empty_cache()  # 清理显存

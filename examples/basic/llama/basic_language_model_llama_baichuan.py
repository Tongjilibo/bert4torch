#! -*- coding: utf-8 -*-
"""
基本测试：baichuan模型的测试 https://github.com/baichuan-inc/Baichuan-7B
使用前需要进行权重转换 https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_llama_hf.py
"""

import torch
from bert4torch.models import build_transformer_model
from bert4torch.generation import SeqGeneration
from transformers import AutoTokenizer
from typing import List
import platform
import os

choice = '13B-Chat'

if choice == '7B':
    dir_path = 'E:\\pretrain_ckpt\\llama\\Baichuan-7B'
    with_prompt = False
    maxlen = 64
    topk, topp, temperature, repetition_penalty = 50, 1, 1, 1.1
elif choice == '13B':
    dir_path = 'E:\\pretrain_ckpt\\llama\\Baichuan-13B'
    with_prompt = False
    maxlen = 64
    topk, topp, temperature, repetition_penalty = 50, 1, 1, 1.1
elif choice == '13B-Chat':
    dir_path = 'E:\\pretrain_ckpt\\llama\\Baichuan-13B-Chat'
    with_prompt = True
    maxlen = 4096
    topk, topp, temperature, repetition_penalty = 5, 0.85, 0.3, 1.1
else:
    raise ValueError(f'{choice} not in pre maintained choices')

include_input = not with_prompt

config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = dir_path + '/bert4torch_pytorch_model.bin'
spm_path = dir_path + '/tokenizer.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=True)
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='llama').half()
# model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8)
model = model.to(device)

tokenizer_config = {'skip_special_tokens': True}
if not with_prompt:
    article_completion = SeqGeneration(model, tokenizer, start_id=None, end_id=2, mode='random_sample', tokenizer_config=tokenizer_config,
                                    maxlen=maxlen, default_rtype='logits', use_states=True)
else:
    class Chat(SeqGeneration):
        def pre_process(self, input_ids):
            return [input_ids]
        
    article_completion = Chat(model, tokenizer, start_id=None, end_id=2, mode='random_sample', tokenizer_config=tokenizer_config,
                                    maxlen=maxlen, default_rtype='logits', use_states=True)


def build_input_ids(messages: List[dict], model_max_length=4096, max_new_tokens=2048):
    user_token_id = 195
    assistant_token_id = 196
    max_input_tokens = model_max_length - max_new_tokens
    max_input_tokens = max(model_max_length // 2, max_input_tokens)
    total_input, round_input = [], []
    for i, message in enumerate(messages[::-1]):
        content_tokens = tokenizer.encode(message['content'])
        if message['role'] == 'user':
            round_input = [user_token_id] + content_tokens + round_input
            if total_input and len(total_input) + len(round_input) > max_input_tokens:
                break
            else:
                total_input = round_input + total_input
                if len(total_input) >= max_input_tokens:
                    break
                else:
                    round_input = []
        elif message['role'] == 'assistant':
            round_input = [assistant_token_id] + content_tokens + [tokenizer.eos_token_id] + round_input
        else:
            raise ValueError(f"message role not supported yet: {message['role']}")
    total_input = total_input[-max_input_tokens:]  # truncate left
    total_input.append(assistant_token_id)
    return total_input

if __name__ == '__main__':
    history = []
    os_name = platform.system()
    print("Welcome to use baichuan model，type `clear` to clear history，type `stop` to stop program")
    while True:
        query = input("\nUser：")
        if query == "stop":
            break
        if query == "clear":
            command = 'cls' if os_name == 'Windows' else 'clear'
            history = []
            os.system(command)
            print("Welcome to use baichuan model，type `clear` to clear history，type `stop` to stop program")
            continue
        
        if with_prompt:
            history.append({"role": "user", "content": query})
            query = build_input_ids(history)
        response = article_completion.generate(query, topk=topk, topp=topp, temperature=temperature, repetition_penalty=1.1, include_input=include_input)      
        torch.cuda.empty_cache()  # 清理显存
        print(f"\nbaichuan：{response}")
        
        if with_prompt:
            history.append({"role": "assistant", "content": response})

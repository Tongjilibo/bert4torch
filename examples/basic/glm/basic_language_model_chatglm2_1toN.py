#! -*- coding: utf-8 -*-
# 基本测试：chatglm2的对话测试
# 官方项目：https://github.com/THUDM/ChatGLM2-6B
# hf链接：https://huggingface.co/THUDM/chatglm2-6b


'''
# 旧实现

import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer
from bert4torch.generation import SeqGeneration
import platform
import os
import re

choice = 'default'  # default, int4, 32k
if choice == 'default':
    dir_path = "E:/pretrain_ckpt/glm/chatglm2-6B"
elif choice == 'int4':
    dir_path = "E:/pretrain_ckpt/glm/chatglm2-6B-int4"
elif choice == '32k':
    dir_path = "E:/pretrain_ckpt/glm/chatglm2-6B-32k"
else:
    raise ValueError(f'{choice} not in pre maintained choices')

checkpoint_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('.bin')]
config_path = dir_path + '/bert4torch_config.json'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'

tokenizer = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=True)
# 建立模型，加载权重
if choice in {'default', '32k'}:
    encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path).half().to(device)
    # encoder = encoder.quantize(quantization_method='cpm_kernels', quantization_bit=8).to(device)
else:
    # 在config中已经写入了量化的配置参数
    encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path).to(device)

generation = SeqGeneration(encoder, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, mode='random_sample',
                           maxlen=2048, default_rtype='logits', use_states=True)

def process_response(response):
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    return response

def chat(query, history=[]):
    # 这里和chatglm的区别是，chatglm的第一轮对话prompt=query, 不加[Round 1]这些前缀
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n".format(i+1, old_query, response)
    prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history)+1, query)

    response = generation.generate(prompt, topk=50, topp=0.7, temperature=0.95, n=5)
    if isinstance(response, list):
        response = [process_response(i) for i in response]
    else:
        response = process_response(response)
    return response


if __name__ == '__main__':
    history = []
    response = chat('如何查询天气？', history=history)
    if isinstance(response, str):
        print(response)
    else:
        for i in response:
            print(''.center(60, '='))
            print(i)
'''

# 新实现
from bert4torch.chat import Chatglm2
model_path = "E:/pretrain_ckpt/glm/chatglm2-6B"
# model_path = "E:/pretrain_ckpt/glm/chatglm2-6B-int4"
# model_path = "E:/pretrain_ckpt/glm/chatglm2-6B-32k"

generation_config  = {'mode':'random_sample',
                      'maxlen':2048, 
                      'default_rtype':'logits', 
                      'use_states':True,
                      'n': 5
                      }

chat = Chatglm2(model_path, **generation_config)
response = chat.generate('如何查询天气？')

if isinstance(response, str):
    print(response)
else:
    for i in response:
        print(''.center(60, '='))
        print(i)
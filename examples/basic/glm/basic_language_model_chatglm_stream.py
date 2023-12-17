#! -*- coding: utf-8 -*-
# 基本测试：chatglm的对话测试

# 官方项目：https://github.com/THUDM/ChatGLM-6B
# hf链接：https://huggingface.co/THUDM/chatglm-6b
# fp16半精度下显存占用14G
# 20230406 官方项目对20000个和图像相关的进行的裁剪，因此本项目之前裁剪及tokenize的作废，使用最新的tokenize不需要进行offset

# 旧实现
import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer
from bert4torch.generation import SeqGeneration
import platform
import os
import re


choice = 'default'  # v1.1.0, default, int4, int8
if choice == 'default':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B"
elif choice == 'v1.1.0':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-v1_1_0"
elif choice == 'int4':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-int4"
elif choice == 'int8':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-int8"
else:
    raise ValueError(f'{choice} not in pre maintained choices')

config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('.bin')]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'

tokenizer = AutoTokenizer.from_pretrained(dir_path.replace('/', '\\'), trust_remote_code=True)
# 建立模型，加载权重
if choice in {'default', 'v1.1.0'}:
    encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path).half()
    encoder = encoder.quantize(quantization_method='cpm_kernels', quantization_bit=8).to(device)
else:
    # 在config中已经写入了量化的配置参数
    encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path).to(device)

generation = SeqGeneration(encoder, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, mode='random_sample',
                           maxlen=2048, default_rtype='logits', use_states=True)

def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt

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
    if not history:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

    for response in generation.stream_generate(prompt, topk=50, topp=0.7, temperature=0.95):
        response = process_response(response)
        new_history = history + [(query, response)]
        yield response, new_history


def main():
    history = []
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        for response, history in chat(query, history=history):
            os.system(clear_command)
            print(build_prompt(history), flush=True)
        os.system(clear_command)
        print(build_prompt(history), flush=True)
        torch.cuda.empty_cache()  # 清理显存

if __name__ == '__main__':
    main()


'''
# 新实现
from bert4torch.chat import ChatCliDemo
import re

choice = 'default'  # v1.1.0, default, int4, int8
quantization_config = None
if choice == 'default':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B"
    # quantization_config = {'quantization_method': 'cpm_kernels', 'quantization_bit': 8}
elif choice == 'v1.1.0':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-v1_1_0"
    # quantization_config = {'quantization_method': 'cpm_kernels', 'quantization_bit': 8}
elif choice == 'int4':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-int4"
elif choice == 'int8':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-int8"
else:
    raise ValueError(f'{choice} not in pre maintained choices')

generation_config = {'mode': 'random_sample',
                     'maxlen': 2048, 
                     'default_rtype':'logits', 
                     'use_states':True}

class Demo(ChatCliDemo):
    def build_prompt(self, query, history) -> str:
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        return prompt
    
    def process_response(self, response, *args):
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

demo = Demo(dir_path, generation_config=generation_config, quantization_config=quantization_config)
demo.run()
'''
#! -*- coding: utf-8 -*-
# 基本测试：chatglm的对话测试, 使用前请先使用转换脚本转一下权重
# 权重转换脚本：https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_chatglm.py
# 需要安装git上最新版bert4torch: pip install git+https://github.com/Tongjilibo/bert4torch
# 官方项目：https://github.com/THUDM/ChatGLM-6B
# hf链接：https://huggingface.co/THUDM/chatglm-6b
# fp16半精度下显存占用14G
# 20230406 官方项目对20000个和图像相关的进行的裁剪，因此本项目之前裁剪及tokenize的作废，使用最新的tokenize不需要进行offset

import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer
from bert4torch.snippets import AutoRegressiveDecoder, SeqGeneration
import platform
import os
import signal
import re


dir_path = "F:/Projects/pretrain_ckpt/chatglm/6B"
config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = [dir_path + f'/bert4torch_pytorch_model_{i}.bin' for i in range(1,9)]  # 可加载单个，也可以加载多个
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

tokenizer = AutoTokenizer.from_pretrained(dir_path.replace('/', '\\'), trust_remote_code=True)
encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='glm').half().quantize(8).to(device)  # 建立模型，加载权重


class Chat(SeqGeneration):
    def pre_process(self, text):
        return [tokenizer.encode(text)]
    def post_process(self, output_ids):
        return tokenizer.decode(output_ids[0].cpu().numpy())
generation = Chat(encoder, tokenizer, start_id=None, end_id=tokenizer.encode(['<eop>'])[0], mode='random_sample',
                  maxlen=2048, default_rtype='logits', use_states=True)

def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt

def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

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
    global stop_stream
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
        count = 0
        for response, history in chat(query, history=history):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)
        torch.cuda.empty_cache()  # 清理显存

if __name__ == '__main__':
    main()
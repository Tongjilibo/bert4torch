#! -*- coding: utf-8 -*-
# 基本测试：chatglm的对话测试
# 转换脚本：https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_chatglm.py
# fp16半精度下显存占用14G

import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer
from bert4torch.snippets import AutoRegressiveDecoder
import platform
import os


dir_path = "F:/Projects/pretrain_ckpt/chatglm/6B"
config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = dir_path + '/bert4torch_pytorch_model.bin'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=True)
encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='glm').half().to(device)  # 建立模型，加载权重

class Chat(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = torch.cat([inputs[0], output_ids], 1)
        logits = encoder.predict([token_ids])
        return logits[:, -1, :]

    def generate(self, text, n=1, topk=50, topp=0.7, temperature=0.95):
        token_ids = tokenizer.encode(text)
        results = self.random_sample([token_ids], n, topk=topk, topp=topp,  temperature=temperature)  # 基于随机采样
        return tokenizer.decode(results[0].cpu().numpy())
generation = Chat(start_id=None, end_id=150005, maxlen=2048, device=device)

def chat(query, history=[]):
    if not history:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

    response = generation.generate(prompt, topk=50, topp=0.7, temperature=0.95)
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    history = history + [(query, response)]
    return response, history


if __name__ == '__main__':
    os_name = platform.system()
    history = []
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query == "stop":
            break
        if query == "clear":
            history = []
            command = 'cls' if os_name == 'Windows' else 'clear'
            os.system(command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        response, history = chat(query, history=history)
        print(f"ChatGLM-6B：{response}")

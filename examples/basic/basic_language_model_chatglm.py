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


dir_path = "F:/Projects/pretrain_ckpt/chatglm/6B"
config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = [dir_path + f'/bert4torch_pytorch_model_{i}.bin' for i in range(1,9)]  # 可加载单个，也可以加载多个
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path.replace('/', '\\'), trust_remote_code=True)
encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='glm').half().quantize(8).to(device)  # 建立模型，加载权重

# 第一种方式: 自定义解码
# class Chat(AutoRegressiveDecoder):
#     @AutoRegressiveDecoder.wraps(default_rtype='logits')
#     def predict(self, inputs, output_ids, states):
#         token_ids = torch.cat([inputs[0], output_ids], 1)
#         logits = encoder.predict([token_ids])
#         return logits[:, -1, :]

#     def generate(self, text, n=1, topk=50, topp=0.7, temperature=0.95):
#         token_ids = tokenizer.encode(text)
#         results = self.random_sample([token_ids], n, topk=topk, topp=topp,  temperature=temperature)  # 基于随机采样
#         return tokenizer.decode(results[0].cpu().numpy())
# generation = Chat(start_id=None, end_id=tokenizer.encode(['<eop>'])[0], maxlen=2048, device=device)

# 第二种方式：调用封装好的接口，可使用cache
class Chat(SeqGeneration):
    def pre_process(self, text):
        return [tokenizer.encode(text)]
    def post_process(self, input_text, output_ids):
        return tokenizer.decode(output_ids[0].cpu().numpy())
generation = Chat(encoder, tokenizer, start_id=None, end_id=tokenizer.encode(['<eop>'])[0], mode='random_sample',
                  maxlen=2048, default_rtype='logits', use_states=True)

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
    torch.cuda.empty_cache()  # 清理显存
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

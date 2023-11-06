#! -*- coding: utf-8 -*-
# 基本测试：chatglm3的对话测试
# 官方项目：https://github.com/THUDM/ChatGLM3-6B
# hf链接：https://huggingface.co/THUDM/chatglm3-6b

import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer
import platform
import os


choice = 'default'
if choice == 'default':
    dir_path = "E:/pretrain_ckpt/glm/chatglm3-6b"
else:
    raise ValueError(f'{choice} not in pre maintained choices')

config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('.bin')]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'

tokenizer = AutoTokenizer.from_pretrained(dir_path.replace('/', '\\'), trust_remote_code=True)
# 建立模型，加载权重
if choice == 'default':
    model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path).half().to(device)
    # model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8).to(device)
else:
    # 在config中已经写入了量化的配置参数
    model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path).to(device)

gen_kwargs = {"maxlen": 2048, 
              "topk": 50, 
              "topp": 0.7, 
              "temperature": 0.95,
              "start_id": None,
              "end_id": [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), 
                         tokenizer.get_command("<|observation|>")],
              "mode": 'random_sample',
              "default_rtype": 'logits',
              "use_states": True}

def build_prompt(history):
    prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for hist in history:
        if hist['role'] == 'user':
            query = hist['content']
            prompt += f"\n\n用户：{query}"
        elif hist['role'] == 'assistant':
            response = hist['content']
            prompt += f"\n\nChatGLM-6B：{response}"
    return prompt

def chat(query, history=[]):
    # 在tokenizer中已经封装了build_chat_input，所以无需传入tokenizer
    input_ids = tokenizer.build_chat_input(query, history=history, role="user")['input_ids']
    for outputs in model.stream_generate(input_ids, **gen_kwargs):
        response = tokenizer.decode(outputs.cpu().tolist()[0])
        yield response

def process_response(output, history):
    content = ""
    for response in output.split("<|assistant|>"):
        metadata, content = response.split("\n", maxsplit=1)
        if not metadata.strip():
            content = content.strip()
            history[-1] = {"role": "assistant", "metadata": metadata, "content": content}
            content = content.replace("[[训练时间]]", "2023年")
        else:
            history[-1] = {"role": "assistant", "metadata": metadata, "content": content}
            if history[0]["role"] == "system" and "tools" in history[0]:
                content = "\n".join(content.split("\n")[1:-1])
                def tool_call(**kwargs):
                    return kwargs
                parameters = eval(content)
                content = {"name": metadata.strip(), "parameters": parameters}
            else:
                content = {"name": metadata.strip(), "content": content}
    return content, history
    
def main():
    history = []
    print("欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": ""})

        for response in chat(query, history=history):
            if response and response[-1] != "�":
                response, history = process_response(response, history)
                os.system(clear_command)
                print(build_prompt(history), flush=True)

        os.system(clear_command)
        print(build_prompt(history), flush=True)
        torch.cuda.empty_cache()  # 清理显存

if __name__ == '__main__':
    main()
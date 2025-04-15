#! -*- coding: utf-8 -*-
"""书生浦语InternLM的测试
Github: https://github.com/InternLM/InternLM
bert4torch_config.json见readme

[1] internlm-chat-7b模型：https://huggingface.co/internlm/internlm-chat-7b
"""

from bert4torch.pipelines import Chat
from bert4torch.snippets import log_info
from bert4torch.models import build_transformer_model
from PIL import Image
import requests


# InternVL2_5-1B
# InternVL2_5-2B
# InternVL2_5-4B
# InternVL2_5-8B
model_dir = 'E:/data/pretrain_ckpt/OpenGVLab/InternVL2_5-1B'
image1 = Image.open(requests.get("https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg", stream=True).raw).convert('RGB')
image2 = Image.open(requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg", stream=True).raw).convert('RGB')


def chat_demo():
    generation_config = {
        'top_p': 0.8, 
        'temperature': 1,
        'repetition_penalty': 1.005, 
        'top_k': 40
    }

    system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."""

    demo = Chat(model_dir, 
                    system=system_prompt,
                    generation_config=generation_config,
                    mode='raw'
                    )
    log_info('pure-text conversation (纯文本对话)')
    question = '你是谁'
    response = demo.chat(question)
    print(f'User: {question}\nAssistant: {response}\n')


    log_info('single-image single-round conversation (单图单轮对话)')
    question = '图片中描述的是什么'
    response = demo.chat(question, image1)
    print(f'User: {question}\nAssistant: {response}\n')


    log_info('single-image multi-round conversation (单图多轮对话)')
    question = '图片中描述的是什么'
    response, history = demo.chat(question, image1, return_history=True)
    print(f'User: {question}\nAssistant: {response}')
    question = '图片中的兔子是什么颜色的'
    response, history = demo.chat(question, history=history, return_history=True)
    print(f'User: {question}\nAssistant: {response}\n')


    log_info('multi-image multi-round conversation, separate images (多图多轮对话，独立图像)')
    question = '图片中描述的是什么'
    response, history = demo.chat(question, [image1, image2], return_history=True)
    print(f'User: {question}\nAssistant: {response}\n')
    question = '这两张图片的区别是什么？'
    response, history = demo.chat(question, history=history, return_history=True)
    print(f'User: {question}\nAssistant: {response}')

if __name__ == '__main__':
    chat_demo()
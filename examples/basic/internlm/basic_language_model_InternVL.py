#! -*- coding: utf-8 -*-
"""书生浦语InternLM的测试
Github: https://github.com/InternLM/InternLM
bert4torch_config.json见readme

[1] internlm-chat-7b模型：https://huggingface.co/internlm/internlm-chat-7b
"""

from bert4torch.pipelines import Chat
from bert4torch.snippets import log_info
from PIL import Image


model_dir = '/data/pretrain_ckpt/OpenGVLab/InternVL2_5-1B'
# InternVL2_5-2B
# InternVL2_5-4B
# InternVL2_5-8B
image1 = Image.open('./data/images/rabbit.jpg').convert('RGB')
image2 = Image.open('./data/images/beach.jpeg').convert('RGB')


def chat_demo():
    generation_config = {
        'top_p': 0.8, 
        'temperature': 1,
        'repetition_penalty': 1.005, 
        'top_k': 40
    }

    system_prompt = """你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"""

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
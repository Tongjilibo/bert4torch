#! -*- coding: utf-8 -*-
# 基本测试：chatglm2的对话测试, openai格式的api

from bert4torch.pipelines import ChatGlm2OpenaiApi
from bert4torch.pipelines import ChatOpenaiClient, ChatOpenaiClientSseclient

def call_openai(stream=True):
    url = 'http://127.0.0.1:8000'
    messages = [
            {"content": "你好", "role": "user"},
            {"content": "你好，我是法律大模型", "role": "assistant"},
            {"content": "基金从业可以购买股票吗", "role": "user"}
            ]
    client = ChatOpenaiClient(url)
    if stream:
        for token in client.stream_chat(messages):
            print(token, end='', flush=True)
    else:
        print(client.chat(messages))


def call_sseclient():
    url = 'http://127.0.0.1:8000/chat/completions'
    body = {
                "messages": [
                    {"content": "你好", "role": "user"},
                    {"content": "你好，我是法律大模型", "role": "assistant"},
                    {"content": "基金从业可以购买股票吗", "role": "user"}],
                "model": "default",
                "stream": True
            }


    client = ChatOpenaiClientSseclient(url)

    # 测试打印
    client.stream_chat_cli(body)

    # 测试返回
    print('\n-------------------------------------------')
    for token in client.stream_chat(body):
        print(token, end='', flush=True)


if __name__ == '__main__':
    model_path = "E:/pretrain_ckpt/glm/chatglm2-6B"
    generation_config  = {'mode':'random_sample',
                        'max_length':2048, 
                        'default_rtype':'logits', 
                        'use_states':True,
                        'offload_when_nocall': 'cpu',
                        'max_callapi_interval': 30,
                        'scheduler_interval': 10
                        }

    chat = ChatGlm2OpenaiApi(model_path, **generation_config)
    chat.run()

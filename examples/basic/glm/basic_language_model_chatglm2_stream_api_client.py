#! -*- coding: utf-8 -*-
# 基本测试：chatglm2的对话测试
# 官方项目：https://github.com/THUDM/ChatGLM2-6B
# hf链接：https://huggingface.co/THUDM/chatglm2-6b
import requests
import sseclient
import json
from bert4torch.chat import ChatOpenaiClient


url = 'http://127.0.0.1:8000/chat'
body = {
                "messages": [
                    {"content": "你好",
                     "role": "user"},
                    {"content": "你好，我是法律大模型",
                     "role": "assistant"},
                    {"content": "基金从业可以购买股票吗",
                     "role": "user"}],
                "model": "default",
                "stream": True
            }


reqHeaders = {'Accept': 'text/event-stream'}
request = requests.post(url, stream=True, headers=reqHeaders, json=body)
client = sseclient.SSEClient(request)
for event in client.events():
    if event.data != '[DONE]':
        data = json.loads(event.data)['choices'][0]['delta']
        if 'content' in data:
            print(data['content'], end="", flush=True)
            # yield data['content']


client = ChatOpenaiClient(url)
client.post(body, verbose=1)

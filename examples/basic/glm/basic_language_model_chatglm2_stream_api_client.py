#! -*- coding: utf-8 -*-
# 基本测试：chatglm2的对话测试
# 官方项目：https://github.com/THUDM/ChatGLM2-6B
# hf链接：https://huggingface.co/THUDM/chatglm2-6b
from bert4torch.chat import ChatOpenaiClient


url = 'http://10.16.38.1:8090/chat'
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


client = ChatOpenaiClient(url)

# 测试打印
client.post_test(body)

# 测试返回
print('\n-------------------------------------------')
for token in client.post(body):
    print(token, end='', flush=True)
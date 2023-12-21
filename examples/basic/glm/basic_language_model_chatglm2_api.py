#! -*- coding: utf-8 -*-
# 基本测试：chatglm2的对话测试, openai格式的api


from bert4torch.chat import ChatOpenaiApiChatglm2

model_path = "E:/pretrain_ckpt/glm/chatglm2-6B"
generation_config  = {'mode':'random_sample',
                      'maxlen':2048, 
                      'default_rtype':'logits', 
                      'use_states':True
                      }

chat = ChatOpenaiApiChatglm2(model_path, **generation_config)
chat.run()


'''
# 基本测试：chatglm2的对话测试，调用openai的api接口


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


client = ChatOpenaiClient(url)

# 测试打印
client.post_test(body)

# 测试返回
print('\n-------------------------------------------')
for token in client.post(body):
    print(token, end='', flush=True)
'''
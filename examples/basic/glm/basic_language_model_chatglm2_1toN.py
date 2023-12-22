#! -*- coding: utf-8 -*-
# 基本测试：chatglm2的对话测试
# 官方项目：https://github.com/THUDM/ChatGLM2-6B
# hf链接：https://huggingface.co/THUDM/chatglm2-6b


# 新实现
from bert4torch.chat import Chatglm2
model_path = "E:/pretrain_ckpt/glm/chatglm2-6B"
# model_path = "E:/pretrain_ckpt/glm/chatglm2-6B-int4"
# model_path = "E:/pretrain_ckpt/glm/chatglm2-6B-32k"

generation_config  = {'mode':'random_sample',
                      'maxlen':2048, 
                      'default_rtype':'logits', 
                      'use_states':True,
                      'n': 5
                      }

chat = Chatglm2(model_path, **generation_config)
response = chat.generate('如何查询天气？')

if isinstance(response, str):
    print(response)
else:
    for i in response:
        print(''.center(60, '='))
        print(i)
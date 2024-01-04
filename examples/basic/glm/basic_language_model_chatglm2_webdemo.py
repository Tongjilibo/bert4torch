#! -*- coding: utf-8 -*-
# 基本测试：chatglm2的对话测试
# 官方项目：https://github.com/THUDM/ChatGLM2-6B
# hf链接：https://huggingface.co/THUDM/chatglm2-6b

from bert4torch.pipelines import ChatGlm2WebGradio

model_path = "E:/pretrain_ckpt/glm/chatglm2-6B"
generation_config  = {'mode':'random_sample',
                      'maxlen':2048, 
                      'default_rtype':'logits', 
                      'use_states':True
                      }
chat = ChatGlm2WebGradio(model_path, **generation_config)

if __name__ == '__main__':
    chat.run(share=True, inbrowser=True)

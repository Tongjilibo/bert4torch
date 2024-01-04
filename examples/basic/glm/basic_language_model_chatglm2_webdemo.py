#! -*- coding: utf-8 -*-
# 基本测试：chatglm2的对话测试
# 官方项目：https://github.com/THUDM/ChatGLM2-6B
# hf链接：https://huggingface.co/THUDM/chatglm2-6b

from bert4torch.pipelines import ChatGlm2WebGradio, ChatGlm2WebStreamlit

choice = 'gradio'
model_path = "E:/pretrain_ckpt/glm/chatglm2-6B"
generation_config  = {'mode':'random_sample',
                      'maxlen':2048, 
                      'default_rtype':'logits', 
                      'use_states':True
                      }


if __name__ == '__main__':
    if choice == 'gradio':
        chat = ChatGlm2WebGradio(model_path, **generation_config)
        chat.run(share=True, inbrowser=True)
    elif choice == 'streamlit':
        chat = ChatGlm2WebStreamlit(model_path, **generation_config)
        chat.run()
        # 需要使用`streamlit run basic_language_model_chatglm2_webdemo.py`

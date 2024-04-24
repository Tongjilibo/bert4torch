#! -*- coding: utf-8 -*-
# 基本测试：chatglm的对话测试

# 官方项目：https://github.com/THUDM/ChatGLM-6B
# hf链接：https://huggingface.co/THUDM/chatglm-6b
# fp16半精度下显存占用14G
# 20230406 官方项目对20000个和图像相关的进行的裁剪，因此本项目之前裁剪及tokenize的作废，使用最新的tokenize不需要进行offset

from bert4torch.pipelines import ChatGlmWebGradio


model_name = 'chatglm-6B'  # chatglm-6B, chatglm-6B-int4, chatglm-6B-int8
dir_path = f"E:/pretrain_ckpt/glm/{model_name}"


generation_config  = {'mode':'random_sample',
                      'max_length':2048, 
                      'default_rtype':'logits', 
                      'use_states':True
                      }
chat = ChatGlmWebGradio(dir_path, **generation_config)


if __name__ == '__main__':
    chat.run(share=True, inbrowser=True)

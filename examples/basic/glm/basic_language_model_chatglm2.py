#! -*- coding: utf-8 -*-
# 基本测试：chatglm2的对话测试
# 官方项目：https://github.com/THUDM/ChatGLM2-6B
# hf链接：https://huggingface.co/THUDM/chatglm2-6b

from bert4torch.pipelines import ChatGlm2Cli


model_path = "E:/pretrain_ckpt/glm/chatglm2-6B"
# model_path = "E:/pretrain_ckpt/glm/chatglm2-6B-int4"
# model_path = "E:/pretrain_ckpt/glm/chatglm2-6B-32k"

generation_config  = {'mode':'random_sample',
                      'maxlen':2048, 
                      'default_rtype':'logits', 
                      'use_states':True
                      }

demo = ChatGlm2Cli(model_path, **generation_config)

if __name__ == '__main__':
    choice = 'cli'  # cli, gen_1toN

    if choice == 'cli':
        # 命令行demo
        demo.run(stream=True)
    elif choice == 'gen_1toN':
        # 一次性输出N条记录
        demo.generation_config['n'] = 5
        res = demo.chat('如何查询天气？')
        print(res)

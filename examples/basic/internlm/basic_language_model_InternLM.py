#! -*- coding: utf-8 -*-
"""书生浦语InternLM的测试
Github: https://github.com/InternLM/InternLM
bert4torch_config.json见readme

[1] internlm-chat-7b模型：https://huggingface.co/internlm/internlm-chat-7b
"""

from bert4torch.pipelines import Chat
import re

# internlm-7b, internlm-chat-7b
# internlm2-1_8b, internlm2-chat-1_8b, internlm2-7b, internlm2-chat-7b, internlm2-20b, internlm2-chat-20b
# internlm2_5-7b, internlm2_5-7b-chat, internlm2_5-7b-chat-1m
model_dir = 'E:/data/pretrain_ckpt/internlm/internlm2-chat-1_8b'

generation_config = {
    'top_p': 0.8, 
    'temperature': 0.8,
    'include_input': False if re.search('chat', model_dir) else True
}

cli_demo = Chat(model_dir, 
                system='You are an AI assistant whose name is InternLM (书生·浦语).\n- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.',
                generation_config=generation_config,
                mode='cli'
                )
cli_demo.run()
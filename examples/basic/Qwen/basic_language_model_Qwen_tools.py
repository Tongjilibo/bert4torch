#! -*- coding: utf-8 -*-
"""通义千问Qwen的测试
阿里云的通义千问: https://github.com/QwenLM/Qwen
bert4torch_config.json见readme
- [Qwen-7B](https://huggingface.co/Qwen/Qwen-7B)
- [Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)
"""
from bert4torch.pipelines import ChatQwenCli, ChatQwenOpenaiApi, ChatQwen2Cli, ChatQwen2OpenaiApi
from bert4torch.pipelines import ChatOpenaiClient, ChatOpenaiClientSseclient
import re


functions = [
        {
            'name_for_human': '谷歌搜索',
            'name_for_model': 'google_search',
            'description_for_model': '谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。 Format the arguments as a JSON object.',
            'parameters': [{
                'name': 'search_query',
                'description': '搜索关键词或短语',
                'required': True,
                'schema': {
                    'type': 'string'
                },
            }],
        },
        {
            'name_for_human': '文生图',
            'name_for_model': 'image_gen',
            'description_for_model': '文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL。 Format the arguments as a JSON object.',
            'parameters': [{
                'name': 'prompt',
                'description': '英文关键词，描述了希望图像具有什么内容',
                'required': True,
                'schema': {
                    'type': 'string'
                },
            }],
        },
    ]

# Qwen-1_8B-Chat  Qwen-7B-Chat Qwen-14B-Chat
# Qwen1.5-0.5B-Chat  Qwen1.5-1.8B-Chat  Qwen1.5-7B-Chat  Qwen1.5-14B-Chat
# Qwen2-0.5B-Instruct  Qwen2-1.5B-Instruct  Qwen2-7B-Instruct
model_dir = f'/data/pretrain_ckpt/Qwen/Qwen2-7B-Instruct'

generation_config = {'top_k': 40, 'repetition_penalty': 1.1, 'temperature': 0.7, 'use_states': False}
if 'Qwen-' in model_dir:
    Chat = ChatQwenCli
    # Chat = ChatQwenOpenaiApi
else:
    Chat = ChatQwen2Cli
    # Chat = ChatQwen2OpenaiApi

demo = Chat(model_dir, 
            system='你是一个乐于助人的AI助手。', 
            generation_config=generation_config,
            # quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
            )

demo.run(functions=functions)

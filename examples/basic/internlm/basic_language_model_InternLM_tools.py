#! -*- coding: utf-8 -*-
"""书生浦语InternLM的测试
Github: https://github.com/InternLM/InternLM
bert4torch_config.json见readme

[1] internlm-chat-7b模型：https://huggingface.co/internlm/internlm-chat-7b
"""

from bert4torch.pipelines import Chat
import re

# internlm-chat-7b
# iinternlm2-chat-1_8b, internlm2-chat-7b, internlm2-chat-20b
# internlm2_5-7b-chat, internlm2_5-7b-chat-1m
model_dir = 'E:/data/pretrain_ckpt/internlm/internlm2_5-7b-chat'

generation_config = {
    'top_p': 0.8, 
    'temperature': 0.1,
    'top_k': None,
}


functions = [
    {
        'name': 'IPythonInterpreter',
        'description': "When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 60.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is disabled. Do not make external web requests or API calls as they will fail.",
        'parameters': [{
            'name': 'command',
            'type': 'STRING',
            'description': 'Python code'
        }, {
            'name': 'timeout',
            'type': 'NUMBER',
            'description': 'Upper bound of waiting time for Python script execution.'
        }],
        'required': ['command'],
        'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'
    },
    {
        'name': 'ArxivSearch.get_arxiv_article_information',
        'description': 'Run Arxiv search and get the article meta information.',
        'parameters': [{
            'name': 'query',
            'type': 'STRING',
            'description': 'the content of search query'
        }],
        'required': ['query'],
        'return_data': [{
            'name': 'content',
            'description': 'a list of 3 arxiv search papers',
            'type': 'STRING'
        }],
        'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'
    }
]

cli_demo = Chat(model_dir, 
                system='当开启工具以及代码时，根据需求选择合适的工具进行调用',
                generation_config=generation_config,
                mode='cli'
                )
cli_demo.run(functions=functions)
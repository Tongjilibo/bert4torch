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
model_dir = '/data/pretrain_ckpt/internlm/internlm2_5-7b-chat'

generation_config = {
    'topp': 0.8, 
    'temperature': 0.1,
    'topk': None,
}

functions = [
    {
        "name": "interpreter", 
        "description": "你现在已经能够在一个有状态的 Jupyter 笔记本环境中运行 Python 代码。当你向 python 发送含有 Python 代码的消息时，它将在该环境中执行。这个工具适用于多种场景，如数据分析或处理（包括数据操作、统计分析、图表绘制），复杂的计算问题（解决数学和物理难题），编程示例（理解编程概念或特性），文本处理和分析（比如文本解析和自然语言处理），机器学习和数据科学（用于展示模型训练和数据可视化），以及文件操作和数据导入（处理CSV、JSON等格式的文件）。",
        "parameters":
            {
                "type": "object", 
                "properties": {"symbol":
                                {
                                    "description": "需要追踪的股票代码"
                                }
                            },
                "required": []
            }
    }
]


cli_demo = Chat(model_dir, 
                system='当开启工具以及代码时，根据需求选择合适的工具进行调用',
                generation_config=generation_config,
                mode='cli'
                )
cli_demo.run(functions=functions)
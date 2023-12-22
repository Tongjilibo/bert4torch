'''
该模块的主要功能有两个
1. 很多chat大模型有build_prompt操作，有的操作较为复杂，这里预制以减轻代码重复
2. 提供CliDemo, WebDemo, OpenApiDemo用于快速搭建demo
'''

from bert4torch.chat.api import *
from bert4torch.chat.base import *
from bert4torch.chat.llm import *
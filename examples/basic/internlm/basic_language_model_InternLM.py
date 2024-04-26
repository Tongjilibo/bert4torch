#! -*- coding: utf-8 -*-
"""书生浦语InternLM的测试
Github: https://github.com/InternLM/InternLM
bert4torch_config.json见readme

[1] internlm-chat-7b模型：https://huggingface.co/internlm/internlm-chat-7b
"""


from bert4torch.pipelines import ChatInternLMCli
from transformers import AutoTokenizer

dir_path = 'E:/pretrain_ckpt/internlm/internlm-chat-7b'
tokenizer = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=True)

generation_config = {
    'max_length': 1024, 
    'topp': 0.8, 
    'temperature': 0.8
}

cli_demo = ChatInternLMCli(dir_path, generation_config=generation_config)
cli_demo.run(stream=True)
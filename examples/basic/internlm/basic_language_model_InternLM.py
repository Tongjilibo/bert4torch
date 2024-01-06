#! -*- coding: utf-8 -*-
"""书生浦语InternLM的测试
Github: https://github.com/InternLM/InternLM
bert4torch_config.json见readme
"""


from bert4torch.pipelines import ChatInternLMCli
from transformers import AutoTokenizer

dir_path = 'E:/pretrain_ckpt/internlm/internlm-chat-7b'
tokenizer = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=True)

generation_config = {
    'tokenizer_config': {'skip_special_tokens': True},
    'end_id': [tokenizer.eos_token_id, tokenizer.encode('<eoa>')[-1]], 
    'mode': 'random_sample', 
    'max_length': 1024, 
    'default_rtype': 'logits',
    'use_states': True,
    'topp': 0.8, 
    'temperature': 0.8
}

cli_demo = ChatInternLMCli(dir_path, generation_config=generation_config)
cli_demo.run(stream=True)
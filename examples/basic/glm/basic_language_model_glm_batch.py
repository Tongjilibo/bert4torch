#! -*- coding: utf-8 -*-
# 基本测试：chatglm的batch生成测试

import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer
from bert4torch.generation import SeqGeneration
from bert4torch.snippets import TimeitLogger
import time
import os

# chatglm-6b, chatglm-6b-int4, chatglm-6b-int8
model_dir = "E:/data/pretrain_ckpt/THUDM/chatglm-6b"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
texts = ['你好', '你是谁', '你有哪些功能可以介绍一下吗']


tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
encoder = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir).to(device)
generation = SeqGeneration(encoder, tokenizer, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, 
                           mode='random_sample', max_length=2048, default_rtype='logits', use_states=True)


print('===============single================')
ti = TimeitLogger()
for text in texts:
    response = generation.generate(text, top_k=50, top_p=0.7, temperature=0.95)
    print(response)
ti('single')


print('===============batch_cache================')
response = generation.generate(texts, top_k=50, top_p=0.7, temperature=0.95)
print(response)
ti('batch_cache')


print('===============batch_nocache================')
generation = SeqGeneration(encoder, tokenizer, bos_token_id=None, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, 
                           mode='random_sample', max_length=2048, default_rtype='logits', use_states=False)
ti.restart()
response = generation.generate(texts, top_k=50, top_p=0.7, temperature=0.95)
print(response)
ti('batch_nocache')
ti.end()
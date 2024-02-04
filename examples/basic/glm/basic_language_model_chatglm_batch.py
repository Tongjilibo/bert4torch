#! -*- coding: utf-8 -*-
# 基本测试：chatglm的batch生成测试

import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer
from bert4torch.generation import SeqGeneration
from bert4torch.snippets import Timeit2
import time
import os


dir_path = "E:/pretrain_ckpt/glm/chatglm-6B"
# dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-int4"
# dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-int8"


config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('.bin')]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
texts = ['你好', '你是谁', '你有哪些功能可以介绍一下吗']


tokenizer = AutoTokenizer.from_pretrained(dir_path.replace('/', '\\'), trust_remote_code=True)
encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path).to(device)
generation = SeqGeneration(encoder, tokenizer, end_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id, 
                           mode='random_sample', maxlen=2048, default_rtype='logits', use_states=True)


print('===============single================')
ti = Timeit2()
for text in texts:
    response = generation.generate(text, topk=50, topp=0.7, temperature=0.95)
    print(response)
ti('single')


print('===============batch_cache================')
response = generation.generate(texts, topk=50, topp=0.7, temperature=0.95)
print(response)
ti('batch_cache')


print('===============batch_nocache================')
generation = SeqGeneration(encoder, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id, 
                           mode='random_sample', maxlen=2048, default_rtype='logits', use_states=False)
ti.restart()
response = generation.generate(texts, topk=50, topp=0.7, temperature=0.95)
print(response)
ti('batch_nocache')
ti.end()
#! -*- coding: utf-8 -*-
# 基本测试：chatglm的batch生成测试

import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer
from bert4torch.generation import AutoRegressiveDecoder, SeqGeneration
import time


choice = 'int4'  # default, int4, int8
if choice == 'default':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B"
    config_path = dir_path + '/bert4torch_config.json'
    checkpoint_path = [dir_path + f'/pytorch_model-0000{i}-of-00008.bin' for i in range(1,9)]
elif choice == 'int4':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-int4"
    config_path = dir_path + '/bert4torch_config.json'
    checkpoint_path = dir_path + '/pytorch_model.bin'
elif choice == 'int8':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-int8"
    config_path = dir_path + '/bert4torch_config.json'
    checkpoint_path = dir_path + '/pytorch_model.bin'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
texts = ['你好', '你是谁', '你有哪些功能可以介绍一下吗']


tokenizer = AutoTokenizer.from_pretrained(dir_path.replace('/', '\\'), trust_remote_code=True)
# 建立模型，加载权重
if choice == 'default':
    encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='glm').half()
    encoder = encoder.quantize(quantization_method='cpm_kernels', quantization_bit=8).to(device)
else:
    # 在config中已经写入了量化的配置参数
    encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='glm').to(device)

generation = SeqGeneration(encoder, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id, 
                           mode='random_sample', maxlen=2048, default_rtype='logits', use_states=True)


print('===============single================')
start = time.time()
for text in texts:
    response = generation.generate(text, topk=50, topp=0.7, temperature=0.95)
    print(response)
print(f'Consume: {time.time()-start}s')

print('===============batch_cache================')
start = time.time()
response = generation.batch_generate(texts, topk=50, topp=0.7, temperature=0.95)
print(response)
print(f'Consume: {time.time()-start}s')


print('===============batch_nocache================')
start = time.time()
generation = SeqGeneration(encoder, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, pad_id=tokenizer.pad_token_id, 
                           mode='random_sample', maxlen=2048, default_rtype='logits', use_states=False)
response = generation.batch_generate(texts, topk=50, topp=0.7, temperature=0.95)
print(response)
print(f'Consume: {time.time()-start}s')
#! -*- coding: utf-8 -*-
# 测试代码可用性: 结合MLM的Gibbs采样

from tqdm import tqdm
import numpy as np
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
import torch


root_model_path = "F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12"
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/bert_config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'

tokenizer = Tokenizer(vocab_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, with_mlm='softmax'
)  # 建立模型，加载权重

sentences = []
init_sent = u'科学技术是第一生产力。'  # 给定句子或者None
minlen, maxlen = 8, 32
steps = 10000
converged_steps = 1000
vocab_size = tokenizer._vocab_size

if init_sent is None:
    length = np.random.randint(minlen, maxlen + 1)
    tokens = ['[CLS]'] + ['[MASK]'] * length + ['[SEP]']
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
else:
    token_ids, segment_ids = tokenizer.encode(init_sent)
    length = len(token_ids) - 2

device='cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()
with torch.no_grad():
    for _ in tqdm(range(steps), desc='Sampling'):
        # Gibbs采样流程：随机mask掉一个token，然后通过MLM模型重新采样这个token。
        i = np.random.choice(length) + 1
        token_ids[i] = tokenizer._token_mask_id
        token_ids_tensor, segment_ids_tensor = torch.tensor([token_ids], device=device), torch.tensor([segment_ids], device=device)
        _, probas = model([token_ids_tensor, segment_ids_tensor])
        probas = probas[0, i]
        token = np.random.choice(vocab_size, p=probas.cpu().numpy())
        token_ids[i] = token
        sentences.append(tokenizer.decode(token_ids))

print(u'部分随机采样结果: ')
for _ in range(10):
    print(np.random.choice(sentences[converged_steps:]))

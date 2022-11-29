# 转换huggingface上ethanyt-guwenbert-base权重
# 权重链接：https://huggingface.co/ethanyt/guwenbert-base
# 由于key和框架的key没有完全对齐，主要里面用的都是roberta前缀来保存权重和偏置

import torch

state_dict = torch.load('F:/Projects/pretrain_ckpt/bert/[guwen_hf_torch_base]--ethanyt-guwenbert-base/pytorch_model.bin')
state_dict_new = {}
for k, v in state_dict.items():
    state_dict_new[k.replace('roberta', 'bert')] = v
torch.save(state_dict_new, 'F:/Projects/pretrain_ckpt/bert/[guwen_hf_torch_base]--ethanyt-guwenbert-base/bert4torch_pytorch_model.bin')
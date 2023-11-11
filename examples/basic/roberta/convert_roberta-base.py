# 转换huggingface上bert-base-chinese权重
# 权重链接：https://huggingface.co/roberta-base

import torch

dir_path = 'E:/pretrain_ckpt/roberta/huggingface@roberta-base/'
state_dict = torch.load(dir_path + 'pytorch_model.bin')

mapping = {
    'lm_head.bias': 'cls.predictions.bias', 
    'lm_head.dense.weight': 'cls.predictions.transform.dense.weight', 
    'lm_head.dense.bias': 'cls.predictions.transform.dense.bias', 
    'lm_head.layer_norm.weight': 'cls.predictions.transform.LayerNorm.weight', 
    'lm_head.layer_norm.bias': 'cls.predictions.transform.LayerNorm.bias', 
    'lm_head.decoder.weight': 'cls.predictions.decoder.weight',
}

state_dict_new = {}
for k, v in state_dict.items():
    if 'roberta.' in k:
        k = k.replace('roberta.', 'bert.')
        state_dict_new[k] = v
    else:
        state_dict_new[mapping.get(k, k)] = v
state_dict_new['cls.predictions.decoder.bias'] = state_dict_new['cls.predictions.bias']
torch.save(state_dict_new, dir_path + 'bert4torch_pytorch_model.bin')

# config配置，直接使用hf上下载的config.json即可
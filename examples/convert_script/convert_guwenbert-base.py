# 转换huggingface上ethanyt-guwenbert-base权重
# 权重链接：https://huggingface.co/ethanyt/guwenbert-base
# 由于key和框架的key没有完全对齐，主要里面用的都是roberta前缀来保存权重和偏置

import torch
state_dict = torch.load('F:/Projects/pretrain_ckpt/roberta/[guwen_hf_torch_base]--ethanyt-guwenbert-base/pytorch_model.bin')

mapping = {
            'lm_head.dense.weight': 'cls.predictions.transform.dense.weight',
            'lm_head.dense.bias': 'cls.predictions.transform.dense.bias',
            'lm_head.layer_norm.weight': 'cls.predictions.transform.LayerNorm.weight',
            'lm_head.layer_norm.bias': 'cls.predictions.transform.LayerNorm.bias',
            'lm_head.bias': 'cls.predictions.bias',
            'lm_head.decoder.weight': 'cls.predictions.decoder.weight',
            'lm_head.decoder.bias': 'cls.predictions.decoder.bias',
}

state_dict_new = {}
for k, v in state_dict.items():
    if 'roberta' in k:
        state_dict_new[k.replace('roberta', 'bert')] = v
    else:
        state_dict_new[mapping[k]] = v

torch.save(state_dict_new, 'F:/Projects/pretrain_ckpt/roberta/[guwen_hf_torch_base]--ethanyt-guwenbert-base/bert4torch_pytorch_model.bin')
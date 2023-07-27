# 转换哈工大的chinese-roberta-wwm-ext-base权重
# 权重链接：https://huggingface.co/hfl/chinese-roberta-wwm-ext
# 不转换也没关系，只是在with_mlm=True的时候，会报[WARNIMG] cls.predictions.decoder.bias not found in pretrain models

import torch
import json


dir_path = 'E:/pretrain_ckpt/roberta/[hit_torch_base]--chinese-roberta-wwm-ext-base/'
state_dict = torch.load(dir_path + 'pytorch_model.bin')
state_dict['cls.predictions.decoder.bias'] = state_dict['cls.predictions.bias']
torch.save(state_dict, dir_path + 'bert4torch_pytorch_model.bin')

config = \
{
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "type_vocab_size": 2,
  "vocab_size": 21128
}

with open(dir_path+'bert4torch_config.json', 'w') as f:
    f.write(json.dumps(config, indent=4))

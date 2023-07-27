# 转换huggingface上bert-base-chinese权重
# 权重链接：https://huggingface.co/bert-base-chinese
# 由于key和框架的key没有完全对齐，主要里面用的都是Laynorm.gamma和Laynorm.beta来保存权重和偏置

# 也可使用transformer自带命令转换tf权重https://github.com/google-research/bert
# 转换命令https://huggingface.co/docs/transformers/converting_tensorflow_models

import torch
import json

dir_path = 'E:/pretrain_ckpt/bert/[google_torch_base]--bert-base-chinese/'
state_dict = torch.load(dir_path + 'pytorch_model.bin')
state_dict_new = {}
for k, v in state_dict.items():
    if 'LayerNorm.gamma' in k:
        k = k.replace('LayerNorm.gamma', 'LayerNorm.weight')
        state_dict_new[k] = v
    elif 'LayerNorm.beta' in k:
        k = k.replace('LayerNorm.beta', 'LayerNorm.bias')
        state_dict_new[k] = v
    else:
        state_dict_new[k] = v
state_dict_new['cls.predictions.decoder.bias'] = state_dict['cls.predictions.bias']
torch.save(state_dict_new, dir_path + 'bert4torch_pytorch_model.bin')

# config配置
config = \
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 21128
}
with open(dir_path+'bert4torch_config.json', 'w') as f:
    f.write(json.dumps(config, indent=4))

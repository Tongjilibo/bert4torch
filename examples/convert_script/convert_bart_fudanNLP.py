#! -*- coding: utf-8 -*-
# 将FUDAN(fastnlp)的预训练bart模型转换为bert4torch可用的权重
# 权重地址：https://github.com/fastnlp/CPT
# hf地址：分v1.0和v2.0版本

import torch
import json

version = 2

if version == 1:
  # v1.0转换脚本，下载链接为https://huggingface.co/fnlp/bart-base-chinese/commits/v1.0
  dir_path = 'E:/pretrain_ckpt/bart/[FudanNLP_torch_base]/'
  state_dict = torch.load(dir_path + 'pytorch_model.bin')
  state_dict_new = {}
  for k, v in state_dict.items():
      # 主要变更就是默认有514个位置，舍弃前两个位置
      if 'embed_positions.weight' in k:
          v = v[2:]
          state_dict_new[k] = v
      else:
          state_dict_new[k] = v
  torch.save(state_dict_new, dir_path + 'bert4torch_pytorch_model.bin')
  config = \
  {
    "attention_probs_dropout_prob": 0.1, 
    "hidden_act": "gelu", 
    "hidden_dropout_prob": 0.1, 
    "hidden_size": 768, 
    "initializer_range": 0.02, 
    "intermediate_size": 3072, 
    "max_position_embeddings": 512, 
    "num_attention_heads": 12, 
    "num_hidden_layers": 6, 
    "type_vocab_size": 0, 
    "vocab_size": 21128
  }
  with open(dir_path+'bert4torch_config.json', 'w') as f:
    f.write(json.dumps(config, indent=4))

elif version == 2:
  # v2.0转换脚本，下载链接为https://huggingface.co/fnlp/bart-base-chinese/tree/v2.0
  dir_path = 'E:/pretrain_ckpt/bart/[FudanNLP_torch_base_v2.0]/'
  state_dict = torch.load(dir_path + 'pytorch_model.bin')
  state_dict_new = {}
  for k, v in state_dict.items():
      # 这两个权重丢弃，因为一个为0，一个和decoder的embedding一样
      if k in {'final_logits_bias', 'lm_head.weight'}:
          continue
      
      k = k.replace('model.', '')
      # 主要变更就是舍弃前两个位置
      if 'embed_positions.weight' in k:
          v = v[2:]
          state_dict_new[k] = v
      else:
          state_dict_new[k] = v
  torch.save(state_dict_new, dir_path + 'bert4torch_pytorch_model.bin')

  config = \
  {
    "attention_probs_dropout_prob": 0.1, 
    "hidden_act": "gelu", 
    "hidden_dropout_prob": 0.1, 
    "hidden_size": 768, 
    "initializer_range": 0.02, 
    "intermediate_size": 3072, 
    "max_position_embeddings": 1024, 
    "num_attention_heads": 12, 
    "num_hidden_layers": 6, 
    "type_vocab_size": 0, 
    "vocab_size": 51271
  }
  with open(dir_path+'bert4torch_config.json', 'w') as f:
    f.write(json.dumps(config, indent=4))

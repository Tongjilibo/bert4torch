#! -*- coding: utf-8 -*-
# 将CDial-GPT的pytorch权重转换为bert4torch可适配的权重，base和large都可转换
# 项目链接(torch版本)：https://github.com/thu-coai/CDial-GPT

import torch

ckpt_dir = 'F:/Projects/pretrain_ckpt/gpt/[thu-coai_torch_base]--CDial-GPT-LCCC-base'
ckpt_file = f'{ckpt_dir}/pytorch_model.bin'
output_ckpt_file = 'F:/Projects/pretrain_ckpt/gpt/[thu-coai_torch_base]--CDial-GPT-LCCC-base/bert4torch_pytorch_model.bin'
num_hidden_layers = 12


def convert():
    torch_weights = torch.load(ckpt_file)
    new_weights = {}
    prefix = 'gpt'

    # CDial-GPT的[CLS]是0、[PAD]是1，不符合一般习惯，所以交换一下
    w = torch_weights['transformer.tokens_embed.weight']
    w = torch.cat([w[1:2], w[:1], w[2:]], axis=0)
    new_weights[f'{prefix}.embeddings.word_embeddings.weight'] = w

    w = torch_weights['transformer.positions_embed.weight']
    new_weights[f'{prefix}.embeddings.position_embeddings.weight'] = w

    qkv = ['query', 'key', 'value']
    for i in range(num_hidden_layers):
        prefix_i = f'{prefix}.encoder.layer.%d.' % i

        # q, k, v
        w = torch_weights['transformer.h.%s.attn.c_attn.weight' % i]
        ws = torch.chunk(w, 3, dim=1)
        for k, w in zip(qkv, ws):
            name = prefix_i + f'attention.self.{k}.weight'
            new_weights[name] = w.T
        b = torch_weights['transformer.h.%s.attn.c_attn.bias' % i]
        bs = torch.chunk(b, 3, dim=0)
        for k, b in zip(qkv, bs):
            name = prefix_i + f'attention.self.{k}.bias'
            new_weights[name] = b

        # hdsz-hdsz的全连接
        w = torch_weights['transformer.h.%s.attn.c_proj.weight' % i]
        name = prefix_i + 'attention.output.dense.weight'
        new_weights[name] = w.T
        b = torch_weights['transformer.h.%s.attn.c_proj.bias' % i]
        name = prefix_i + 'attention.output.dense.bias'
        new_weights[name] = b

        # layernorm1
        w = torch_weights['transformer.h.%s.ln_1.weight' % i]
        name = prefix_i + 'attention.output.LayerNorm.weight'
        new_weights[name] = w
        b = torch_weights['transformer.h.%s.ln_1.bias' % i]
        name = prefix_i + 'attention.output.LayerNorm.bias'
        new_weights[name] = b

        # feed forward 第一层
        w = torch_weights['transformer.h.%s.mlp.c_fc.weight' % i]
        name = prefix_i + 'intermediate.dense.weight'
        new_weights[name] = w.T
        b = torch_weights['transformer.h.%s.mlp.c_fc.bias' % i]
        name = prefix_i + 'intermediate.dense.bias'
        new_weights[name] = b

        # feed forward 第二层
        w = torch_weights['transformer.h.%s.mlp.c_proj.weight' % i]
        name = prefix_i + 'output.dense.weight'
        new_weights[name] = w.T
        b = torch_weights['transformer.h.%s.mlp.c_proj.bias' % i]
        name = prefix_i + 'output.dense.bias'
        new_weights[name] = b

        # layernorm2
        w = torch_weights['transformer.h.%s.ln_2.weight' % i]
        name = prefix_i + 'output.LayerNorm.weight'
        new_weights[name] = w
        b = torch_weights['transformer.h.%s.ln_2.bias' % i]
        name = prefix_i + 'output.LayerNorm.bias'
        new_weights[name] = b

        
    torch.save(new_weights, output_ckpt_file)

if __name__ == '__main__':
    convert()

# config文件
'''
{
  "attention_probs_dropout_prob": 0.1, 
  "directionality": "bidi", 
  "hidden_act": "gelu", 
  "hidden_dropout_prob": 0.1, 
  "hidden_size": 768, 
  "initializer_range": 0.02, 
  "intermediate_size": 3072, 
  "max_position_embeddings": 513, 
  "num_attention_heads": 12, 
  "num_hidden_layers": 12, 
  "vocab_size": 13088,
  "type_vocab_size": 3,
  "shared_segment_embeddings": true
}
'''
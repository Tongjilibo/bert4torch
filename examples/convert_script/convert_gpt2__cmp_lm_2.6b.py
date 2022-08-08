#! -*- coding: utf-8 -*-
# 将清华开源的中文GPT2模型（26亿参数）
# 项目链接(tf版本)：https://github.com/TsinghuaAI/CPM-Generate
# pytorch版权重下载链接：https://huggingface.co/TsinghuaAI/CPM-Generate，经过本脚本转成bert4torch适用的权重

import torch

ckpt_dir = 'F:/Projects/pretrain_ckpt/gpt2/[cpm_gpt2_torch]--cpm_lm_2.6b'
ckpt_file = f'{ckpt_dir}/pytorch_model.bin'
output_ckpt_file = f'{ckpt_dir}/bert4torch_pytorch_model.bin'
num_hidden_layers = 32


def convert():
    torch_weights = torch.load(ckpt_file)
    new_weights = {}
    prefix = 'gpt2'

    w = torch_weights['transformer.wte.weight']
    new_weights[f'{prefix}.embeddings.word_embeddings.weight'] = w

    w = torch_weights['transformer.wpe.weight']
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

        # layernorm_final
        w = torch_weights['transformer.ln_f.weight']
        new_weights[f'{prefix}.LayerNormFinal.weight'] = w
        b = torch_weights['transformer.ln_f.bias']
        new_weights[f'{prefix}.LayerNormFinal.bias'] = b
        
    torch.save(new_weights, output_ckpt_file)

if __name__ == '__main__':
    convert()

# config文件
'''
{
  "vocab_size": 30000,
  "hidden_size": 2560,
  "attention_probs_dropout_prob": 0.1,
  "hidden_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "initializer_range": 0.014142135623731,
  "intermediate_size": 10240,
  "max_position_embeddings": 1024,
  "num_attention_heads": 32,
  "num_hidden_layers": 32
}
'''
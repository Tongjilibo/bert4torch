#! -*- coding: utf-8 -*-
# 阿里云的通义千问: https://github.com/QwenLM/Qwen-7B
# Qwen-7B: https://huggingface.co/Qwen/Qwen-7B
# Qwen-7B-Chat: https://huggingface.co/Qwen/Qwen-7B-Chat

import torch
import json

ckpt_dir = 'E:/pretrain_ckpt/Qwen/Qwen-7B-Chat'

# 单文件版本
# ckpt_file = f'{ckpt_dir}/pytorch_model.bin'
# output_ckpt_file = f'{ckpt_dir}/bert4torch_pytorch_model.bin'

# 多文件版本
ckpt_file = [f'{ckpt_dir}/pytorch_model-0000{i}-of-00008.bin' for i in range(1, 9)]
output_ckpt_file = [f'{ckpt_dir}/bert4torch_pytorch_model-0000{i}-of-00008.bin' for i in range(1, 9)]

num_hidden_layers = 32


def convert_single(ckpt_file, output_ckpt_file):
    torch_weights = torch.load(ckpt_file)
    new_weights = {}
    prefix = 'qwen'

    # 词向量
    w = torch_weights.get('transformer.wte.weight', None)
    new_weights[f'{prefix}.embeddings.word_embeddings.weight'] = w

    # lm_head
    w = torch_weights.get('lm_head.weight', None)
    new_weights[f'{prefix}.lm_head.weight'] = w

    # layernorm_final
    w = torch_weights.get('transformer.ln_f.weight', None)
    new_weights[f'{prefix}.LayerNormFinal.weight'] = w

    qkv = ['query', 'key', 'value']
    for i in range(num_hidden_layers):
        prefix_i = f'{prefix}.encoder.layer.%d.' % i

        # q, k, v
        w = torch_weights.get('transformer.h.%s.attn.c_attn.weight' % i, None)
        if w is not None:
            ws = torch.chunk(w, 3, dim=0)
            for k, w in zip(qkv, ws):
                name = prefix_i + f'attention.self.{k}.weight'
                new_weights[name] = w
        b = torch_weights.get('transformer.h.%s.attn.c_attn.bias' % i, None)
        if b is not None:
            bs = torch.chunk(b, 3, dim=0)
            for k, b in zip(qkv, bs):
                name = prefix_i + f'attention.self.{k}.bias'
                new_weights[name] = b

        # hdsz-hdsz的全连接
        w = torch_weights.get('transformer.h.%s.attn.c_proj.weight' % i, None)
        name = prefix_i + 'attention.output.dense.weight'
        new_weights[name] = w

        # layernorm1
        w = torch_weights.get('transformer.h.%s.ln_1.weight' % i, None)
        name = prefix_i + 'attention.output.LayerNorm.weight'
        new_weights[name] = w

        # feed forward 第一层
        w = torch_weights.get('transformer.h.%s.mlp.w2.weight' % i, None)
        name = prefix_i + 'intermediate.dense.weight'
        new_weights[name] = w

        w = torch_weights.get('transformer.h.%s.mlp.w1.weight' % i, None)
        name = prefix_i + 'intermediate2.dense.weight'
        new_weights[name] = w

        # feed forward 第二层
        w = torch_weights.get('transformer.h.%s.mlp.c_proj.weight' % i, None)
        name = prefix_i + 'output.dense.weight'
        new_weights[name] = w

        # layernorm2
        w = torch_weights.get('transformer.h.%s.ln_2.weight' % i, None)
        name = prefix_i + 'output.LayerNorm.weight'
        new_weights[name] = w
    
    new_weights = {k:v for k, v in new_weights.items() if v is not None}
    torch.save(new_weights, output_ckpt_file)

        
if __name__ == '__main__':
    if isinstance(ckpt_file, str):
        convert_single(ckpt_file, output_ckpt_file)
    else:
        for ckpt_file_i, output_ckpt_file_i in zip(ckpt_file, output_ckpt_file):
            convert_single(ckpt_file_i, output_ckpt_file_i)


    config = \
    {
        "hidden_act": "silu",
        "bias_dropout_fusion": True,
        "bos_token_id": 151643,
        "embd_pdrop": 0.1,
        "eos_token_id": 151643,
        "intermediate_size": 22016,
        "initializer_range": 0.02,
        "kv_channels": 128,
        "layer_norm_eps": 1e-05,
        "model_type": "qwen",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "n_positions": 6144,
        "resid_pdrop": 0.1,
        "rotary_emb_base": 10000,
        "rotary_pct": 1.0,
        "scale_attn_weights": True,
        "seq_length": 2048,
        "tie_word_embeddings": False,
        "use_cache": True,
        "use_flash_attn": True,
        "vocab_size": 151936,
        "use_dynamic_ntk": True,
        "logn_attn_len": 32768,
        "segment_vocab_size": 0,
        "skip_init": True,
        "model": "qwen",
        "rope_rank": "updown",
        "max_position": 2048
    }
    with open(ckpt_dir+'/bert4torch_config.json', 'w') as f:
        f.write(json.dumps(config, indent=4))

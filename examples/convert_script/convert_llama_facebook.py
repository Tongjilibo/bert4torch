#! -*- coding: utf-8 -*-
# llama模型：https://github.com/facebookresearch/llama

import torch

ckpt_dir = 'F:/Projects/pretrain_ckpt/llama/7B'
ckpt_file = f'{ckpt_dir}/consolidated.00.pth'
output_ckpt_file = f'{ckpt_dir}/bert4torch_pytorch_model.bin'
num_hidden_layers = 32


def convert():
    torch_weights = torch.load(ckpt_file)
    new_weights = {}
    prefix = 'llama'

    # token编码
    w = torch_weights['tok_embeddings.weight']
    new_weights[f'{prefix}.embeddings.word_embeddings.weight'] = w

    # layernorm_final 
    new_weights[f'{prefix}.LayerNormFinal.weight'] = torch_weights['norm.weight']
    
    # output 
    new_weights[f'{prefix}.dense.weight'] = torch_weights['output.weight']

    for i in range(num_hidden_layers):
        prefix_i = f'{prefix}.encoder.layer.%d.' % i

        # q, k, v
        new_weights[prefix_i + f'attention.self.query.weight'] = torch_weights['layers.%s.attention.wq.weight' % i]
        new_weights[prefix_i + f'attention.self.key.weight'] = torch_weights['layers.%s.attention.wk.weight' % i]
        new_weights[prefix_i + f'attention.self.value.weight'] = torch_weights['layers.%s.attention.wv.weight' % i]

        # hdsz-hdsz的全连接
        new_weights[prefix_i + 'attention.output.dense.weight'] = torch_weights['layers.%s.attention.wo.weight' % i]

        # layernorm1
        new_weights[prefix_i + 'attention.output.LayerNorm.weight'] = torch_weights['layers.%s.attention_norm.weight' % i]

        # feed forward 第一层
        new_weights[prefix_i + 'intermediate.dense.weight'] = torch_weights['layers.%s.feed_forward.w1.weight' % i]

        # feed forward 第二层
        new_weights[prefix_i + 'output.dense.weight'] = torch_weights['layers.%s.feed_forward.w2.weight' % i]

        # feed forward 第三层(bert结构没有)
        new_weights[prefix_i + 'intermediate2.dense.weight'] = torch_weights['layers.%s.feed_forward.w3.weight' % i]

        # layernorm2
        new_weights[prefix_i + 'output.LayerNorm.weight'] = torch_weights['layers.%s.ffn_norm.weight' % i]

    torch.save(new_weights, output_ckpt_file)

if __name__ == '__main__':
    convert()

# config文件
'''
{
	"hidden_size": 4096,
    "intermediate_size": 11008, 
	"multiple_of": 256,
	"num_attention_heads": 32,
	"num_hidden_layers": 32,
	"norm_eps": 1e-06,
	"hidden_act": "silu",
	"vocab_size": 32000,
	"segment_vocab_size": 0
}
'''
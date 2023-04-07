#! -*- coding: utf-8 -*-
# belle模型：https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M

import torch

ckpt_dir = './llama/pt_llama_belle_7b/'
ckpt_file = ckpt_dir + 'pytorch_model.bin'
output_ckpt_file = ckpt_file + 'bert4torch_pytorch_model.bin'
num_hidden_layers = 32

state_dict = torch.load(ckpt_file)
new_state_dict = {}
prefix = 'llama'

new_state_dict[f'{prefix}.embeddings.word_embeddings.weight'] = state_dict['model.embed_tokens.weight']
new_state_dict[f'{prefix}.LayerNormFinal.weight'] = state_dict['model.norm.weight']
new_state_dict[f'{prefix}.dense.weight'] = state_dict['lm_head.weight']

for i in range(num_hidden_layers):
    prefix_i = f'{prefix}.encoder.layer.%d.' % i

    # k,q,v,o
    new_state_dict[prefix_i + f'attention.self.query.weight'] = state_dict['model.layers.{0}.self_attn.q_proj.weight'.format(i)]
    new_state_dict[prefix_i + f'attention.self.key.weight'] = state_dict['model.layers.{0}.self_attn.k_proj.weight'.format(i)]
    new_state_dict[prefix_i + f'attention.self.value.weight'] = state_dict['model.layers.{0}.self_attn.v_proj.weight'.format(i)]
    new_state_dict[prefix_i + 'attention.output.dense.weight'] = state_dict['model.layers.{0}.self_attn.o_proj.weight'.format(i)]

    # layernorm1
    new_state_dict[prefix_i + 'attention.output.LayerNorm.weight'] = state_dict['model.layers.{0}.input_layernorm.weight'.format(i)]

    # feed forward 第一层
    new_state_dict[prefix_i + 'intermediate.dense.weight'] = state_dict['model.layers.{0}.mlp.gate_proj.weight'.format(i)]

    # feed forward 第二层
    new_state_dict[prefix_i + 'output.dense.weight'] = state_dict['model.layers.{0}.mlp.down_proj.weight'.format(i)]

    # feed forward 第三层(bert结构没有)
    new_state_dict[prefix_i + 'intermediate2.dense.weight'] = state_dict['model.layers.{0}.mlp.up_proj.weight'.format(i)]

    # layernorm2
    new_state_dict[prefix_i + 'output.LayerNorm.weight'] = state_dict['model.layers.{0}.post_attention_layernorm.weight'.format(i)]

torch.save(new_state_dict, output_ckpt_file)

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
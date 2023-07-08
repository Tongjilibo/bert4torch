#! -*- coding: utf-8 -*-

'''
[1] belle-llama模型：https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc
模型说明： https://github.com/LianjiaTech/BELLE/tree/main/models
LLaMA模型不允许发布调优后的完整模型权重，但是可以发布原始的模型的diff。因此需要先用脚本合并llama官方权重bell_llama的模型diff

[2] llama_vicuna模型：https://huggingface.co/AlekseyKorshuk/vicuna-7b
模型说明： https://github.com/lm-sys/FastChat

[3]. Ziya-LLaMA-13B_v1.1: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1
[4]. Ziya-LLaMA-13B_v1: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
[5]. Ziya-LLaMA-13B_pretrain: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1
1）下载llama-13b-hf权重：https://huggingface.co/decapoda-research/llama-13b-hf
2）用项目中脚本https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/utils/apply_delta.py合并权重
    python3 -m apply_delta 
    --base E:/pretrain_ckpt/llama/13B-hf 
    --target E:/pretrain_ckpt/llama/[IDEA-CCNL]--Ziya-LLaMA-13B-v1.1 
    --delta E:/pretrain_ckpt/llama/[IDEA-CCNL]--Ziya-LLaMA-13B-v1.1-delta
3）转换为bert4torch的适配权重

'''
import torch
import os

choice = 'Ziya-LLaMA-13B_v1.1'

if choice == 'belle':
    ckpt_dir = 'E:/pretrain_ckpt/llama/belle-llama-7b-2m/'
    ckpt_file = ckpt_dir + 'pytorch_model.bin'
    output_ckpt_file = ckpt_dir + 'bert4torch_pytorch_model.bin'
    num_hidden_layers = 32
elif choice == 'vicuna':
    ckpt_dir = 'E:/pretrain_ckpt/llama/llama_vicuna-7b/'
    ckpt_file = [
        'pytorch_model-00001-of-00014.bin',
        'pytorch_model-00002-of-00014.bin',
        'pytorch_model-00003-of-00014.bin',
        'pytorch_model-00004-of-00014.bin',
        'pytorch_model-00005-of-00014.bin',
        'pytorch_model-00006-of-00014.bin',
        'pytorch_model-00007-of-00014.bin',
        'pytorch_model-00008-of-00014.bin',
        'pytorch_model-00009-of-00014.bin',
        'pytorch_model-00010-of-00014.bin',
        'pytorch_model-00011-of-00014.bin',
        'pytorch_model-00012-of-00014.bin',
        'pytorch_model-00013-of-00014.bin',
        'pytorch_model-00014-of-00014.bin'
    ]
    num_hidden_layers = 32
elif choice == 'Ziya-LLaMA-13B_v1.1':
    ckpt_dir = 'E:/pretrain_ckpt/llama/[IDEA-CCNL]--Ziya-LLaMA-13B-v1.1/'
    ckpt_file = [i for i in os.listdir(ckpt_dir) if i.endswith('.bin')]
    num_hidden_layers = 40

output_ckpt_file = ckpt_dir + 'bert4torch_pytorch_model.bin'

new_state_dict = {}
prefix = 'llama'
if isinstance(ckpt_file, str):
    state_dict = torch.load(ckpt_file)
elif isinstance(ckpt_file, list):
    state_dict = {}
    for file in ckpt_file:
        ckpt_file = ckpt_dir + file
        for k, v in torch.load(ckpt_file).items():
            state_dict[k] = v

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

# =================config文件=====================
# llama_belle
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
	"segment_vocab_size": 0,
    "skip_init": true,
	"rope_rank": "updown"
}
'''

# vicuna
'''
{
"hidden_size": 4096,
"intermediate_size": 11008, 
"multiple_of": 256,
"num_attention_heads": 32,
"num_hidden_layers": 32,
"norm_eps": 1e-06,
"hidden_act": "silu",
"vocab_size": 32001,
"segment_vocab_size": 0,
"skip_init": true,
"rope_rank": "updown"
}
'''
#! -*- coding: utf-8 -*-

'''本脚本支持多个internlm系列模型转换

[1] internlm-chat-7b模型：https://huggingface.co/internlm/internlm-chat-7b

'''
import torch
import json
from tqdm import tqdm

choice = 'internlm-chat-7b'

if choice == 'internlm-chat-7b':
    ckpt_dir = 'E:/pretrain_ckpt/internlm/internlm-chat-7b/'
    ckpt_file = [f'{ckpt_dir}/pytorch_model-0000{i}-of-00008.bin' for i in range(1, 9)]
    output_ckpt_file = [f'{ckpt_dir}/bert4torch_pytorch_model-0000{i}-of-00008.bin' for i in range(1, 9)]
    num_hidden_layers = 32
else:
    raise ValueError(f'{choice} not in pre maintained choices')

def convert_single(ckpt_file, output_ckpt_file):
    new_state_dict = {}
    prefix = 'internlm'
    state_dict = torch.load(ckpt_file)

    new_state_dict[f'{prefix}.embeddings.word_embeddings.weight'] = state_dict.get('model.embed_tokens.weight')
    new_state_dict[f'{prefix}.LayerNormFinal.weight'] = state_dict.get('model.norm.weight')
    new_state_dict[f'{prefix}.lm_head.weight'] = state_dict.get('lm_head.weight')

    for i in range(num_hidden_layers):
        prefix_i = f'{prefix}.encoder.layer.%d.' % i

        # k,q,v,o
        new_state_dict[prefix_i + f'attention.self.query.weight'] = state_dict.get('model.layers.{0}.self_attn.q_proj.weight'.format(i))
        new_state_dict[prefix_i + f'attention.self.key.weight'] = state_dict.get('model.layers.{0}.self_attn.k_proj.weight'.format(i))
        new_state_dict[prefix_i + f'attention.self.value.weight'] = state_dict.get('model.layers.{0}.self_attn.v_proj.weight'.format(i))
        new_state_dict[prefix_i + 'attention.output.dense.weight'] = state_dict.get('model.layers.{0}.self_attn.o_proj.weight'.format(i))

        # k,q,v,o bias
        new_state_dict[prefix_i + f'attention.self.query.bias'] = state_dict.get('model.layers.{0}.self_attn.q_proj.bias'.format(i))
        new_state_dict[prefix_i + f'attention.self.key.bias'] = state_dict.get('model.layers.{0}.self_attn.k_proj.bias'.format(i))
        new_state_dict[prefix_i + f'attention.self.value.bias'] = state_dict.get('model.layers.{0}.self_attn.v_proj.bias'.format(i))
        new_state_dict[prefix_i + 'attention.output.dense.bias'] = state_dict.get('model.layers.{0}.self_attn.o_proj.bias'.format(i))

        # attnLayerNorm
        new_state_dict[prefix_i + 'attention.output.LayerNorm.weight'] = state_dict.get('model.layers.{0}.input_layernorm.weight'.format(i))

        # feed forward 第一层
        new_state_dict[prefix_i + 'intermediate.dense.weight'] = state_dict.get('model.layers.{0}.mlp.gate_proj.weight'.format(i))

        # feed forward 第二层
        new_state_dict[prefix_i + 'output.dense.weight'] = state_dict.get('model.layers.{0}.mlp.down_proj.weight'.format(i))

        # feed forward 第三层(bert结构没有)
        new_state_dict[prefix_i + 'intermediate2.dense.weight'] = state_dict.get('model.layers.{0}.mlp.up_proj.weight'.format(i))

        # ffnLayerNorm
        new_state_dict[prefix_i + 'output.LayerNorm.weight'] = state_dict.get('model.layers.{0}.post_attention_layernorm.weight'.format(i))

    new_state_dict = {k:v for k, v in new_state_dict.items() if v is not None}
    torch.save(new_state_dict, output_ckpt_file)

if __name__ == '__main__':
    for ckpt_file_i, output_ckpt_file_i in tqdm(zip(ckpt_file, output_ckpt_file)):
        convert_single(ckpt_file_i, output_ckpt_file_i)

    # =================config文件=====================
    if choice == 'internlm-chat-7b':
        config = \
        {
            "model": "internlm",
            "hidden_size": 4096,
            "intermediate_size": 11008, 
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "layer_norm_eps": 1e-06,
            "hidden_act": "silu",
            "vocab_size": 103168,
            "pad_token_id": 0,
            "segment_vocab_size": 0,
            "skip_init": True,
            "rope_rank": "updown",
            "torch_dtype": "float16",
            "tie_word_embeddings": False
        }

    with open(ckpt_dir+'/bert4torch_config.json', 'w') as f:
        f.write(json.dumps(config, indent=4))

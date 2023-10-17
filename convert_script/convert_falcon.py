'''falcon的转换脚本
falcon-rw-1b:   https://huggingface.co/tiiuae/falcon-rw-1b
falcon-7b:   https://huggingface.co/tiiuae/falcon-7b  测试中
falcon-7b-instruct:   https://huggingface.co/tiiuae/falcon-7b-instruct  测试中
'''
import torch
import json
import os

choice = 'falcon-7b-instruct'
if choice == 'falcon-rw-1b':
    ckpt_dir = 'E:/pretrain_ckpt/falcon/falcon-rw-1b/'
    num_hidden_layers = 24
elif choice == 'falcon-7b':
    ckpt_dir = 'E:/pretrain_ckpt/falcon/falcon-7b/'
    num_hidden_layers = 32
elif choice == 'falcon-7b-instruct':
    ckpt_dir = 'E:/pretrain_ckpt/falcon/falcon-7b-instruct/'
    num_hidden_layers = 32
else:
    raise ValueError(f'{choice} not in pre maintained choices')

ckpt_files = [ckpt_dir + i for i in os.listdir(ckpt_dir) if i.startswith('pytorch') and i.endswith('.bin')]
output_ckpt_files = [ckpt_dir + 'bert4torch_' + i for i in os.listdir(ckpt_dir) if i.startswith('pytorch') and i.endswith('.bin')]
new_state_dict = {}
prefix = 'falcon'

for ckpt_file, output_ckpt_file in zip(ckpt_files, output_ckpt_files):
    state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))
    new_state_dict[f'{prefix}.embeddings.word_embeddings.weight'] = state_dict.get('transformer.word_embeddings.weight')
    new_state_dict[f'{prefix}.LayerNormFinal.weight'] = state_dict.get('transformer.ln_f.weight')
    new_state_dict[f'{prefix}.LayerNormFinal.bias'] = state_dict.get('transformer.ln_f.bias')
    new_state_dict[f'{prefix}.lm_head.weight'] = state_dict.get('lm_head.weight')

    for i in range(num_hidden_layers):
        prefix_i = f'{prefix}.encoder.layer.%d.' % i

        # k,q,v,o
        qkv = state_dict.get(f'transformer.h.{i}.self_attention.query_key_value.weight')
        if qkv is not None:
            hidden_size = int(qkv.size(0) / 3)
            tensor_list = torch.split(qkv, 64, 0)
            q, k, v = tensor_list[0::3], tensor_list[1::3], tensor_list[2::3]
            q, k, v = torch.cat(q), torch.cat(k), torch.cat(v)
            new_state_dict[prefix_i + f'attention.self.query.weight'] = q
            new_state_dict[prefix_i + f'attention.self.key.weight'] = k
            new_state_dict[prefix_i + f'attention.self.value.weight'] = v

        qkv = state_dict.get(f'transformer.h.{i}.self_attention.query_key_value.bias')
        if qkv is not None:
            tensor_list = torch.split(qkv, 64, 0)
            q, k, v = tensor_list[0::3], tensor_list[1::3], tensor_list[2::3]
            q, k, v = torch.cat(q), torch.cat(k), torch.cat(v)
            new_state_dict[prefix_i + f'attention.self.query.bias'] = q
            new_state_dict[prefix_i + f'attention.self.key.bias'] = k
            new_state_dict[prefix_i + f'attention.self.value.bias'] = v

        new_state_dict[prefix_i + 'attention.output.dense.weight'] = state_dict.get(f'transformer.h.{i}.self_attention.dense.weight')
        new_state_dict[prefix_i + 'attention.output.dense.bias'] = state_dict.get(f'transformer.h.{i}.self_attention.dense.bias')


        # layernorm1
        new_state_dict[prefix_i + 'attention.output.LayerNorm.weight'] = state_dict.get(f'transformer.h.{i}.input_layernorm.weight')
        new_state_dict[prefix_i + 'attention.output.LayerNorm.bias'] = state_dict.get(f'transformer.h.{i}.input_layernorm.bias')

        # feed forward 第一层
        new_state_dict[prefix_i + 'intermediate.dense.weight'] = state_dict.get(f'transformer.h.{i}.mlp.dense_h_to_4h.weight')
        new_state_dict[prefix_i + 'intermediate.dense.bias'] = state_dict.get(f'transformer.h.{i}.mlp.dense_h_to_4h.bias')

        # feed forward 第二层
        new_state_dict[prefix_i + 'output.dense.weight'] = state_dict.get(f'transformer.h.{i}.mlp.dense_4h_to_h.weight')
        new_state_dict[prefix_i + 'output.dense.bias'] = state_dict.get(f'transformer.h.{i}.mlp.dense_4h_to_h.bias')

        # layernorm2
        new_state_dict[prefix_i + 'output.LayerNorm.weight'] = state_dict.get(f'transformer.h.{i}.post_attention_layernorm.weight'.format(i))
        new_state_dict[prefix_i + 'output.LayerNorm.bias'] = state_dict.get(f'transformer.h.{i}.post_attention_layernorm.bias'.format(i))

    new_state_dict = {k:v for k, v in new_state_dict.items() if v is not None}
    torch.save(new_state_dict, output_ckpt_file)


if choice == 'falcon-rw-1b':
    # falcon-rw-1b
    config = \
    {
    "model": "falcon",
    "type_vocab_size": 0,
    "p_bias": "alibi",
    "apply_residual_post_layernorm": False,
    "attention_dropout": 0.0,
    "bias": True,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_dropout": 0.0,
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 8192,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-05,
    "multi_query": False,
    "num_attention_heads": 32,
    "num_hidden_layers": 24,
    "parallel_attn": False,
    "torch_dtype": "bfloat16",
    "vocab_size": 50304,
    "skip_init": True
    }
else:
    config = \
{
	"model": "falcon",
	"type_vocab_size": 0,
    "p_bias": "rotary",
    "apply_residual_post_layernorm": False,
    "attention_dropout": 0.0,
    "bias": False,
    "bos_token_id": 11,
    "eos_token_id": 11,
    "hidden_dropout": 0.0,
    "hidden_size": 4544,
    "initializer_range": 0.02,
	"intermediate_size": 8192,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-05,
    "model_type": "falcon",
    "multi_query": True,
    "num_attention_heads": 71,
    "num_hidden_layers": 32,
    "parallel_attn": True,
    "torch_dtype": "bfloat16",
    "parallel_attn": true,
    "vocab_size": 65024,
	"skip_init": True
}

with open(ckpt_dir+'bert4torch_config.json', 'w') as f:
    f.write(json.dumps(config, indent=4))

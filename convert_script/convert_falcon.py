'''chatglm-6b的转换脚本
falcon-rw-1b:   https://huggingface.co/tiiuae/falcon-rw-1b
'''
import torch
import json

ckpt_dir = '/Users/lb/Documents/pretrain_ckpt/falcon/falcon-rw-1b/'
ckpt_file = ckpt_dir + 'pytorch_model.bin'
output_ckpt_file = ckpt_dir + 'bert4torch_pytorch_model.bin'
num_hidden_layers = 24
new_state_dict = {}
prefix = 'falcon'

state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))
new_state_dict[f'{prefix}.embeddings.word_embeddings.weight'] = state_dict['transformer.word_embeddings.weight']
new_state_dict[f'{prefix}.LayerNormFinal.weight'] = state_dict['transformer.ln_f.weight']
new_state_dict[f'{prefix}.LayerNormFinal.bias'] = state_dict['transformer.ln_f.bias']
new_state_dict[f'{prefix}.lm_head.weight'] = state_dict['lm_head.weight']

for i in range(num_hidden_layers):
    prefix_i = f'{prefix}.encoder.layer.%d.' % i

    # k,q,v,o
    qkv = state_dict[f'transformer.h.{i}.self_attention.query_key_value.weight']
    hidden_size = int(qkv.size(0) / 3)
    tensor_list = torch.split(qkv, 64, 0)
    q, k, v = tensor_list[0::3], tensor_list[1::3], tensor_list[2::3]
    q, k, v = torch.cat(q), torch.cat(k), torch.cat(v)
    new_state_dict[prefix_i + f'attention.self.query.weight'] = q
    new_state_dict[prefix_i + f'attention.self.key.weight'] = k
    new_state_dict[prefix_i + f'attention.self.value.weight'] = v

    qkv = state_dict[f'transformer.h.{i}.self_attention.query_key_value.bias']
    tensor_list = torch.split(qkv, 64, 0)
    q, k, v = tensor_list[0::3], tensor_list[1::3], tensor_list[2::3]
    q, k, v = torch.cat(q), torch.cat(k), torch.cat(v)
    new_state_dict[prefix_i + f'attention.self.query.bias'] = q
    new_state_dict[prefix_i + f'attention.self.key.bias'] = k
    new_state_dict[prefix_i + f'attention.self.value.bias'] = v

    new_state_dict[prefix_i + 'attention.output.dense.weight'] = state_dict[f'transformer.h.{i}.self_attention.dense.weight']
    new_state_dict[prefix_i + 'attention.output.dense.bias'] = state_dict[f'transformer.h.{i}.self_attention.dense.bias']


    # layernorm1
    new_state_dict[prefix_i + 'attention.output.LayerNorm.weight'] = state_dict[f'transformer.h.{i}.input_layernorm.weight']
    new_state_dict[prefix_i + 'attention.output.LayerNorm.bias'] = state_dict[f'transformer.h.{i}.input_layernorm.bias']

    # feed forward 第一层
    new_state_dict[prefix_i + 'intermediate.dense.weight'] = state_dict[f'transformer.h.{i}.mlp.dense_h_to_4h.weight']
    new_state_dict[prefix_i + 'intermediate.dense.bias'] = state_dict[f'transformer.h.{i}.mlp.dense_h_to_4h.bias']

    # feed forward 第二层
    new_state_dict[prefix_i + 'output.dense.weight'] = state_dict[f'transformer.h.{i}.mlp.dense_4h_to_h.weight']
    new_state_dict[prefix_i + 'output.dense.bias'] = state_dict[f'transformer.h.{i}.mlp.dense_4h_to_h.bias']

    # layernorm2
    new_state_dict[prefix_i + 'output.LayerNorm.weight'] = state_dict[f'transformer.h.{i}.post_attention_layernorm.weight'.format(i)]
    new_state_dict[prefix_i + 'output.LayerNorm.bias'] = state_dict[f'transformer.h.{i}.post_attention_layernorm.bias'.format(i)]

torch.save(new_state_dict, output_ckpt_file)


# falcon-rw-1b
config = \
{
  "model": "falcon",
  "type_vocab_size": 0,
  "p_bias": "alibi",
  "apply_residual_connection_post_layernorm": False,
  "attention_dropout": 0.0,
  "bias": True,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_dropout": 0.0,
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "hidden_act": "silu",
  "layer_norm_eps": 1e-05,
  "model_type": "falcon",
  "multi_query": False,
  "num_attention_heads": 32,
  "num_hidden_layers": 24,
  "parallel_attn": False,
  "torch_dtype": "bfloat16",
  "vocab_size": 50304
}

with open(ckpt_dir+'bert4torch_config.json', 'w') as f:
    f.write(json.dumps(config, indent=4))

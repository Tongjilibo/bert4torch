'''chatglm-6b的转换脚本
bloom-560m:  https://huggingface.co/bigscience/bloom-560m
'''
import torch

ckpt_dir = 'E:/pretrain_ckpt/bloom/bloom-560m/'
ckpt_file = ckpt_dir + 'pytorch_model.bin'
output_ckpt_file = ckpt_dir + 'bert4torch_pytorch_model.bin'
num_hidden_layers = 24
new_state_dict = {}
prefix = 'bloom'

state_dict = torch.load(ckpt_file)
new_state_dict[f'{prefix}.embeddings.word_embeddings.weight'] = state_dict['word_embeddings.weight']
new_state_dict[f'{prefix}.embeddings.LayerNorm.weight'] = state_dict['word_embeddings_layernorm.weight']
new_state_dict[f'{prefix}.embeddings.LayerNorm.bias'] = state_dict['word_embeddings_layernorm.bias']

new_state_dict[f'{prefix}.LayerNormFinal.weight'] = state_dict['ln_f.weight']
new_state_dict[f'{prefix}.LayerNormFinal.bias'] = state_dict['ln_f.bias']
new_state_dict[f'{prefix}.lm_head.weight'] = state_dict['word_embeddings.weight']

for i in range(num_hidden_layers):
    prefix_i = f'{prefix}.encoder.layer.%d.' % i

    # k,q,v,o
    qkv = state_dict[f'h.{i}.self_attention.query_key_value.weight']
    hidden_size = int(qkv.size(0) / 3)
    tensor_list = torch.split(qkv, 64, 0)
    q, k, v = tensor_list[0::3], tensor_list[1::3], tensor_list[2::3]
    q, k, v = torch.cat(q), torch.cat(k), torch.cat(v)
    new_state_dict[prefix_i + f'attention.self.query.weight'] = q
    new_state_dict[prefix_i + f'attention.self.key.weight'] = k
    new_state_dict[prefix_i + f'attention.self.value.weight'] = v

    qkv = state_dict[f'h.{i}.self_attention.query_key_value.bias']
    tensor_list = torch.split(qkv, 64, 0)
    q, k, v = tensor_list[0::3], tensor_list[1::3], tensor_list[2::3]
    q, k, v = torch.cat(q), torch.cat(k), torch.cat(v)
    new_state_dict[prefix_i + f'attention.self.query.bias'] = q
    new_state_dict[prefix_i + f'attention.self.key.bias'] = k
    new_state_dict[prefix_i + f'attention.self.value.bias'] = v

    new_state_dict[prefix_i + 'attention.output.dense.weight'] = state_dict[f'h.{i}.self_attention.dense.weight']
    new_state_dict[prefix_i + 'attention.output.dense.bias'] = state_dict[f'h.{i}.self_attention.dense.bias']


    # layernorm1
    new_state_dict[prefix_i + 'attention.output.LayerNorm.weight'] = state_dict[f'h.{i}.input_layernorm.weight']
    new_state_dict[prefix_i + 'attention.output.LayerNorm.bias'] = state_dict[f'h.{i}.input_layernorm.bias']

    # feed forward 第一层
    new_state_dict[prefix_i + 'intermediate.dense.weight'] = state_dict[f'h.{i}.mlp.dense_h_to_4h.weight']
    new_state_dict[prefix_i + 'intermediate.dense.bias'] = state_dict[f'h.{i}.mlp.dense_h_to_4h.bias']

    # feed forward 第二层
    new_state_dict[prefix_i + 'output.dense.weight'] = state_dict[f'h.{i}.mlp.dense_4h_to_h.weight']
    new_state_dict[prefix_i + 'output.dense.bias'] = state_dict[f'h.{i}.mlp.dense_4h_to_h.bias']

    # layernorm2
    new_state_dict[prefix_i + 'output.LayerNorm.weight'] = state_dict[f'h.{i}.post_attention_layernorm.weight'.format(i)]
    new_state_dict[prefix_i + 'output.LayerNorm.bias'] = state_dict[f'h.{i}.post_attention_layernorm.bias'.format(i)]

torch.save(new_state_dict, output_ckpt_file)


# bloom-560m
'''
{
  "apply_residual_connection_post_layernorm": false,
  "attention_dropout": 0.0,
  "attention_softmax_in_fp32": true,
  "bias_dropout_fusion": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 3,
  "unk_token_id": 0,
  "hidden_dropout": 0.0,
  "hidden_act": "gelu",
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-05,
  "hidden_size": 1024,
  "intermediate_size": 4096,
  "num_hidden_layers": 24,
  "num_attention_heads": 16,
  "offset_alibi": 100,
  "pretraining_tp": 1,
  "skip_bias_add": true,
  "skip_bias_add_qkv": false,
  "vocab_size": 250880,
  "segment_vocab_size": 0,
  "pre_layernorm": true,
  "tie_emb_prj_weight": true
}
'''



#! -*- coding: utf-8 -*-

'''本脚本支持多个llama系列模型转换，原格式为pth

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

[6]. baichuan：https://github.com/baichuan-inc/Baichuan-7B
[7]. baichuan：https://github.com/baichuan-inc/Baichuan-13B
[8]. baichuan：https://github.com/baichuan-inc/Baichuan-13B-Chat
    其实baichuan-7b就是llama架构，baichuan-13b是把rope相对编码换成了alibi位置编码

[9]. Llama-2-7B: https://huggingface.co/meta-llama/Llama-2-7b-hf
[10]. Llama-2-13B: https://huggingface.co/meta-llama/Llama-2-13b-hf
[11]. Llama-2-c7B-Chat: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
[12]. Llama-2-13B-Chat: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
'''
import torch
import os
import json

choice = 'llama-2-13b-chat'

if choice == 'belle-llama':
    ckpt_dir = 'E:/pretrain_ckpt/llama/belle-llama-7b-2m/'
    ckpt_file = ckpt_dir + 'pytorch_model.bin'
    num_hidden_layers = 32
elif choice == 'vicuna':
    ckpt_dir = 'E:/pretrain_ckpt/llama/llama_vicuna-7b/'
    ckpt_file = [i for i in os.listdir(ckpt_dir) if i.endswith('.bin') and i.startswith('pytorch')]
    num_hidden_layers = 32
elif choice == 'Ziya-LLaMA-13B_v1.1':
    ckpt_dir = 'E:/pretrain_ckpt/llama/[IDEA-CCNL]--Ziya-LLaMA-13B-v1.1/'
    ckpt_file = [i for i in os.listdir(ckpt_dir) if i.endswith('.bin') and i.startswith('pytorch')]
    num_hidden_layers = 40
elif choice == 'Baichuan-7B':
    ckpt_dir = 'E:/pretrain_ckpt/llama/Baichuan-7B/'
    ckpt_file = ckpt_dir + 'pytorch_model.bin'
    num_hidden_layers = 32
    hidden_size = 4096
elif choice in {'Baichuan-13B', 'Baichuan-13B-Chat'}:
    ckpt_dir = f'E:/pretrain_ckpt/llama/{choice}/'
    ckpt_file = [i for i in os.listdir(ckpt_dir) if i.endswith('.bin') and i.startswith('pytorch')]
    num_hidden_layers = 40
    hidden_size = 5120
elif choice in {'llama-2-7b', 'llama-2-7b-chat'}:
    ckpt_dir = f'E:/pretrain_ckpt/llama-2/{choice}/'
    ckpt_file = [i for i in os.listdir(ckpt_dir) if i.endswith('.bin') and i.startswith('pytorch')]
    num_hidden_layers = 32
elif choice in {'llama-2-13b', 'llama-2-13b-chat'}:
    ckpt_dir = f'E:/pretrain_ckpt/llama-2/{choice}/'
    ckpt_file = [i for i in os.listdir(ckpt_dir) if i.endswith('.bin') and i.startswith('pytorch')]
    num_hidden_layers = 40
else:
    raise ValueError(f'{choice} not in pre maintained choices')

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
new_state_dict[f'{prefix}.lm_head.weight'] = state_dict['lm_head.weight']  # 在v3.0.0之后（不含），这里的dense改为了lm_head, 如果使用v3.0.0（含）之前的，需要改为dense
# new_state_dict[f'{prefix}.dense.weight'] = state_dict['lm_head.weight']

for i in range(num_hidden_layers):
    prefix_i = f'{prefix}.encoder.layer.%d.' % i

    # k,q,v,o
    if 'Baichuan' in choice:
        W_pack = state_dict['model.layers.{0}.self_attn.W_pack.weight'.format(i)]
        tensor_list = torch.split(W_pack, [hidden_size, hidden_size, hidden_size], 0)
        new_state_dict[prefix_i + f'attention.self.query.weight'] = tensor_list[0]
        new_state_dict[prefix_i + f'attention.self.key.weight'] = tensor_list[1]
        new_state_dict[prefix_i + f'attention.self.value.weight'] = tensor_list[2]
    else:
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
if choice == 'belle-llama':
    config = \
    {
        "hidden_size": 4096,
        "intermediate_size": 11008, 
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "layer_norm_eps": 1e-06,
        "hidden_act": "silu",
        "vocab_size": 32000,
        "segment_vocab_size": 0,
        "skip_init": True,
        "rope_rank": "updown"
    }
elif choice == 'vicuna':
    config = \
    {
    "hidden_size": 4096,
    "intermediate_size": 11008, 
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "layer_norm_eps": 1e-06,
    "hidden_act": "silu",
    "vocab_size": 32001,
    "segment_vocab_size": 0,
    "skip_init": True,
    "rope_rank": "updown"
    }
elif choice == 'Ziya-LLaMA-13B_v1.1':
    config = \
    {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 13824,
    "max_position_embeddings": 2048,
    "num_attention_heads": 40,
    "num_hidden_layers": 40,
    "pad_token_id": 0,
    "layer_norm_eps": 1e-06,
    "tie_word_embeddings": False,
    "use_cache": True,
    "vocab_size": 39424,
    "segment_vocab_size": 0,
    "skip_init": True,
    "rope_rank": "updown"
    }
elif choice == 'Baichuan-7B':
    config = \
    {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 4096,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "pad_token_id": 0,
    "layer_norm_eps": 1e-06,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "vocab_size": 64000,
    "segment_vocab_size": 0,
    "rope_rank": "updown",
    "skip_init": True
    }
elif choice in {'Baichuan-13B', 'Baichuan-13B-Chat'}:
    config = \
    {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 13696,
    "model_max_length": 4096,
    "num_attention_heads": 40,
    "num_hidden_layers": 40,
    "pad_token_id": 0,
    "layer_norm_eps": 1e-06,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "vocab_size": 64000,
    "segment_vocab_size": 0,
    "rope_rank": "updown",
    "p_bias": "alibi",
    "skip_init": True
    }
elif choice == 'llama-2-7b':
    config = \
    {
        "hidden_size": 4096,
        "intermediate_size": 11008, 
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "hidden_act": "silu",
        "vocab_size": 32000,
        "segment_vocab_size": 0,
        "skip_init": True,
        "layer_norm_eps": 1e-5,
        "rope_rank": "updown"
    }
elif choice == 'llama-2-7b-chat':
    # layer_norm_eps和llama-2-7b有细微区别
    config = \
    {
        "hidden_size": 4096,
        "intermediate_size": 11008, 
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "hidden_act": "silu",
        "vocab_size": 32000,
        "segment_vocab_size": 0,
        "skip_init": True,
        "layer_norm_eps": 1e-6,
        "rope_rank": "updown"
    }
elif choice in {'llama-2-13b', 'llama-2-13b-chat'}:
    config = \
    {
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
	"hidden_size": 5120,
    "intermediate_size": 13824, 
	"num_attention_heads": 40,
	"num_hidden_layers": 40,
	"initializer_range": 0.02,
	"hidden_act": "silu",
	"vocab_size": 32000,
	"segment_vocab_size": 0,
	"skip_init": True,
	"layer_norm_eps": 1e-5,
	"rope_rank": "updown",
    "tie_word_embeddings": False,
	"torch_dtype": "float16"
    }

with open(ckpt_dir+'/bert4torch_config.json', 'w') as f:
    f.write(json.dumps(config, indent=4))

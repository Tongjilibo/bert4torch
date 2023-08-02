#! -*- coding: utf-8 -*-
"""本脚本支持多个llama系列模型转换，原格式为pth

[1]. llama模型：https://github.com/facebookresearch/llama
    权重下载：[Github](https://github.com/facebookresearch/llama)
    [huggingface](https://huggingface.co/decapoda-research/llama-7b-hf)
    [torrent](https://pan.baidu.com/s/1yBaYZK5LHIbJyCCbtFLW3A?pwd=phhd)


[2]. chinese_llama_plus_7b: https://github.com/ymcui/Chinese-LLaMA-Alpaca
    分为3步，前2步是项目中的，本脚本为第四步
    1）用transformer脚本转换facebook的llama模型；
        python D:/ProgramData/Anaconda3/Lib/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py  
        --input_dir E:/pretrain_ckpt/llama  
        --model_size 7B  
        --output_dir E:/pretrain_ckpt/llama/7B-hf
    2）用项目中脚本合并lora权重；
        python scripts/merge_llama_with_chinese_lora.py 
        --base_model E:/pretrain_ckpt/llama/7B-hf  
        --lora_model E:/pretrain_ckpt/llama/chinese-llama/chinese_llama_plus_lora_7b  
        --output_type pth
        --output_dir E:/pretrain_ckpt/llama/chinese-llama/chinese_llama_plus_7b 
    3）用本脚本转换为bert4torch的适配权重


[3]. chinese_alpaca_plus_7b: https://github.com/ymcui/Chinese-LLaMA-Alpaca
    转换同上，只是合并lora权重需要合并多个lora权重
        python scripts/merge_llama_with_chinese_lora.py 
        --base_model E:/pretrain_ckpt/llama/7B-hf 
        --lora_model E:/pretrain_ckpt/llama/chinese-llama/chinese_llama_plus_lora_7b,E:/pretrain_ckpt/llama/chinese-alpaca/chinese_alpaca_plus_lora_7b  
        --output_type pth 
        --output_dir E:/pretrain_ckpt/llama/chinese-alpaca/chinese_alpaca_plus_7b 

"""

import torch
import json

choice = 'llama'
if choice == 'llame':
    ckpt_dir = 'E:/pretrain_ckpt/llama/7B'
elif choice == 'chinese_llama_plus_7b':
    ckpt_dir = 'E:/pretrain_ckpt/llama/chinese-llama/chinese_llama_plus_7b'
elif choice == 'chinese_alpaca_plus_7b':
    ckpt_dir = 'E:/pretrain_ckpt/llama/chinese-alpaca/chinese_alpaca_plus_7b'
else:
    raise ValueError(f'{choice} not in pre maintained choices')
    
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
    # new_weights[f'{prefix}.dense.weight'] = torch_weights['output.weight']
    new_weights[f'{prefix}.lm_head.weight'] = torch_weights['output.weight']

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

    if choice == 'llame':
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
            "skip_init": True
            }
    elif choice == 'chinese_llama_plus_7b':
        config = \
        {
            "hidden_size": 4096,
            "intermediate_size": 11008, 
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "layer_norm_eps": 1e-06,
            "hidden_act": "silu",
            "vocab_size": 49953,
            "segment_vocab_size": 0,
            "skip_init": True
        }
    elif choice == 'chinese_alpaca_plus_7b':
        config = \
            {
            "hidden_size": 4096,
            "intermediate_size": 11008, 
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "layer_norm_eps": 1e-06,
            "hidden_act": "silu",
            "vocab_size": 49954,
            "segment_vocab_size": 0,
            "skip_init": True
        }

    with open(ckpt_dir+'/bert4torch_config.json', 'w') as f:
        f.write(json.dumps(config, indent=4))

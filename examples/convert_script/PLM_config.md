# 预训练权重config
记录bert4torch需要另外配置的config，部分权重可在转换脚本中查看
----
- xlnet/[hit_torch_base]--chinese-xlnet-base
```json
{
  "architectures": [
    "XLNetLMHeadModel"
  ],
  "attn_type": "bi",
  "bi_data": false,
  "bos_token_id": 1,
  "clamp_len": -1,
  "intermediate_size": 3072,
  "hidden_size": 768,
  "hidden_dropout_prob": 0.1,
  "end_n_top": 5,
  "eos_token_id": 2,
  "hidden_act": "relu",
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-12,
  "mem_len": null,
  "model_type": "xlnet",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "output_past": true,
  "pad_token_id": 5,
  "reuse_len": null,
  "same_length": false,
  "start_n_top": 5,
  "summary_activation": "tanh",
  "summary_last_hidden_dropout_prob": 0.1,
  "summary_type": "last",
  "summary_use_proj": true,
  "untie_r": true,
  "vocab_size": 32000
}
```

- gau/[sushen-torch]--chinese_GAU-alpha-char_L-24_H-768
```json
{
  "hidden_act": "swish",
  "hidden_size": 768,
  "hidden_dropout_prob": 0.1,
  "attention_probs_dropout_prob": 0.1,
  "num_attention_heads": 1,
  "attention_key_size": 128,
  "intermediate_size": 1536,
  "num_hidden_layers": 24,
  "type_vocab_size": 2,
  "vocab_size": 12000
}
```

- gpt/[thu-coai_torch_base]--CDial-GPT-LCCC-base
```json
{
  "attention_probs_dropout_prob": 0.1, 
  "directionality": "bidi", 
  "hidden_act": "gelu", 
  "hidden_dropout_prob": 0.1, 
  "hidden_size": 768, 
  "initializer_range": 0.02, 
  "intermediate_size": 3072, 
  "max_position_embeddings": 513, 
  "num_attention_heads": 12, 
  "num_hidden_layers": 12, 
  "vocab_size": 13088,
  "type_vocab_size": 3,
  "shared_segment_embeddings": true
}
```

- gpt2/[cpm_gpt2_torch]--cpm_lm_2.6b
```json
{
  "vocab_size": 30000,
  "hidden_size": 2560,
  "attention_probs_dropout_prob": 0.1,
  "hidden_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "initializer_range": 0.014142135623731,
  "intermediate_size": 10240,
  "max_position_embeddings": 1024,
  "num_attention_heads": 32,
  "num_hidden_layers": 32
}
```

- gpt2/[gpt2-ml_torch_15g]
```json
{
  "vocab_size": 21130,
  "hidden_size": 1536,
  "attention_probs_dropout_prob": 0.1,
  "hidden_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "initializer_range": 0.014142135623731,
  "intermediate_size": 6144,
  "max_position_embeddings": 1024,
  "num_attention_heads": 24,
  "num_hidden_layers": 48
}
```

- t5/[google_mt5_torch_base]
```json
{
  "attention_probs_dropout_prob": 0.1, 
  "hidden_act": "gelu_new", 
  "hidden_dropout_prob": 0.1, 
  "hidden_size": 768, 
  "initializer_range": 0.02, 
  "intermediate_size": 2048, 
  "max_position_embeddings": 512, 
  "num_attention_heads": 12, 
  "num_hidden_layers": 12, 
  "type_vocab_size": 2, 
  "vocab_size": 250112,
  "relative_attention_num_buckets": 32,
  "attention_scale":  false,
  "is_dropout": true
}
```

- t5/[uer_t5_torch_base]--t5-base-chinese-cluecorpussmall
```json
{
  "attention_probs_dropout_prob": 0.1, 
  "hidden_act": "relu", 
  "hidden_dropout_prob": 0.1, 
  "hidden_size": 768, 
  "initializer_range": 0.02, 
  "intermediate_size": 3072, 
  "max_position_embeddings": 512, 
  "num_attention_heads": 12, 
  "num_hidden_layers": 12, 
  "type_vocab_size": 2, 
  "vocab_size": 21228,
  "relative_attention_num_buckets": 32,
  "attention_scale": false,
  "is_dropout": true
}
```

- t5/[uer_t5_torch_small]--t5-small-chinese-cluecorpussmall
```json
{
  "attention_probs_dropout_prob": 0.1, 
  "hidden_act": "relu", 
  "hidden_dropout_prob": 0.1, 
  "hidden_size": 512, 
  "initializer_range": 0.02, 
  "intermediate_size": 2048, 
  "max_position_embeddings": 512, 
  "num_attention_heads": 8, 
  "num_hidden_layers": 6, 
  "type_vocab_size": 2, 
  "vocab_size": 21228,
  "relative_attention_num_buckets": 32,
  "attention_scale": false,
  "is_dropout": true
}
```

- t5/[sushen_t5_pegasus_torch_small]--chinese_t5_pegasus_small
```json
{
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 512,
  "initializer_range": 0.02,
  "intermediate_size": 1024,
  "num_attention_heads": 6,
  "attention_head_size": 64,
  "num_hidden_layers": 8,
  "vocab_size": 50000,
  "relative_attention_num_buckets": 32,
  "attention_scale":  false,
  "is_dropout": true
}
```

- t5/[sushen_t5_pegasus_torch_base]--chinese_t5_pegasus_base
```json
{
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 2048,
  "num_attention_heads": 12,
  "attention_head_size": 64,
  "num_hidden_layers": 12,
  "vocab_size": 50000,
  "relative_attention_num_buckets": 32,
  "attention_scale":  false,
  "is_dropout": true
}
```

- bart/[FudanNLP_torch_base]
```json
{
  "attention_probs_dropout_prob": 0.1, 
  "hidden_act": "gelu", 
  "hidden_dropout_prob": 0.1, 
  "hidden_size": 768, 
  "initializer_range": 0.02, 
  "intermediate_size": 3072, 
  "max_position_embeddings": 512, 
  "num_attention_heads": 12, 
  "num_hidden_layers": 6, 
  "type_vocab_size": 2, 
  "vocab_size": 21128
}
```
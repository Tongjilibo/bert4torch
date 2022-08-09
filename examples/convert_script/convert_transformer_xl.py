# 权重链接：https://huggingface.co/transfo-xl-wt103
# 该项目是英文的：只用于bert4torch中transformer_xl的调试模型结构，并未实际用于finetune
import torch


ckpt_file = 'F:/Projects/pretrain_ckpt/transformer_xl/[english_hugging_face_torch]--transfo-xl-wt103/pytorch_model.bin'
torch_state_dict = {}
# 1表示transpose, 0表示不变
key_map = {
    'transformer.word_emb.emb_layers.0.weight': 'embeddings.emb_layers.0.weight',
    'transformer.word_emb.emb_layers.1.weight': 'embeddings.emb_layers.1.weight',
    'transformer.word_emb.emb_layers.2.weight': 'embeddings.emb_layers.2.weight',
    'transformer.word_emb.emb_layers.3.weight': 'embeddings.emb_layers.3.weight',
    'transformer.word_emb.emb_projs.0': 'embeddings.emb_projs.0',
    'transformer.word_emb.emb_projs.1': 'embeddings.emb_projs.1',
    'transformer.word_emb.emb_projs.2': 'embeddings.emb_projs.2',
    'transformer.word_emb.emb_projs.3': 'embeddings.emb_projs.3',

    }

for i in range(18):
    key_map.update({
        f'transformer.layers.{i}.dec_attn.r_r_bias': f'encoderLayer.{i}.multiHeadAttention.r_r_bias',
        f'transformer.layers.{i}.dec_attn.r_w_bias': f'encoderLayer.{i}.multiHeadAttention.r_w_bias',
        f'transformer.layers.{i}.dec_attn.o_net.weight': f'encoderLayer.{i}.multiHeadAttention.o.weight',
        f'transformer.layers.{i}.dec_attn.layer_norm.weight': f'encoderLayer.{i}.layerNorm1.weight',
        f'transformer.layers.{i}.dec_attn.layer_norm.bias': f'encoderLayer.{i}.layerNorm1.bias',
        f'transformer.layers.{i}.dec_attn.r_net.weight': f'encoderLayer.{i}.multiHeadAttention.r.weight',
        f'transformer.layers.{i}.pos_ff.CoreNet.0.weight': f'encoderLayer.{i}.feedForward.intermediateDense.weight',
        f'transformer.layers.{i}.pos_ff.CoreNet.0.bias': f'encoderLayer.{i}.feedForward.intermediateDense.bias',
        f'transformer.layers.{i}.pos_ff.CoreNet.3.weight': f'encoderLayer.{i}.feedForward.outputDense.weight',
        f'transformer.layers.{i}.pos_ff.CoreNet.3.bias': f'encoderLayer.{i}.feedForward.outputDense.bias',
        f'transformer.layers.{i}.pos_ff.layer_norm.weight': f'encoderLayer.{i}.layerNorm2.weight',
        f'transformer.layers.{i}.pos_ff.layer_norm.bias': f'encoderLayer.{i}.layerNorm2.bias',
    })


torch_weights = torch.load(ckpt_file)
model_new = {}
for key, value in key_map.items():
    model_new[value] = torch_weights[key]

for i in range(18):
    qkv_net = torch_weights[f'transformer.layers.{i}.dec_attn.qkv_net.weight']
    model_new[f'encoderLayer.{i}.multiHeadAttention.q.weight'], model_new[f'encoderLayer.{i}.multiHeadAttention.k.weight'], model_new[f'encoderLayer.{i}.multiHeadAttention.v.weight'] = qkv_net.chunk(3, dim=0)
torch.save(model_new, 'F:/Projects/pretrain_ckpt/transformer_xl/[english_hugging_face_torch]--transfo-xl-wt103/bert4torch_pytorch_model.bin')

# config文件
'''
{
  "adaptive": true,
  "architectures": [
    "TransfoXLLMHeadModel"
  ],
  "attn_type": 0,
  "clamp_len": 1000,
  "cutoffs": [
    20000,
    40000,
    200000
  ],
  "d_embed": 1024,
  "d_head": 64,
  "intermediate_size": 4096,
  "hidden_size": 1024,
  "div_val": 4,
  "is_dropout": true,
  "adaptive_embedding": true,
  "attention_probs_dropout_prob": 0.0,
  "hidden_dropout_prob": 0.1,
  "hidden_act": "relu", 
  "eos_token_id": 0,
  "ext_len": 0,
  "init": "normal",
  "init_range": 0.01,
  "init_std": 0.02,
  "layer_norm_epsilon": 1e-05,
  "mem_len": 1600,
  "model_type": "transfo-xl",
  "num_attention_heads": 16,
  "num_hidden_layers": 18,
  "pre_lnorm": false,
  "proj_init_std": 0.01,
  "same_length": true,
  "sample_softmax": -1, 
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 250
    }
  },
  "tgt_len": 128,
  "tie_projs": [
    false,
    true,
    true,
    true
  ],
  "tie_weight": true,
  "untie_r": true,
  "vocab_size": 267735
}
'''
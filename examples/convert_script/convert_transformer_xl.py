import numpy as np
import h5py
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
        f'transformer.layers.{i}.dec_attn.r_net.weight': f'encoderLayer.{i}.multiHeadAttention.r_net.weight',
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
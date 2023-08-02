# 用 语言模型+棋谱 的方式监督训练一个下中国象棋模型
# 介绍：https://kexue.fm/archives/7877
# 只是转换苏神已经train好的模型，注意不是预训练模型
import numpy as np
import h5py
import torch
# 这里用的keras==2.3.1
from keras.engine import saving


tf_path = 'E:/Github/bert4keras/examples/best_model_chess.weights'
torch_state_dict = {}
# 1表示transpose, 0表示不变
key_map = {
    'Embedding-Token/embeddings:0': ['embeddings.word_embeddings.weight', 0],
    'Embedding-Segment/embeddings:0': ['embeddings.segment_embeddings.weight', 0],
    'Embedding-Position/embeddings:0': ['embeddings.position_embeddings.weight', 0],
    'Embedding-Norm/gamma:0': ['embeddings.layerNorm.weight', 0],
    'Embedding-Norm/beta:0': ['embeddings.layerNorm.bias', 0],
    'MLM-Dense/kernel:0': ['mlmDense.weight', 1],
    'MLM-Dense/bias:0': ['mlmDense.bias', 0],
    'MLM-Norm/gamma:0': ['mlmLayerNorm.weight', 0],
    'MLM-Norm/beta:0': ['mlmLayerNorm.bias', 0],
    'MLM-Bias/bias:0': ['mlmBias', 0],
    }

for i in range(12):
    key_map.update({
    f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+1}/kernel:0': [f'encoderLayer.{i}.multiHeadAttention.q.weight', 1],
    f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+1}/bias:0': [f'encoderLayer.{i}.multiHeadAttention.q.bias', 0],
    f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+2}/kernel:0': [f'encoderLayer.{i}.multiHeadAttention.k.weight', 1],
    f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+2}/bias:0': [f'encoderLayer.{i}.multiHeadAttention.k.bias', 0],
    f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+3}/kernel:0': [f'encoderLayer.{i}.multiHeadAttention.v.weight', 1],
    f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+3}/bias:0': [f'encoderLayer.{i}.multiHeadAttention.v.bias', 0],
    f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+4}/kernel:0': [f'encoderLayer.{i}.multiHeadAttention.o.weight', 1],
    f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+4}/bias:0': [f'encoderLayer.{i}.multiHeadAttention.o.bias', 0],
    f'Transformer-{i}-MultiHeadSelfAttention-Norm/gamma:0': [f'encoderLayer.{i}.layerNorm1.weight', 0],
    f'Transformer-{i}-MultiHeadSelfAttention-Norm/beta:0': [f'encoderLayer.{i}.layerNorm1.bias', 0],
    f'Transformer-{i}-FeedForward/dense_{i*6+5}/kernel:0': [f'encoderLayer.{i}.feedForward.intermediateDense.weight', 1],
    f'Transformer-{i}-FeedForward/dense_{i*6+5}/bias:0': [f'encoderLayer.{i}.feedForward.intermediateDense.bias', 0],
    f'Transformer-{i}-FeedForward/dense_{i*6+6}/kernel:0': [f'encoderLayer.{i}.feedForward.outputDense.weight', 1],
    f'Transformer-{i}-FeedForward/dense_{i*6+6}/bias:0': [f'encoderLayer.{i}.feedForward.outputDense.bias', 0],
    f'Transformer-{i}-FeedForward-Norm/gamma:0': [f'encoderLayer.{i}.layerNorm2.weight', 0],
    f'Transformer-{i}-FeedForward-Norm/beta:0': [f'encoderLayer.{i}.layerNorm2.bias', 0],
    })

consume_keys = set()
with h5py.File(tf_path, mode='r') as f:
    if 'layer_names' not in f.attrs and 'model_weights' in f:
        f = f['model_weights']

    layer_names = saving.load_attributes_from_hdf5_group(f, 'layer_names')
    
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = saving.load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]

        for i, weight_name in enumerate(weight_names):
            new_key = key_map[weight_name][0]
            if key_map[weight_name][1] == 1:  # transpose
                torch_state_dict[new_key] = torch.from_numpy(weight_values[i]).T
            else:
                torch_state_dict[new_key] = torch.from_numpy(weight_values[i])
            assert new_key not in consume_keys, 'duplicate keys'
            consume_keys.add(new_key)

    if hasattr(f, 'close'):
        f.close()
    elif hasattr(f.file, 'close'):
        f.file.close()


torch_state_dict['mlmDecoder.weight'] = torch_state_dict['embeddings.word_embeddings.weight']
torch_state_dict['mlmDecoder.bias'] = torch_state_dict['mlmBias']

# for k, v in torch_state_dict.items():
#     print(k, v.shape)
torch.save(torch_state_dict, 'E:/Github/bert4torch/examples/others/best_model_chess.pt')

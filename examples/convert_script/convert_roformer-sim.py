# roformer-sim(simbert v2)预训练模型tensorflow转pytorch
# 源项目：https://github.com/ZhuiyiTechnology/roformer-sim
# 跟simbert(v1)最主要的不同是不需要加载位置编码的部分,苏神ckpt也同样无该部分,加载进模型后使用rope位置编码

import torch
import tensorflow as tf
import json

tf_dir = 'F:/Projects/pretrain_ckpt/simbert/[sushen_tf_base]--chinese_roformer-sim-char_L-12_H-768_A-12/'
tf_path = tf_dir + 'bert_model.ckpt'
torch_path = 'F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--chinese_roformer-sim-char_L-12_H-768_A-12/pytorch_model.bin'

with open(tf_dir + 'bert_config.json', 'r') as f:
    config = json.load(f)
    num_layers = config['num_hidden_layers']

torch_state_dict = {}

prefix = 'roformer'
mapping = {
'bert/embeddings/word_embeddings':  f'{prefix}.embeddings.word_embeddings.weight',
'bert/embeddings/token_type_embeddings': f'{prefix}.embeddings.token_type_embeddings.weight',
'bert/embeddings/LayerNorm/beta': f'{prefix}.embeddings.LayerNorm.bias',
'bert/embeddings/LayerNorm/gamma': f'{prefix}.embeddings.LayerNorm.weight',
'cls/predictions/transform/dense/kernel': 'cls.predictions.transform.dense.weight##',
'cls/predictions/transform/dense/bias': 'cls.predictions.transform.dense.bias',
'cls/predictions/transform/LayerNorm/beta': 'cls.predictions.transform.LayerNorm.bias',
'cls/predictions/transform/LayerNorm/gamma': 'cls.predictions.transform.LayerNorm.weight',
'cls/predictions/output_bias': 'cls.predictions.bias',
'bert/pooler/dense/kernel': f'{prefix}.pooler.dense.weight##',
'bert/pooler/dense/bias': f'{prefix}.pooler.dense.bias'}

if ('embedding_size' in config) and (config['embedding_size'] != config['hidden_size']):
    mapping.update({'bert/encoder/embedding_hidden_mapping_in/kernel': f'{prefix}.encoder.embedding_hidden_mapping_in.weight##',
    'bert/encoder/embedding_hidden_mapping_in/bias': f'{prefix}.encoder.embedding_hidden_mapping_in.bias'})

for i in range(num_layers):
    prefix_i = f'{prefix}.encoder.layer.%d.' % i
    mapping.update({
        f'bert/encoder/layer_{i}/attention/self/query/kernel': prefix_i + 'attention.self.query.weight##',  # 转置标识
        f'bert/encoder/layer_{i}/attention/self/query/bias': prefix_i + 'attention.self.query.bias',
        f'bert/encoder/layer_{i}/attention/self/key/kernel': prefix_i + 'attention.self.key.weight##',
        f'bert/encoder/layer_{i}/attention/self/key/bias': prefix_i + 'attention.self.key.bias',
        f'bert/encoder/layer_{i}/attention/self/value/kernel': prefix_i + 'attention.self.value.weight##',
        f'bert/encoder/layer_{i}/attention/self/value/bias': prefix_i + 'attention.self.value.bias',
        f'bert/encoder/layer_{i}/attention/output/dense/kernel': prefix_i + 'attention.output.dense.weight##',
        f'bert/encoder/layer_{i}/attention/output/dense/bias': prefix_i + 'attention.output.dense.bias',
        f'bert/encoder/layer_{i}/attention/output/LayerNorm/beta': prefix_i + 'attention.output.LayerNorm.bias',
        f'bert/encoder/layer_{i}/attention/output/LayerNorm/gamma': prefix_i + 'attention.output.LayerNorm.weight',
        f'bert/encoder/layer_{i}/intermediate/dense/kernel': prefix_i + 'intermediate.dense.weight##',
        f'bert/encoder/layer_{i}/intermediate/dense/bias': prefix_i + 'intermediate.dense.bias',
        f'bert/encoder/layer_{i}/output/dense/kernel': prefix_i + 'output.dense.weight##',
        f'bert/encoder/layer_{i}/output/dense/bias': prefix_i + 'output.dense.bias',
        f'bert/encoder/layer_{i}/output/LayerNorm/beta': prefix_i + 'output.LayerNorm.bias',
        f'bert/encoder/layer_{i}/output/LayerNorm/gamma': prefix_i + 'output.LayerNorm.weight'
    })


for key, value in mapping.items():
    ts = tf.train.load_variable(tf_path, key)
    if value.endswith('##'):
        value = value.replace('##', '')
        torch_state_dict[value] = torch.from_numpy(ts).T
    else:
        torch_state_dict[value] = torch.from_numpy(ts)
torch_state_dict['cls.predictions.decoder.weight'] = torch_state_dict[f'{prefix}.embeddings.word_embeddings.weight']
torch_state_dict['cls.predictions.decoder.bias'] = torch_state_dict['cls.predictions.bias']

torch.save(torch_state_dict, torch_path)

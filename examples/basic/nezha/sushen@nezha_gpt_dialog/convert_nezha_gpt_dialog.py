# NEZHA模型做闲聊任务，苏神已经finetune好的权重，注意不是预训练模型
# 源项目：https://github.com/bojone/nezha_gpt_dialog
# 示例：

import torch
import tensorflow as tf
import json

ckpt_dir = 'E:/pretrain_ckpt/nezha/sushen@nezha_gpt_dialog/'
tf_path = ckpt_dir + 'model.ckpt'
torch_state_dict = {}

prefix = 'bert'
mapping = {
'bert/embeddings/word_embeddings':  f'{prefix}.embeddings.word_embeddings.weight',
'bert/embeddings/token_type_embeddings': f'{prefix}.embeddings.token_type_embeddings.weight',
'bert/embeddings/LayerNorm/beta': f'{prefix}.embeddings.LayerNorm.bias',
'bert/embeddings/LayerNorm/gamma': f'{prefix}.embeddings.LayerNorm.weight',
'cls/predictions/transform/dense/kernel': 'cls.predictions.transform.dense.weight##',
'cls/predictions/transform/dense/bias': 'cls.predictions.transform.dense.bias',
'cls/predictions/transform/LayerNorm/beta': 'cls.predictions.transform.LayerNorm.bias',
'cls/predictions/transform/LayerNorm/gamma': 'cls.predictions.transform.LayerNorm.weight',
'cls/predictions/output_bias': 'cls.predictions.bias'
}

for i in range(12):
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

torch.save(torch_state_dict, ckpt_dir + 'pytorch_model.bin')

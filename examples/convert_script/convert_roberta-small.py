# roberta-small/tiny预训练模型tensorflow转pytorch
# 源项目：https://github.com/ZhuiyiTechnology/pretrained-models
# torch版本： https://huggingface.co/uer UER旗下各个缩微版roberta-small/tiny/mini(注意层规格与苏神的是不一样的)
# 注意苏神版本的roberta-small/tiny的ckpt无pooler层, 区别于bert base转换脚本需要删除pooler层
# 使用的时候需要with_pool=False, 否则会有warnings, CLS的输出直接按last_hidden_state[:, 0]取得

# PS: 暂时不支持苏神的roberta-small/tiny key+版本, 暂时列入todo list

import torch
import tensorflow as tf
import json

# 也可以从huggingface下第三方转换的https://huggingface.co/peterchou/simbert-chinese-base
tf_dir = './tf_chinese_roberta_L-6_H-384-384_A-12/'
tf_path = tf_dir + 'bert_model.ckpt'
torch_path = './pt_chinese_roberta_L-6_H-384-384_A-12/pytorch_model.bin'


with open(tf_dir + 'bert_config.json', 'r') as f:
    config = json.load(f)
    num_layers = config['num_hidden_layers']

torch_state_dict = {}

prefix = 'bert'
mapping = {
'bert/embeddings/word_embeddings':  f'{prefix}.embeddings.word_embeddings.weight',
'bert/embeddings/position_embeddings':  f'{prefix}.embeddings.position_embeddings.weight',
'bert/embeddings/token_type_embeddings': f'{prefix}.embeddings.token_type_embeddings.weight',
'bert/embeddings/LayerNorm/beta': f'{prefix}.embeddings.LayerNorm.bias',
'bert/embeddings/LayerNorm/gamma': f'{prefix}.embeddings.LayerNorm.weight',
'cls/predictions/transform/dense/kernel': 'cls.predictions.transform.dense.weight##',
'cls/predictions/transform/dense/bias': 'cls.predictions.transform.dense.bias',
'cls/predictions/transform/LayerNorm/beta': 'cls.predictions.transform.LayerNorm.bias',
'cls/predictions/transform/LayerNorm/gamma': 'cls.predictions.transform.LayerNorm.weight',
'cls/predictions/output_bias': 'cls.predictions.bias'}

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

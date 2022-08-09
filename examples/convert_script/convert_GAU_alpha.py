# tensorflow权重链接：https://github.com/ZhuiyiTechnology/GAU-alpha
# 这里直接映射到GAU_alpha的结构上了，因此不需要mapping
import torch
import tensorflow as tf

tf_path = 'F:/Projects/pretrain_ckpt/gau/[sushen-tf]--chinese_GAU-alpha-char_L-24_H-768/bert_model.ckpt'
torch_state_dict = {}

ts = tf.train.load_variable(tf_path, 'bert/embeddings/word_embeddings')
torch_state_dict['embeddings.word_embeddings.weight'] = torch.from_numpy(ts)
torch_state_dict['mlmDecoder.weight'] = torch.from_numpy(ts)

ts = tf.train.load_variable(tf_path, 'bert/embeddings/token_type_embeddings')
torch_state_dict['embeddings.segment_embeddings.weight'] = torch.from_numpy(ts)


for i in range(24):
    ts = tf.train.load_variable(tf_path, f'GAU_alpha/encoder/layer_{i}/gau/i_dense/kernel')
    torch_state_dict[f'encoderLayer.{i}.gau.i_dense.weight'] = torch.from_numpy(ts.T)

    ts = tf.train.load_variable(tf_path, f'GAU_alpha/encoder/layer_{i}/gau/o_dense/kernel')
    torch_state_dict[f'encoderLayer.{i}.gau.o_dense.weight'] = torch.from_numpy(ts.T)
    
    ts1 = tf.train.load_variable(tf_path, f'GAU_alpha/encoder/layer_{i}/gau/q_scaleoffset/gamma')
    ts2 = tf.train.load_variable(tf_path, f'GAU_alpha/encoder/layer_{i}/gau/k_scaleoffset/gamma')
    ts = torch.stack([torch.from_numpy(ts1), torch.from_numpy(ts2)], dim=0)
    torch_state_dict[f'encoderLayer.{i}.gau.offsetscale.gamma'] = ts

torch.save(torch_state_dict, 'F:/Projects/pretrain_ckpt/gau/[sushen-torch]--chinese_GAU-alpha-char_L-24_H-768/pytorch_model.bin')


# config文件
'''
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
'''
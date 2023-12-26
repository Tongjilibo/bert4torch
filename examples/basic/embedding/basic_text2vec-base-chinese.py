#! -*- coding: utf-8 -*-
# 预训练模型：https://huggingface.co/shibing624/text2vec-base-chinese
# 方案是Cosent方案

from bert4torch.pipelines import Text2Vec

# 加载模型，请更换成自己的路径
root_model_path = "E:/pretrain_ckpt/embedding/shibing624@text2vec-base-chinese"
sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']

text2vec = Text2Vec(root_model_path)
sentence_embeddings = text2vec.encode(sentences, pool_strategy='mean')
print(sentence_embeddings)

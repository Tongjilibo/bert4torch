#! -*- coding: utf-8 -*-
# 预训练模型：https://huggingface.co/shibing624/text2vec-base-chinese
# 方案是Cosent方案

root_model_path = "E:/data/pretrain_ckpt/embedding/shibing624@text2vec-base-chinese"
sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']

print('=========================================sentence transformer====================================')
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(root_model_path)
sentence_embeddings = m.encode(sentences)
print(sentence_embeddings)

print('=========================================bert4torch====================================')
from bert4torch.pipelines import Text2Vec
text2vec = Text2Vec(root_model_path)
sentence_embeddings = text2vec.encode(sentences)
print(sentence_embeddings)

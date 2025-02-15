root_model_path = 'E:/data/pretrain_ckpt/embedding/moka-ai/m3e-base'
sentences = [
    '* Mixed 此文本嵌入模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索，ALL in one',
    '* Moka 此文本嵌入模型由 MokaAI 训练并开源，训练脚本使用 uniem',
    '* Massive 此文本嵌入模型通过**千万级**的中文句对数据集进行训练'
]

print('=========================================sentence transformer====================================')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(root_model_path)
#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)
#Print the embeddings
print(embeddings)


print('=========================================bert4torch====================================')
from bert4torch.pipelines import Text2Vec
text2vec = Text2Vec(root_model_path)
embeddings = text2vec.encode(sentences)
print(embeddings)
root_model_path = 'E:/pretrain_ckpt/embedding/thenlper@gte-large-zh'
sentences = ['That is a happy person', 'That is a very happy person']

# print('=========================================sentence transformer====================================')
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import cos_sim
# model = SentenceTransformer(root_model_path)
# embeddings = model.encode(sentences)
# print(cos_sim(embeddings[0], embeddings[1]))


print('=========================================bert4torch====================================')
from bert4torch.pipelines import Text2Vec
text2vec = Text2Vec(model_path=root_model_path, device='cuda')
embeddings = text2vec.encode(sentences, normalize_embeddings=True)
similarity = embeddings[0] @ embeddings[1].T
print(similarity)

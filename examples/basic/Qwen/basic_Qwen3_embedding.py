# Qwen3-Embedding-0.6B, Qwen3-Embedding-4B, Qwen3-Embedding-8B
root_model_path = "E:/data/pretrain_ckpt/Qwen/Qwen3-Embedding-0.6B"
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]


print('=========================================bert4torch====================================')
from bert4torch.pipelines import Text2Vec
model1 = Text2Vec(root_model_path)
model1.model.float()  # 模型权重默认是fp16的，转换为fp32，为了和sentence_transformers的结果一致，真实使用时候不必
query_embeddings = model1.encode(queries, prompt_name="query", normalize_embeddings=True)
document_embeddings = model1.encode(documents, normalize_embeddings=True)

scores = (query_embeddings @ document_embeddings.T)
print(scores.tolist())


print('=========================================sentence transformer====================================')
from sentence_transformers import SentenceTransformer
model2 = SentenceTransformer(root_model_path)
query_embeddings = model2.encode(queries, prompt_name="query")
document_embeddings = model2.encode(documents)

# Compute the (cosine) similarity between the query and document embeddings
similarity = model2.similarity(query_embeddings, document_embeddings)
print(similarity)
'''测试bert和transformer的结果比对'''
import torch
import pytest
from sentence_transformers import SentenceTransformer
from bert4torch.pipelines import Text2Vec

# model_path = "E:/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny"
# model_path = "E:/pretrain_ckpt/simbert/sushen@simbert_chinese_small"
# model_path = "E:/pretrain_ckpt/simbert/sushen@simbert_chinese_base"
# model_path = "E:/pretrain_ckpt/simbert/sushen@roformer_chinese_sim_char_base"
# model_path = "E:/pretrain_ckpt/simbert/sushen@roformer_chinese_sim_char_ft_base"

@pytest.mark.parametrize("model_dir", ['E:/pretrain_ckpt/embedding/BAAI@bge-large-en-v1.5',
                                       'E:/pretrain_ckpt/embedding/BAAI@bge-large-zh-v1.5',
                                       'E:/pretrain_ckpt/embedding/thenlper@gte-base-zh',
                                       'E:/pretrain_ckpt/embedding/thenlper@gte-base-zh',
                                       'E:/pretrain_ckpt/embedding/moka-ai@m3e-base',
                                       "E:/pretrain_ckpt/embedding/shibing624@text2vec-base-chinese"
                                        ])
@torch.inference_mode()
def test_embedding(model_dir):
    sentences_1 = ["样例数据-1", "样例数据-2"]
    sentences_2 = ["样例数据-3", "样例数据-4"]

    print('=========================================sentence transformer====================================')
    model = SentenceTransformer(model_dir)
    trans_embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
    trans_embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
    trans_similarity = trans_embeddings_1 @ trans_embeddings_2.T
    print(trans_similarity)


    print('=========================================bert4torch====================================')
    text2vec = Text2Vec(model_path=model_dir)
    b4t_embeddings_1 = text2vec.encode(sentences_1, normalize_embeddings=True)
    b4t_embeddings_2 = text2vec.encode(sentences_2, normalize_embeddings=True)
    b4t_similarity = b4t_embeddings_1 @ b4t_embeddings_2.T
    print(b4t_similarity)

    assert abs(trans_embeddings_1 - b4t_embeddings_1).max() < 1e-4
    assert abs(trans_embeddings_2 - b4t_embeddings_2).max() < 1e-4
    assert abs(trans_similarity - b4t_similarity).max()< 1e-4


if __name__=='__main__':
    test_embedding('E:/pretrain_ckpt/embedding/BAAI@bge-large-en-v1.5')
model_path = "/data/pretrain_ckpt/Tongjilibo/simbert-chinese-tiny"
# model_path = "/data/pretrain_ckpt/Tongjilibo/simbert-chinese-small"
# model_path = "/data/pretrain_ckpt/Tongjilibo/simbert-chinese-base"
# model_path = "/data/pretrain_ckpt/junnyu/roformer_chinese_sim_char_base"
# model_path = "/data/pretrain_ckpt/junnyu/roformer_chinese_sim_char_ft_base"

from bert4torch.pipelines import Text2Vec
sentences = ['我想去首都北京玩玩', '我想去北京玩', '北京有啥好玩的吗？我想去看看', '好渴望去北京游玩啊']
text2vec = Text2Vec(model_path)
embeddings = text2vec.encode(sentences)
print(embeddings)


# [[-0.02802323  0.01794542  0.04988703 ...  0.08008677 -0.00621421
#    0.1870464 ]
#  [ 0.00412573  0.03380035  0.01500554 ...  0.12664588 -0.05869386
#    0.12219049]
#  [-0.09986295  0.01410954  0.0489939  ...  0.0875893  -0.00238749
#    0.05643408]
#  [-0.02329382  0.0388242  -0.1130961  ...  0.00424874 -0.07753429
#    0.17554764]]
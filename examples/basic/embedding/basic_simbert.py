model_path = "E:/data/pretrain_ckpt/Tongjilibo/simbert-chinese-tiny"
# model_path = "E:/data/pretrain_ckpt/Tongjilibo/simbert-chinese-small"
# model_path = "E:/data/pretrain_ckpt/Tongjilibo/simbert-chinese-base"
# model_path = "E:/data/pretrain_ckpt/junnyu/roformer_chinese_sim_char_base"
# model_path = "E:/data/pretrain_ckpt/junnyu/roformer_chinese_sim_char_ft_base"

from bert4torch.pipelines import Text2Vec
sentences = ['我想去首都北京玩玩', '我想去北京玩', '北京有啥好玩的吗？我想去看看', '好渴望去北京游玩啊']
text2vec = Text2Vec(model_path)
embeddings = text2vec.encode(sentences)
print(embeddings)
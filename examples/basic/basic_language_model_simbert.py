#! -*- coding: utf-8 -*-
# SimBERT/RoFormer-Sim测试相似问生成效果，以及句子之间相似度效果
# 官方项目：https://github.com/ZhuiyiTechnology/simbert
# 官方项目：https://github.com/ZhuiyiTechnology/roformer-sim

import torch
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, AutoRegressiveDecoder, get_pool_emb
from bert4torch.tokenizers import Tokenizer, load_vocab

# 基本信息
maxlen = 32
choice = 'simbert_v2'  # simbert simbert_v2
if choice == 'simbert':
    args_model_path = "F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--simbert_chinese_base"
    args_model = 'bert'
else:
    args_model_path = "F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--roformer_chinese_sim_char_base"
    args_model = 'roformer'

# 加载simbert权重或roformer_v2
root_model_path = args_model_path
dict_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


# 建立加载模型
class Model(BaseModel):
    def __init__(self, pool_method='cls'):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool='linear', model=args_model,
                                            application='unilm', keep_tokens=keep_tokens)
        self.pool_method = pool_method

    def forward(self, token_ids, segment_ids):
        hidden_state, pooler, seq_logit = self.bert([token_ids, segment_ids])
        sen_emb = get_pool_emb(hidden_state, pooler, token_ids.gt(0).long(), self.pool_method)
        return seq_logit, sen_emb

model = Model(pool_method='cls').to(device)

class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps('logits')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = torch.cat([token_ids, output_ids], 1)
        segment_ids = torch.cat([segment_ids, torch.ones_like(output_ids, device=device)], 1)
        seq_logit, _ = model.predict([token_ids, segment_ids])
        return seq_logit[:, -1, :]

    def generate(self, text, n=1, topk=5):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.random_sample([token_ids, segment_ids], n, topk)  # 基于随机采样
        return [tokenizer.decode(ids.cpu().numpy()) for ids in output_ids]


synonyms_generator = SynonymsGenerator(start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen, device=device)


def cal_sen_emb(text_list):
    '''输入text的list，计算sentence的embedding
    '''
    X, S = [], []
    for t in text_list:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = torch.tensor(sequence_padding(X), dtype=torch.long, device=device)
    S = torch.tensor(sequence_padding(S), dtype=torch.long, device=device)
    _, Z = model.predict([X, S])
    return Z
    

def gen_synonyms(text, n=100, k=20):
    """"含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    效果：
        >>> gen_synonyms(u'微信和支付宝哪个好？')
        [
            u'微信和支付宝，哪个好?',
            u'微信和支付宝哪个好',
            u'支付宝和微信哪个好',
            u'支付宝和微信哪个好啊',
            u'微信和支付宝那个好用？',
            u'微信和支付宝哪个好用',
            u'支付宝和微信那个更好',
            u'支付宝和微信哪个好用',
            u'微信和支付宝用起来哪个好？',
            u'微信和支付宝选哪个好',
        ]
    """
    r = synonyms_generator.generate(text, n)
    r = [i for i in set(r) if i != text]  # 不和原文相同
    r = [text] + r
    Z = cal_sen_emb(r)
    Z /= (Z**2).sum(dim=1, keepdims=True)**0.5
    argsort = torch.matmul(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]


if __name__ == '__main__':
    choice = 'generate'  # generate  similarity
    
    if choice == 'generate':
        print(gen_synonyms('我想去北京玩玩可以吗', 10, 10))

    elif choice == 'similarity':
        target_text = '我想去首都北京玩玩'
        text_list = ['我想去北京玩', '北京有啥好玩的吗？我想去看看', '好渴望去北京游玩啊']
        Z = cal_sen_emb([target_text]+text_list)
        Z /= (Z**2).sum(dim=1, keepdims=True)**0.5
        similarity = torch.matmul(Z[1:], Z[0])
        for i, line in enumerate(text_list):
            print(f'cos_sim: {similarity[i].item():.4f}, tgt_text: "{target_text}", cal_text: "{line}"')

else:
    model.load_weights('./best_model.pt')

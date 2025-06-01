#! -*- coding: utf-8 -*-
# 利用自带的接口，将SimBERT的同义句生成搭建成Web服务。
# 基于bottlepy简单封装，仅作为临时测试使用，不保证性能。
# 具体用法请看 https://github.com/bojone/bert4keras/blob/8ffb46a16a79f87aa8cdf045df7994036b4be47d/bert4keras/snippets.py#L580

import torch
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, get_pool_emb
from bert4torch.generation import AutoRegressiveDecoder
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.snippets import WebServing

# 基本信息
maxlen = 32
choice = 'simbert'  # simbert simbert_v2
if choice == 'simbert':
    args_model_path = "E:/data/pretrain_ckpt/Tongjilibo/simbert_chinese_base"
    args_model = 'bert'
else:
    args_model_path = "E:/data/pretrain_ckpt/junnyu/roformer_chinese_sim_char_base"
    args_model = 'roformer'

# 加载simbert权重或simbert_v2
root_model_path = args_model_path
dict_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/bert4torch_config.json"
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
        hidden_state, pool_cls, seq_logit = self.bert([token_ids, segment_ids])
        sen_emb = get_pool_emb(hidden_state, pool_cls, token_ids.gt(0).long(), self.pool_method)
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

    def generate(self, text, n=1, top_k=5):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.random_sample([token_ids, segment_ids], n=n, top_k=top_k)  # 基于随机采样
        return [tokenizer.decode(ids.cpu().numpy()) for ids in output_ids]


synonyms_generator = SynonymsGenerator(bos_token_id=None, eos_token_id=tokenizer._token_end_id, max_new_tokens=maxlen, device=device)


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
    """
    r = synonyms_generator.generate(text, n)
    r = [i for i in set(r) if i != text]  # 不和原文相同
    r = [text] + r
    Z = cal_sen_emb(r)
    Z /= (Z**2).sum(dim=1, keepdims=True)**0.5
    argsort = torch.matmul(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]


if __name__ == '__main__':
    arguments = {'text': (None, True), 'n': (int, False), 'k': (int, False)}
    web = WebServing(port=8864)
    web.route('/gen_synonyms', gen_synonyms, arguments)
    web.start()
    # 现在可以测试访问 http://127.0.0.1:8864/gen_synonyms?text=苹果多少钱一斤

#! -*- coding: utf-8 -*-
# 调用transformer_xl模型

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.snippets import AutoRegressiveDecoder

# bert配置
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = None
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载并精简词表，建立分词器
token_dict = load_vocab(
    dict_path=dict_path,
    simplified=False,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='transformer_xl',
    segment_vocab_size=0,
).to(device)

class ArticleCompletion(AutoRegressiveDecoder):
    """基于随机采样的文章续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = torch.cat([inputs[0], output_ids], 1)
        _, mlm_scores = model.predict([token_ids])
        return mlm_scores[:, -1, :]

    def generate(self, text, n=1, topp=0.95):
        token_ids = tokenizer.encode(text)[0][:-1]
        results = self.random_sample([token_ids], n, topp=topp)  # 基于随机采样
        return [text + tokenizer.decode(ids.cpu().numpy()) for ids in results]

article_completion = ArticleCompletion(
    start_id=None,
    end_id=511,  # 511是中文句号
    maxlen=256,
    minlen=128,
    device=device
)


if __name__ == '__main__':
    print(article_completion.generate(u'今天天气不错'))


#! -*- coding: utf-8 -*-
# SimBERT训练代码

import json
from turtle import forward
import numpy as np
from collections import Counter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, ListDataset, text_segmentate, AutoRegressiveDecoder, Callback
from bert4torch.tokenizers import Tokenizer, load_vocab

# 基本信息
maxlen = 32
batch_size = 32

# bert配置
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """读取语料，每行一个json
        示例：{"text": "懂英语的来！", "synonyms": ["懂英语的来！！！", "懂英语的来", "一句英语翻译  懂英语的来"]}
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                D.append(json.loads(l))
        return D

def truncate(text):
    """截断句子
    """
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    return text_segmentate(text, maxlen - 2, seps, strips)[0]

def collate_fn(batch):
    batch_token_ids, batch_segment_ids = [], []
    for d in batch:
        text, synonyms = d['text'], d['synonyms']
        synonyms = [text] + synonyms
        np.random.shuffle(synonyms)
        text, synonym = synonyms[:2]
        text, synonym = truncate(text), truncate(synonym)
        token_ids, segment_ids = tokenizer.encode(text, synonym, maxlen=maxlen * 2)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        token_ids, segment_ids = tokenizer.encode(synonym, text, maxlen=maxlen * 2)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
    
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], [batch_token_ids, batch_segment_ids]

train_dataloader = DataLoader(MyDataset('../datasets/data_similarity.json'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

# 建立加载模型
class Model(BaseModel):
    def __init__(self, pool_method='cls'):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool='linear', 
                                            application='unilm', keep_tokens=keep_tokens)
        self.pool_method = pool_method

    def get_pool_emb(self, hidden_state, pool_cls, attention_mask):
        if self.pool_method == 'cls':
            return pool_cls
        elif self.pool_method == 'mean':
            hidden_state = torch.sum(hidden_state * attention_mask[:, :, None], dim=1)
            attention_mask = torch.sum(attention_mask, dim=1)[:, None]
            return hidden_state / attention_mask
        elif self.pool_method == 'max':
            seq_state = hidden_state * attention_mask[:, :, None]
            return torch.max(seq_state, dim=1)
        else:
            raise ValueError('pool_method illegal')

    def forward(self, token_ids, segment_ids):
        hidden_state, pool_cls, seq_logit = self.bert([token_ids, segment_ids])
        sen_emb = self.get_pool_emb(hidden_state, pool_cls, attention_mask=token_ids.gt(0).long())
        return seq_logit, sen_emb

    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pool_cls = self.bert([token_ids])
            attention_mask = token_ids.gt(0).long()
            output = self.get_pool_emb(hidden_state, pool_cls, attention_mask)
        return output
model = Model(pool_method='cls').to(device)

class TotalLoss(nn.Module):
    """loss分两部分，一是seq2seq的交叉熵，二是相似度的交叉熵。
    """
    def forward(self, outputs, target):
        seq_logit, sen_emb = outputs
        seq_label, seq_mask = target

        seq2seq_loss = self.compute_loss_of_seq2seq(seq_logit, seq_label, seq_mask)
        similarity_loss = self.compute_loss_of_similarity(sen_emb)
        return {'loss': seq2seq_loss + similarity_loss, 'seq2seq_loss': seq2seq_loss, 'similarity_loss': similarity_loss}

    def compute_loss_of_seq2seq(self, y_pred, y_true, y_mask):
        '''
        y_pred: [btz, seq_len, hdsz]
        y_true: [btz, seq_len]
        y_mask: [btz, seq_len]
        '''
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # 指示了要预测的部分
        y_pred = y_pred[:, :-1, :]  # 预测序列，错开一位
        
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = (y_true*y_mask).flatten()
        return F.cross_entropy(y_pred, y_true, ignore_index=0)

    def compute_loss_of_similarity(self, y_pred):
        y_true = self.get_labels_of_similarity(y_pred)  # 构建标签
        y_pred = F.normalize(y_pred, p=2, dim=-1)  # 句向量归一化
        similarities = torch.matmul(y_pred, y_pred.T)  # 相似度矩阵
        similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12  # 排除对角线
        similarities = similarities * 30  # scale

        loss = F.cross_entropy(similarities, y_true)
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = torch.arange(0, y_pred.shape[0], device=device)
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = idxs_1.eq(idxs_2).float()
        return labels

model.compile(loss=TotalLoss(), optimizer=optim.Adam(model.parameters(), 1e-5), metrics=['seq2seq_loss', 'similarity_loss'])


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
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = torch.tensor(sequence_padding(X), dtype=torch.long, device=device)
    S = torch.tensor(sequence_padding(S), dtype=torch.long, device=device)
    _, Z = model.predict([X, S])
    Z /= (Z**2).sum(dim=1, keepdims=True)**0.5
    argsort = torch.matmul(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]


def just_show(some_samples):
    """随机观察一些样本的效果
    """
    S = [np.random.choice(some_samples) for _ in range(3)]
    for s in S:
        try:
            print(u'原句子：%s' % s)
            print(u'同义句子：')
            print(gen_synonyms(s, 10, 10))
            print()
        except:
            pass


class Evaluator(Callback):
    """评估模型
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, global_step, epoch, logs=None):
        model.save_weights('./best_model.pt')
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            # model.save_weights('./best_model.pt')
        # 演示效果
        just_show(['微信和支付宝拿个好用？',
                   '微信和支付宝，哪个好?',
                   '微信和支付宝哪个好',
                   '支付宝和微信哪个好',
                   '支付宝和微信哪个好啊',
                   '微信和支付宝那个好用？',
                   '微信和支付宝哪个好用',
                   '支付宝和微信那个更好',
                   '支付宝和微信哪个好用',
                   '微信和支付宝用起来哪个好？',
                   '微信和支付宝选哪个好'
                   ])


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=50, steps_per_epoch=200, callbacks=[evaluator])
else:
    model.load_weights('./best_model.pt')

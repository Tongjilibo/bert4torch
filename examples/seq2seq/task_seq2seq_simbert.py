#! -*- coding: utf-8 -*-
# SimBERT预训练代码，也可用于微调，微调方式用其他方式比如sentence_bert的可能更好
# 官方项目：https://github.com/ZhuiyiTechnology/simbert

import json
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, ListDataset, text_segmentate, get_pool_emb
from bert4torch.generation import AutoRegressiveDecoder
from bert4torch.callbacks import Callback
from bert4torch.tokenizers import Tokenizer, load_vocab

# 基本信息
maxlen = 32
batch_size = 32

# 这里加载的是simbert权重，在此基础上用自己的数据继续pretrain/finetune
# 自己从头预训练也可以直接加载bert/roberta等checkpoint
config_path = 'E:/data/pretrain_ckpt/Tongjilibo/simbert_chinese_base/bert4torch_config.json'
checkpoint_path = 'E:/data/pretrain_ckpt/Tongjilibo/simbert_chinese_base/pytorch_model.bin'
dict_path = 'E:/data/pretrain_ckpt/Tongjilibo/simbert_chinese_base/vocab.txt'
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
                                            with_mlm='linear', application='unilm', keep_tokens=keep_tokens)
        self.pool_method = pool_method

    def forward(self, token_ids, segment_ids):
        hidden_state, pool_cls, seq_logit = self.bert([token_ids, segment_ids])
        sen_emb = get_pool_emb(hidden_state, pool_cls, token_ids.gt(0).long(), self.pool_method)
        return seq_logit, sen_emb

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


def just_show(some_samples):
    """随机观察一些样本的效果
    """
    S = [np.random.choice(some_samples) for _ in range(3)]
    for s in S:
        try:
            print(u'原句子：%s' % s)
            print(u'同义句子：', gen_synonyms(s, 10, 10))
            print()
        except:
            pass


class Evaluator(Callback):
    """评估模型
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, global_step, epoch, logs=None):
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
    choice = 'similarity'  # train  generate  similarity
    
    if choice == 'train':
        evaluator = Evaluator()
        model.fit(train_dataloader, epochs=50, steps_per_epoch=None, callbacks=[evaluator])

    elif choice == 'generate':
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

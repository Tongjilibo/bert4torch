#! -*- coding: utf-8 -*-
# SimBERT_v2预训练代码stage2，把simbert的相似度蒸馏到roformer-sim上
# 官方项目：https://github.com/ZhuiyiTechnology/roformer-sim

import json
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, ListDataset, text_segmentate, get_pool_emb
from bert4torch.snippets import AutoRegressiveDecoder, Callback, truncate_sequences
from bert4torch.tokenizers import Tokenizer
import jieba
jieba.initialize()

# 基本信息
maxlen = 64
batch_size = 12

# bert配置，需要加载stage1训练后的权重，这里直接加载官方最终的权重以示例
config_path = 'F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--roformer_chinese_sim_char_base/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--roformer_chinese_sim_char_base/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--roformer_chinese_sim_char_base/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 这里语料和stage1保持一致
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

def masked_encode(text):
    """wwm随机mask
    """
    words = jieba.lcut(text)
    rands = np.random.random(len(words))
    source, target = [tokenizer._token_start_id], [0]
    for r, w in zip(rands, words):
        ids = tokenizer.encode(w)[0][1:-1]
        if r < 0.15 * 0.8:
            source.extend([tokenizer._token_mask_id] * len(ids))
            target.extend(ids)
        elif r < 0.15 * 0.9:
            source.extend(ids)
            target.extend(ids)
        elif r < 0.15:
            source.extend(
                np.random.choice(tokenizer._vocab_size - 1, size=len(ids)) + 1
            )
            target.extend(ids)
        else:
            source.extend(ids)
            target.extend([0] * len(ids))
    source = source[:maxlen - 1] + [tokenizer._token_end_id]
    target = target[:maxlen - 1] + [0]
    return source, target

# ========== 蒸馏用：开始 ==========
# simbert配置
sim_config_path = 'F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--simbert_chinese_base/config.json'
sim_checkpoint_path = 'F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--simbert_chinese_base/pytorch_model.bin'
sim_dict_path = 'F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--simbert_chinese_base/vocab.txt'

# 建立分词器
sim_tokenizer = Tokenizer(sim_dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
simbert = build_transformer_model(sim_config_path, sim_checkpoint_path, with_pool='linear', application='unilm').to(device)
# ========== 蒸馏用：结束 ==========


def collate_fn(batch):
    batch_token_ids, batch_segment_ids = [], []
    batch_sim_token_ids, batch_sim_segment_ids = [], []
    for d in batch:
        text, synonyms = d['text'], d['synonyms']
        synonyms = [text] + synonyms
        np.random.shuffle(synonyms)
        for _ in range(2):
            text, synonym = synonyms[:2]
            if np.random.random() < 0.5:
                text_ids = masked_encode(text)[0]
            else:
                text_ids = tokenizer.encode(text)[0]
            synonym_ids = tokenizer.encode(synonym)[0][1:]
            truncate_sequences(maxlen * 2, -2, text_ids, synonym_ids)
            token_ids = text_ids + synonym_ids
            segment_ids = [0] * len(text_ids) + [1] * len(synonym_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)

            # ==== 蒸馏用：开始 ====
            token_ids, segment_ids = sim_tokenizer.encode(text, maxlen=maxlen)
            batch_sim_token_ids.append(token_ids)
            batch_sim_segment_ids.append(segment_ids)
            # ==== 蒸馏用：结束 ====

            text, synonym = synonym, text
    
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    
    # ==== 蒸馏用：开始 ====
    batch_sim_token_ids = torch.tensor(sequence_padding(batch_sim_token_ids), dtype=torch.long, device=device)
    batch_sim_segment_ids = torch.tensor(sequence_padding(batch_sim_segment_ids), dtype=torch.long, device=device)
    sim_vecs = simbert.predict([batch_sim_token_ids, batch_sim_segment_ids])[1]
    sim_vecs /= (sim_vecs**2).sum(dim=-1, keepdims=True)**0.5
    sims = torch.matmul(sim_vecs, sim_vecs.T)
    # ==== 蒸馏用：结束 ====

    return [batch_token_ids, batch_segment_ids], [batch_token_ids, batch_segment_ids, sims]

train_dataloader = DataLoader(MyDataset('../datasets/data_similarity.json'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

# 建立加载模型
class Model(BaseModel):
    def __init__(self, pool_method='cls'):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='roformer', 
                                            with_pool='linear', with_mlm=True, dropout_rate=0.2, application='unilm')
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
        seq_label, seq_mask, sims = target

        seq2seq_loss = self.compute_loss_of_seq2seq(seq_logit, seq_label, seq_mask)
        similarity_loss = self.compute_loss_of_similarity(sen_emb, sims)
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

    def compute_loss_of_similarity(self, y_pred, y_true):
        y_pred = F.normalize(y_pred, p=2, dim=-1)  # 句向量归一化
        similarities = torch.matmul(y_pred, y_pred.T)  # 相似度矩阵
        loss = 100 * torch.mean((similarities - y_true) ** 2)
        return loss


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
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=50, steps_per_epoch=None, callbacks=[evaluator])
else:
    model.load_weights('./best_model.pt')

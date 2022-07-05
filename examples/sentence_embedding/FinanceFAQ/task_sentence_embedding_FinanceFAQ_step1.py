#! -*- coding:utf-8 -*-
# loss: MultiNegativeRankingLoss, 和simcse一样，以batch中其他样本作为负样本

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset, get_pool_emb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
import random
random.seed(2022)

maxlen = 256
batch_size = 8
# config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
# dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
config_path = '/Users/lb/Documents/Project/pretrain_ckpt/bert/[hit_tf_base]chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = None
dict_path = '/Users/lb/Documents/Project/pretrain_ckpt/bert/[hit_tf_base]chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
choice = 'random'  # raw, random

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

if choice == 'raw':
    # 原始模式，可能同一个batch中会出现重复标问
    def collate_fn(batch):
        texts_list = [[] for _ in range(2)]
        for texts in batch:
            for i, text in enumerate(texts):
                token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
                texts_list[i].append(token_ids)

        for i, texts in enumerate(texts_list):
            texts_list[i] = torch.tensor(sequence_padding(texts), dtype=torch.long, device=device)
        labels = torch.arange(texts_list[0].size(0), device=texts_list[0].device)
        return texts_list, labels

    class MyDataset(ListDataset):
        @staticmethod
        def load_data(filename):
            D = []
            with open(filename, encoding='utf-8') as f:
                for row, l in enumerate(f):
                    if row == 0:  # 跳过首行
                        continue
                    q_std, q_sim = l.strip().split('\t')
                    D.append((q_std.replace(' ', ''), q_sim.replace(' ', '')))
            return D

elif choice == 'random':
    # 以标准问为key的键值对, 保证一个batch内不存在同样q_std的样本
    def collate_fn(batch):
        texts_list = [[] for _ in range(2)]
        for text_list in batch:  # q_std有0.5的概率被抽样到
            p = [0.5] + [0.5/(len(text_list)-1)] * (len(text_list)-1)
            texts = np.random.choice(text_list, 2, replace=False, p=p)
            for i, text in enumerate(texts):
                token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
                texts_list[i].append(token_ids)

        for i, texts in enumerate(texts_list):
            texts_list[i] = torch.tensor(sequence_padding(texts), dtype=torch.long, device=device)
        labels = torch.arange(texts_list[0].size(0), device=texts_list[0].device)
        return texts_list, labels

    class MyDataset(ListDataset):
        @staticmethod
        def load_data(filename):
            D = dict()
            with open(filename, encoding='utf-8') as f:
                for row, l in enumerate(f):
                    if row == 0:  # 跳过首行
                        continue
                    q_std, q_sim = l.strip().split('\t')
                    q_std = q_std.replace(' ', '')
                    q_sim = q_sim.replace(' ', '')
                    D[q_std] = D.get(q_std, []) + [q_sim]
            return [[k]+v for k, v in D.items()]

# train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/qa/FinanceFAQ_train.tsv'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
# valid_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/qa/FinanceFAQ_valid.tsv'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
train_dataloader = DataLoader(MyDataset('/Users/lb/Documents/Project/data/qa/FinanceFAQ/FinanceFAQ.tsv'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset('/Users/lb/Documents/Project/data/qa/FinanceFAQ/FinanceFAQ.tsv'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 


# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='cls', scale=20.0):
        super().__init__()
        self.bert, self.config = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True, return_model_config=True, segment_vocab_size=0)
        self.pool_method = pool_method
        self.scale = scale

    def forward(self, token_ids_list):
        reps = []
        for token_ids in token_ids_list:
            hidden_state1, pool_cls1 = self.bert([token_ids])
            rep = get_pool_emb(hidden_state1, pool_cls1, token_ids.gt(0).long(), self.pool_method)
            reps.append(rep)
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])
        scores = self.cos_sim(embeddings_a, embeddings_b) * self.scale  # [btz, btz*2]
        return scores

    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pool_cls = self.bert([token_ids])
            output = get_pool_emb(hidden_state, pool_cls, token_ids.gt(0).long(), self.pool_method)
        return output
    
    @staticmethod
    def cos_sim(a, b):
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))


model = Model().to(device)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),  # 用足够小的学习率
)

# 定义评价函数
def evaluate(data):
    embeddings1, embeddings2, labels = [], [], []
    for (batch_token1_ids, batch_token2_ids), label in data:
        embeddings1.append(model.encode(batch_token1_ids))
        embeddings2.append(model.encode(batch_token2_ids))
        labels.append(label)

    embeddings1 = torch.cat(embeddings1).cpu().numpy()
    embeddings2 = torch.cat(embeddings2).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    eval_pearson_cosine, _ = spearmanr(labels, cosine_scores)
    return eval_pearson_cosine


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_consine = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_consine = evaluate(valid_dataloader)
        if val_consine > self.best_val_consine:
            self.best_val_consine = val_consine
            # model.save_weights('best_model.pt')
        print(f'val_consine: {val_consine:.5f}, best_val_consine: {self.best_val_consine:.5f}\n')


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, 
            epochs=10, 
            steps_per_epoch=None, 
            callbacks=[evaluator]
            )
else:
    model.load_weights('best_model.pt')

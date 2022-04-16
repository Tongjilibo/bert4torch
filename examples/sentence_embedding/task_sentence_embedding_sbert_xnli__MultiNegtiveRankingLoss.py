#! -*- coding:utf-8 -*-
# 语义相似度任务：训练集xnli, dev集为sts-b
# loss: MultiNegativeRankingLoss, 和simcse一样，以batch中其他样本作为负样本

import enum
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import random
random.seed(2022)

maxlen = 256
batch_size = 8
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def collate_fn(batch):
    texts_list = [[] for _ in range(3)]
    for texts in batch:
        for i, text in enumerate(texts):
            token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
            texts_list[i].append(token_ids)

    for i, texts in enumerate(texts_list):
        texts_list[i] = torch.tensor(sequence_padding(texts), dtype=torch.long, device=device)
    labels = torch.arange(texts_list[0].size(0), device=texts_list[0].device)
    return texts_list, labels

# 加载数据集
def get_data(filename):
    train_data = {}
    with open(filename, encoding='utf-8') as f:
        for row, l in enumerate(f):
            if row == 0:  # 跳过首行
                continue
            text1, text2, label = l.strip().split('\t')
            text1 = text1.replace(' ', '')  # 原来是分词好的，这里重新tokenize
            text2 = text2.replace(' ', '')

            if text1 not in train_data:
                train_data[text1] = {'contradictory': set(), 'entailment': set(), 'neutral': set()}
            train_data[text1][label].add(text2)
            if text2 not in train_data:
                train_data[text2] = {'contradictory': set(), 'entailment': set(), 'neutral': set()}
            train_data[text2][label].add(text1)

    train_samples, dev_samples = [], []
    for sent1, others in train_data.items():
        if (len(others['entailment']) == 0) or (len(others['contradictory']) == 0):
            continue
        # sentence bert的逻辑是下面两个都加进去，这样的问题是如果shuffle=False，处于同一个batch中，相似句可能label给的负样本
        if random.random() < 0.5:
            train_samples.append((sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradictory']))))
        else:
            train_samples.append((random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradictory']))))
    return train_samples

train_data = get_data('F:/Projects/data/corpus/sentence_embedding/XNLI-MT-1.0/multinli/multinli.train.zh.tsv')
train_dataloader = DataLoader(ListDataset(data=train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
from task_sentence_embedding_sbert_sts_b__CosineSimilarityLoss import valid_dataloader

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='mean', scale=20.0):
        super().__init__()
        self.bert, self.config = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True, return_model_config=True, segment_vocab_size=0)
        self.pool_method = pool_method
        self.scale = scale

    def forward(self, token_ids_list):
        reps = []
        for token_ids in token_ids_list:
            hidden_state1, pool_cls1 = self.bert([token_ids])
            rep = self.get_pool_emb(hidden_state1, pool_cls1, attention_mask=token_ids.gt(0).long())
            reps.append(rep)
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])
        scores = self.cos_sim(embeddings_a, embeddings_b) * self.scale  # [btz, btz*2]
        return scores

    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pool_cls = self.bert([token_ids])
            output = self.get_pool_emb(hidden_state, pool_cls, attention_mask=token_ids.gt(0).long())
        return output
    
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

    embeddings1 = torch.concat(embeddings1).cpu().numpy()
    embeddings2 = torch.concat(embeddings2).cpu().numpy()
    labels = torch.concat(labels).cpu().numpy()
    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
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
            epochs=20, 
            steps_per_epoch=300, 
            callbacks=[evaluator]
            )
else:
    model.load_weights('best_model.pt')

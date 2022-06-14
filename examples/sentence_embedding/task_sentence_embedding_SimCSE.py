#! -*- coding: utf-8 -*-
# SimCSE 中文测试

import sys
from turtle import forward
from bert4torch.snippets import sequence_padding
from tqdm import tqdm
import numpy as np
import scipy.stats
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.tokenizers import Tokenizer
from bert4torch.snippets import sequence_padding
from torch.utils.data import DataLoader
from torch import optim, nn
import torch
from bert4torch.snippets import ListDataset

import jieba
jieba.initialize()


def load_data(filename):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                D.append((l[0], l[1], float(l[2])))
    return D


def convert_to_ids(data, tokenizer, maxlen=64):
    """转换文本数据为id形式
    """
    a_token_ids, b_token_ids, labels = [], [], []
    for d in tqdm(data):
        token_ids = tokenizer.encode(d[0], maxlen=maxlen)[0]
        a_token_ids.append(token_ids)
        token_ids = tokenizer.encode(d[1], maxlen=maxlen)[0]
        b_token_ids.append(token_ids)
        labels.append(d[2])
    a_token_ids = sequence_padding(a_token_ids)
    b_token_ids = sequence_padding(b_token_ids)
    return a_token_ids, b_token_ids, labels


# =============================基本参数=============================
# model_type, pooling, task_name, dropout_rate = sys.argv[1:]  # 传入参数
model_type, pooling, task_name, dropout_rate = 'BERT', 'first-last-avg', 'ATEC', 0.3  # debug使用
assert model_type in {'BERT', 'RoBERTa', 'NEZHA', 'RoFormer', 'SimBERT'}
assert pooling in {'first-last-avg', 'last-avg', 'cls', 'pooler'}
assert task_name in {'ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B'}
dropout_rate = float(dropout_rate)
batch_size = 32

if task_name == 'PAWSX':
    maxlen = 128
else:
    maxlen = 64

# bert配置
model_dir = {
    'BERT': 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12',
    'RoBERTa': 'F:/Projects/pretrain_ckpt/robert/[hit_torch_base]--chinese-roberta-wwm-ext-base',
    'NEZHA': 'F:/Projects/pretrain_ckpt/nezha/[github_torch_base]--nezha-cn-base',
    'RoFormer': 'F:/Projects/pretrain_ckpt/roformer/[sushen_torch_base]--roformer_v1_base',
    'SimBERT': 'F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--simbert_chinese_base',
}[model_type]

config_path = f'{model_dir}/bert_config.json' if model_type == 'BERT' else f'{model_dir}/config.json'
checkpoint_path = f'{model_dir}/pytorch_model.bin'
dict_path = f'{model_dir}/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 建立分词器
if model_type in ['RoFormer']:
    tokenizer = Tokenizer(dict_path, do_lower_case=True, pre_tokenize=lambda s: jieba.lcut(s, HMM=False))
else:
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

# =============================加载数据集=============================
data_path = 'F:/Projects/data/corpus/sentence_embedding/'

datasets = {
    '%s-%s' % (task_name, f):
    load_data('%s%s/%s.%s.data' % (data_path, task_name, task_name, f))
    for f in ['train', 'valid', 'test']
}

# 语料id化
all_names, all_weights, all_token_ids, all_labels = [], [], [], []
train_token_ids = []
for name, data in datasets.items():
    a_token_ids, b_token_ids, labels = convert_to_ids(data, tokenizer, maxlen)
    all_names.append(name)
    all_weights.append(len(data))
    all_token_ids.append((a_token_ids, b_token_ids))
    all_labels.append(labels)
    train_token_ids.extend(a_token_ids)
    train_token_ids.extend(b_token_ids)

if task_name != 'PAWSX':
    np.random.shuffle(train_token_ids)
    train_token_ids = train_token_ids[:10000]

def collate_fn(batch):
    texts_list = [[] for _ in range(2)]
    for token_ids in batch:
        texts_list[0].append(token_ids)
        texts_list[1].append(token_ids)
    for i, texts in enumerate(texts_list):
        texts_list[i] = torch.tensor(sequence_padding(texts), dtype=torch.long, device=device)
    labels = torch.arange(texts_list[0].size(0), device=texts_list[0].device)
    return texts_list, labels

train_dataloader = DataLoader(ListDataset(data=train_token_ids), shuffle=True, collate_fn=collate_fn)


# 建立模型
class Model(BaseModel):
    def __init__(self, pool_method='cls', scale=20.0):
        super().__init__()
        self.pool_method = pool_method
        with_pool = 'linear' if pool_method == 'cls' else False
        output_all_encoded_layers = True if pool_method == 'first-last-avg' else False
        self.bert = build_transformer_model(config_path, checkpoint_path, model=model_type, 
                                            with_pool=with_pool, output_all_encoded_layers=output_all_encoded_layers)
    
    def forward(self, token_ids_list):
        reps = []
        for token_ids in token_ids_list:
            hidden_state1, pooler = self.bert([token_ids])
            rep = self.get_pool_emb(hidden_state1, pooler, attention_mask=token_ids.gt(0).long())
            reps.append(rep)
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])
        scores = self.cos_sim(embeddings_a, embeddings_b) * self.scale  # [btz, btz]
        return scores
    
    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pooler = self.bert([token_ids])
            output = self.get_pool_emb(hidden_state, pooler, attention_mask=token_ids.gt(0).long())
        return output

    def get_pool_emb(self, hidden_state, pooler, attention_mask):
        # 'first-last-avg', 'last-avg', 'cls', 'pooler'
        if self.pool_method == 'pooler':
            return pooler
        elif self.pool_method == 'cls':
            return hidden_state[:, 0]
        elif self.pool_method == 'last-avg':
            hidden_state = torch.sum(hidden_state * attention_mask[:, :, None], dim=1)
            attention_mask = torch.sum(attention_mask, dim=1)[:, None]
            return hidden_state / attention_mask
        elif self.pool_method == 'first-last-avg':
            hidden_state = torch.sum(hidden_state[0] * attention_mask[:, :, None], dim=1)
            hidden_state += torch.sum(hidden_state[-1] * attention_mask[:, :, None], dim=1)
            attention_mask = torch.sum(attention_mask, dim=1)[:, None]
            return hidden_state / (2 * attention_mask)
        else:
            raise ValueError('pool_method illegal')

    @staticmethod
    def cos_sim(a, b):
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

model = Model(pool_method=pooling).to(device)
model.compile(loss=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), 5e-6))

# SimCSE训练
model.fit(train_dataloader, steps_per_epoch=None, epochs=1)

# =============================模型预测=============================
# 语料向量化
all_vecs = []
for a_token_ids, b_token_ids in all_token_ids:
    a_vecs = model.encode(torch.tensor(a_token_ids, dtype=torch.long, device=device))
    b_vecs = model.encode(torch.tensor(b_token_ids, dtype=torch.long, device=device))
    all_vecs.append((a_vecs.cpu.numpy(), b_vecs.cpu.numpy()))

# 标准化，相似度，相关系数
all_corrcoefs = []
for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
    a_vecs = torch.nn.functional.normalize(a_vecs, p=2, dim=1)
    b_vecs = torch.nn.functional.normalize(b_vecs, p=2, dim=1)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = scipy.stats.spearmanr(labels, sims).correlation
    all_corrcoefs.append(corrcoef)

all_corrcoefs.extend([np.average(all_corrcoefs), np.average(all_corrcoefs, weights=all_weights)])

for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
    print('%s: %s' % (name, corrcoef))
#! -*- coding:utf-8 -*-
# loss: MultiNegativeRankingLoss, 和simcse一样，以batch中其他样本作为负样本

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset, get_pool_emb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sentence_transformers import evaluation
from config import config_path, checkpoint_path, dict_path, fst_train_file, dev_datapath, ir_path
import numpy as np
import pandas as pd
import random
import os

# 固定seed
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

maxlen = 64
batch_size = 64
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

train_dataloader = DataLoader(MyDataset(fst_train_file), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 验证集
ir_queries, ir_corpus, ir_relevant_docs = {}, {}, {}
with open(dev_datapath, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        qid, query, duplicate_ids = line.strip().split('\t')
        duplicate_ids = duplicate_ids.split(',')
        ir_queries[qid] = query
        ir_relevant_docs[qid] = set(duplicate_ids)
ir_corpus_df = pd.read_csv(ir_path, sep='\t')
ir_corpus_df.qid = ir_corpus_df.qid.astype('str')
ir_corpus = dict(zip(ir_corpus_df.qid.tolist(), ir_corpus_df.question.tolist()))
evaluate = evaluation.InformationRetrievalEvaluator(ir_queries, ir_corpus, ir_relevant_docs, name=choice)

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

    def encode(self, texts, **kwargs):
        token_ids_list = []
        for text in texts:
            token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
            token_ids_list.append(token_ids)
        token_ids_tensor = torch.tensor(sequence_padding(token_ids_list), dtype=torch.long, device=device)
        valid_dataloader = DataLoader(TensorDataset(token_ids_tensor), batch_size=batch_size)
        valid_sen_emb = []
        self.eval()
        with torch.no_grad():
            for token_ids in tqdm(valid_dataloader, desc='Evaluate'):
                token_ids = token_ids[0]
                hidden_state, pool_cls = self.bert([token_ids])
                output = get_pool_emb(hidden_state, pool_cls, token_ids.gt(0).long(), self.pool_method)
                valid_sen_emb.append(output)
        valid_sen_emb = torch.cat(valid_sen_emb, dim=0)
        return valid_sen_emb
    
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

class Evaluator(Callback):
    def on_dataloader_end(self, logs=None):
        model.train_dataloader = DataLoader(MyDataset(fst_train_file), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def on_epoch_end(self, global_step, epoch, logs=None):
        evaluate(model, epoch=model.epoch, steps=model.global_step, output_path='./')
        model.save_weights(f'./{choice}_best_weights_{model.epoch}.pt')

if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, 
            epochs=10, 
            steps_per_epoch=None, 
            callbacks=[evaluator]
            )
else:
    model.load_weights('best_model.pt')

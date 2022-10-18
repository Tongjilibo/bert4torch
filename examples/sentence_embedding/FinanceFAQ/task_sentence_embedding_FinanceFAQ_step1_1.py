#! -*- coding:utf-8 -*-
# loss: MultiNegativeRankingLoss, 和simcse一样，以batch中其他样本作为负样本

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset, get_pool_emb, seed_everything
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sentence_transformers import evaluation
from config import config_path, checkpoint_path, dict_path, fst_train_file, fst_dev_file, ir_path
import numpy as np
import pandas as pd

# 固定seed
seed_everything(42)

maxlen = 64
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# raw: 原始的版本
# random: 同一个标准问(组内)随机采样，组间互为负样本
# mul_ce: 原始版本修改版，组间也有正样本（标准问一致的时候）
choice = 'mul_ce'
print(f'using {choice} mode in step1 model'.center(60, '-'))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

if choice in {'raw', 'mul_ce'}:
    # 原始模式，可能同一个batch中会出现重复标问
    def collate_fn(batch):
        if choice == 'raw':
            labels = torch.arange(len(batch), device=device)
        else:
            labels = torch.eye(len(batch), dtype=torch.long, device=device)
            # 定位相同元素
            for i, (q_std1, _) in enumerate(batch):
                for j, (q_std2, _) in enumerate(batch[i+1:], start=i+1):
                    if q_std1 == q_std2:
                        labels[i, j] = 1
                        labels[j, i] = 1

        texts_list = [[] for _ in range(2)]
        for texts in batch:
            for i, text in enumerate(texts):
                token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
                texts_list[i].append(token_ids)
        for i, texts in enumerate(texts_list):
            texts_list[i] = torch.tensor(sequence_padding(texts), dtype=torch.long, device=device)
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


# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='cls', scale=20.0):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True, segment_vocab_size=0)
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

    def predict(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pool_cls = self.bert([token_ids])
            output = get_pool_emb(hidden_state, pool_cls, token_ids.gt(0).long(), self.pool_method)
        return output

    def encode(self, texts, **kwargs):
        token_ids_list = []
        for text in texts:
            token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
            token_ids_list.append(token_ids)
        token_ids_tensor = torch.tensor(sequence_padding(token_ids_list), dtype=torch.long)
        valid_dataloader = DataLoader(TensorDataset(token_ids_tensor), batch_size=batch_size)
        valid_sen_emb = []
        self.eval()
        for token_ids in tqdm(valid_dataloader, desc='Evaluate'):
            token_ids = token_ids[0].to(device)
            output = self.predict(token_ids)
            valid_sen_emb.append(output.cpu())
        valid_sen_emb = torch.cat(valid_sen_emb, dim=0)
        return valid_sen_emb
    
    @staticmethod
    def cos_sim(a, b):
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

model = Model().to(device)

# 多分类
class Myloss(nn.Module):
    def forward(self, y_pred, y_true):
        y_pred = torch.log(torch.softmax(y_pred, dim=-1)) * y_true  # [btz, btz]
        return -y_pred.sum() / len(y_pred)
        # y_pred_pos = (y_pred * y_true).sum(dim=-1)
        # y_pred_sum = torch.logsumexp(y_pred, dim=-1)
        # return (y_pred_sum - y_pred_pos).sum() / len(y_pred)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss = Myloss() if choice == 'mul_ce' else nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),
)

class Evaluator(Callback):
    def __init__(self):
        super().__init__()
        self.best_perf = 0

    def on_dataloader_end(self, logs=None):
        model.train_dataloader = DataLoader(MyDataset(fst_train_file), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def on_epoch_end(self, global_step, epoch, logs=None):
        perf = evaluate(model, epoch=model.epoch, steps=model.global_step, output_path='./')
        if perf > self.best_perf:
            self.best_perf = perf
            model.save_weights(f'./fst_best_weights_{choice}.pt')
        print(f'perf: {perf:.2f}, best perf: {self.best_perf:.2f}\n')

if __name__ == '__main__':
    # 训练集
    train_dataloader = DataLoader(MyDataset(fst_train_file), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 验证集
    ir_queries, ir_corpus, ir_relevant_docs = {}, {}, {}
    with open(fst_dev_file, 'r', encoding='utf-8') as f:
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

    evaluator = Evaluator()
    model.fit(train_dataloader, 
            epochs=10, 
            steps_per_epoch=None, 
            callbacks=[evaluator]
            )
else:
    model.load_weights(f'./fst_best_weights_{choice}.pt')

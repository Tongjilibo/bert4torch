#! -*- coding:utf-8 -*-
# loss: InfoNCE(即sentence_transformer中的MultiNegativeRankingLoss)
# 样本都是正负样本对，因此构造(正，正，负)的三元组时候，正样本对(正，正1)随机抽样负样本为（正，正1，负）
# 负样本对（正，负）重复正样本对（正，正，负）

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset, get_pool_emb, seed_everything
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import spearmanr
import random
from tqdm import tqdm
import sys

# =============================基本参数=============================
# pooling, task_name = sys.argv[1:]  # 传入参数
pooling, task_name = 'cls', 'ATEC'  # debug使用
print('pooling: ', pooling, ' task_name: ', task_name)
assert task_name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B']
assert pooling in {'first-last-avg', 'last-avg', 'cls', 'pooler'}

maxlen = 64 if task_name != 'PAWSX' else 128
batch_size = 32
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# ===========================数据预处理===========================
# 训练
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
    train_data, all_texts = {}, []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) != 3:
                continue
            text1, text2, label = l
            label = str(int(int(label) > 2.5)) if task_name == 'STS-B' else label
            if text1 not in train_data:
                train_data[text1] = {'0': set(), '1': set()}
            train_data[text1][label].add(text2)
            if text2 not in train_data:
                train_data[text2] = {'0': set(), '1': set()}
            train_data[text2][label].add(text1)
            all_texts.extend([text1, text2])

    train_samples = []
    for sent1, others in train_data.items():
        if len(others['1']) == 0:
            others['1'] = [sent1]  # 没有正样本，使用自身作为正阳本，这里其实就是无监督
        elif len(others['0']) == 0:
            others['0'] = [random.choice(all_texts)]  # 没有负样本，随机挑选一个负样本

        # sentence bert的逻辑是下面两个都加进去，这样的问题是如果shuffle=False，处于同一个batch中，相似句可能label给的负样本
        if random.random() < 0.5:
            train_samples.append((sent1, random.choice(list(others['1'])), random.choice(list(others['0']))))
        else:
            train_samples.append((random.choice(list(others['1'])), sent1, random.choice(list(others['0']))))
    return train_samples

train_data = get_data(f'F:/Projects/data/corpus/sentence_embedding/{task_name}/{task_name}.train.data')
train_dataloader = DataLoader(ListDataset(data=train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据
        单条格式：(文本1, 文本2, 标签id)
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = l.strip().split('\t')
                if len(l) == 3:
                    D.append((l[0], l[1], int(l[2])))
        return D

def collate_fn_eval(batch):
    batch_token1_ids, batch_token2_ids, batch_labels = [], [], []
    for text1, text2, label in batch:
        token1_ids, _ = tokenizer.encode(text1, maxlen=maxlen)
        batch_token1_ids.append(token1_ids)
        token2_ids, _ = tokenizer.encode(text2, maxlen=maxlen)
        batch_token2_ids.append(token2_ids)
        batch_labels.append([label])

    batch_token1_ids = torch.tensor(sequence_padding(batch_token1_ids), dtype=torch.long, device=device)
    batch_token2_ids = torch.tensor(sequence_padding(batch_token2_ids), dtype=torch.long, device=device)

    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return (batch_token1_ids, batch_token2_ids), batch_labels.flatten()

# 加载数据集
valid_dataloader = DataLoader(MyDataset(f'F:/Projects/data/corpus/sentence_embedding/{task_name}/{task_name}.valid.data'), batch_size=batch_size, collate_fn=collate_fn_eval)
test_dataloader = DataLoader(MyDataset(f'F:/Projects/data/corpus/sentence_embedding/{task_name}/{task_name}.test.data'), batch_size=batch_size, collate_fn=collate_fn_eval)

# 建立模型
class Model(BaseModel):
    def __init__(self, pool_method='cls', scale=20.0):
        super().__init__()
        self.pool_method = pool_method
        with_pool = 'linear' if pool_method == 'pooler' else True
        output_all_encoded_layers = True if pool_method == 'first-last-avg' else False
        self.bert = build_transformer_model(config_path, checkpoint_path, segment_vocab_size=0, 
                                            with_pool=with_pool, output_all_encoded_layers=output_all_encoded_layers)
        self.scale = scale
    
    def forward(self, token_ids_list):
        reps = []
        for token_ids in token_ids_list:
            hidden_state1, pooler = self.bert([token_ids])
            rep = get_pool_emb(hidden_state1, pooler, token_ids.gt(0).long(), self.pool_method)
            reps.append(rep)
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])
        scores = self.cos_sim(embeddings_a, embeddings_b) * self.scale  # [btz, btz]
        return scores
    
    def predict(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pooler = self.bert([token_ids])
            output = get_pool_emb(hidden_state, pooler, token_ids.gt(0).long(), self.pool_method)
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
    optimizer=optim.Adam(model.parameters(), lr=2e-5),
)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_consine = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_consine = self.evaluate(valid_dataloader)
        test_consine = self.evaluate(test_dataloader)

        if val_consine > self.best_val_consine:
            self.best_val_consine = val_consine
            # model.save_weights('best_model.pt')
        print(f'valid_consine: {val_consine:.5f}, test_consine: {test_consine:.5f}, best_val_consine: {self.best_val_consine:.5f}\n')
        
        # 重新生成dataloader，重新random选择样本
        train_data = get_data(f'F:/Projects/data/corpus/sentence_embedding/{task_name}/{task_name}.train.data')
        model.train_dataloader = DataLoader(ListDataset(data=train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

    # 定义评价函数
    def evaluate(self, data):
        embeddings1, embeddings2, labels = [], [], []
        for (batch_token_ids1, batch_token_ids2), batch_labels in tqdm(data, desc='Evaluate'):
            embeddings1.append(model.predict(batch_token_ids1))
            embeddings2.append(model.predict(batch_token_ids2))
            labels.append(batch_labels)
        embeddings1 = torch.cat(embeddings1).cpu().numpy()
        embeddings2 = torch.cat(embeddings2).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))  # cosine距离是1-paired
        eval_pearson_cosine, _ = spearmanr(labels, cosine_scores)
        return eval_pearson_cosine

if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, 
            epochs=10, 
            steps_per_epoch=None, 
            callbacks=[evaluator]
            )
else:
    model.load_weights('best_model.pt')

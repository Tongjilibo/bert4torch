#! -*- coding:utf-8 -*-
# 原项目：https://kexue.fm/archives/8847

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset, get_pool_emb, seed_everything
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import spearmanr
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

def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for text1, text2, label in batch:
        for text in [text1, text2]:
            token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_labels.append([label])

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.float, device=device)
    return batch_token_ids, batch_labels.flatten()

# 加载数据集
train_dataloader = DataLoader(MyDataset(f'F:/Projects/data/corpus/sentence_embedding/{task_name}/{task_name}.train.data'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset(f'F:/Projects/data/corpus/sentence_embedding/{task_name}/{task_name}.valid.data'), batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(MyDataset(f'F:/Projects/data/corpus/sentence_embedding/{task_name}/{task_name}.test.data'), batch_size=batch_size, collate_fn=collate_fn)

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='cls'):
        super().__init__()
        self.pool_method = pool_method
        with_pool = 'linear' if pool_method == 'pooler' else True
        output_all_encoded_layers = True if pool_method == 'first-last-avg' else False
        self.bert = build_transformer_model(config_path, checkpoint_path, segment_vocab_size=0,
                                            with_pool=with_pool, output_all_encoded_layers=output_all_encoded_layers)

    def forward(self, token_ids):
        hidden_state, pooler = self.bert([token_ids])
        sem_emb = get_pool_emb(hidden_state, pooler, token_ids.gt(0).long(), self.pool_method)
        return sem_emb

model = Model().to(device)

class MyLoss(nn.Module):
    def forward(self, y_pred, y_true):
        # 1. 取出真实的标签
        y_true = y_true[::2]    # tensor([1, 0, 1]) 真实的标签

        # 2. 对输出的句子向量进行l2归一化   后面只需要对应为相乘  就可以得到cos值了
        norms = (y_pred ** 2).sum(axis=1, keepdims=True) ** 0.5
        # y_pred = y_pred / torch.clip(norms, 1e-8, torch.inf)
        y_pred = y_pred / norms

        # 3. 奇偶向量相乘
        y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * 20

        # 4. 取出负例-正例的差值
        y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
        # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
        y_true = y_true[:, None] < y_true[None, :]   # 取出负例-正例的差值
        y_true = y_true.float()
        y_pred = y_pred - (1 - y_true) * 1e12
        y_pred = y_pred.view(-1)
        y_pred = torch.cat((torch.tensor([0.0], device=device), y_pred), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
        return torch.logsumexp(y_pred, dim=0)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=MyLoss(),
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

    # 定义评价函数
    def evaluate(self, data):
        embeddings1, embeddings2, labels = [], [], []
        for batch_token_ids, batch_labels in tqdm(data, desc='Evaluate'):
            embeddings = model.predict(batch_token_ids)
            embeddings1.append(embeddings[::2])
            embeddings2.append(embeddings[1::2])
            labels.append(batch_labels[::2])
        embeddings1 = torch.cat(embeddings1).cpu().numpy()
        embeddings2 = torch.cat(embeddings2).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        eval_pearson_cosine, _ = spearmanr(labels, cosine_scores)
        return eval_pearson_cosine

if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=5, steps_per_epoch=None, callbacks=[evaluator])
else:
    model.load_weights('best_model.pt')

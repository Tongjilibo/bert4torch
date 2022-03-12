#! -*- coding:utf-8 -*-
# 语义相似度任务-无监督：训练集为网上pretrain数据, dev集为sts-b


from bert4pytorch.tokenizers import Tokenizer
from bert4pytorch.models import build_transformer_model, BaseModel
from bert4pytorch.snippets import sequence_padding, Callback, ListDataset
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import pearsonr, spearmanr
import copy
import random
import numpy as np
random.seed(2022)
np.random.seed(2002)

maxlen = 256
batch_size = 8
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def collate_fn(batch):
    def add_noise(token_ids, del_ratio=0.6):
        n = len(token_ids)
        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True # guarantee that at least one word remains
        return list(np.array(token_ids)[keep_or_not])

    texts_list = [[] for _ in range(3)]
    
    for text in batch:
        token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
        texts_list[0].append([tokenizer._token_start_id] + add_noise(token_ids[1:-1]) + [tokenizer._token_end_id])
        texts_list[1].append(token_ids[:-1])
        texts_list[2].append(token_ids[1:])

    for i, texts in enumerate(texts_list):
        texts_list[i] = torch.tensor(sequence_padding(texts), dtype=torch.long, device=device)
    
    return texts_list[:2], texts_list[2].flatten()

# 加载数据集
def get_data(filename):
    train_data = []
    with open(filename, encoding='utf-8') as f:
        for row, l in enumerate(f):
            if row == 0:  # 跳过首行
                continue
            text = l.strip().replace(' ', '')
            if len(text) > 0:
                train_data.append(text)
    return train_data

train_data = get_data('F:/Projects/data/corpus/pretrain/film/film.txt')
train_dataloader = DataLoader(ListDataset(data=train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
from task_sentence_embedding_sbert_sts_b__CosineSimilarityLoss import valid_dataloader

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='mean', scale=20.0):
        super().__init__()
        self.encoder, self.config = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True, with_mlm=True, return_model_config=True, segment_vocab_size=0)
        self.decoder = self.encoder # 这里可以通过使用copy和不使用copy来决定一个模型还是两个独立的模型
        self.pool_method = pool_method
        self.scale = scale

    def forward(self, token_ids_list):
        token_ids1 = token_ids_list[0]
        hidden_state1, pool_cls1, _ = self.encoder([token_ids1])
        embeddings_a = self.get_pool_emb(hidden_state1, pool_cls1, attention_mask=token_ids1.gt(0).long())

        token_ids2 = token_ids_list[1]
        _, _, mlm_score2 = self.decoder([token_ids2, embeddings_a.unsqueeze(1), torch.ones_like(token_ids1)[:, 0:1]])

        return mlm_score2.reshape(-1, mlm_score2.shape[-1])

    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pool_cls, _ = self.encoder([token_ids])
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

model = Model().to(device)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=nn.CrossEntropyLoss(ignore_index=0),
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
            steps_per_epoch=100, 
            callbacks=[evaluator]
            )
else:
    model.load_weights('best_model.pt')

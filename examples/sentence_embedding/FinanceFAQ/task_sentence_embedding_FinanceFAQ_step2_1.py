#! -*- coding:utf-8 -*-
# 二阶段训练: 基于困难负样本的进一步精排

from bert4torch.tokenizers import Tokenizer
from bert4torch.losses import ContrastiveLoss
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset, get_pool_emb, seed_everything
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from config import config_path, checkpoint_path, dict_path, sec_train_file, sec_dev_file
import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import roc_auc_score

# 固定seed
seed_everything(42)

maxlen = 64
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def collate_fn(batch):
    tokens_ids_list = [[] for _ in range(2)]
    labels = []
    for text1, text2, label in batch:
        tokens_ids_list[0].append(tokenizer.encode(text1, maxlen=maxlen)[0])
        tokens_ids_list[1].append(tokenizer.encode(text2, maxlen=maxlen)[0])
        labels.append(label)

    for i, token_ids in enumerate(tokens_ids_list):
        tokens_ids_list[i] = torch.tensor(sequence_padding(token_ids), dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return tokens_ids_list, labels

class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                label, text1, text2 = l.strip().split('\t')
                D.append((text1.replace(' ', ''), text2.replace(' ', ''), int(label)))
        return D

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='cls'):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True, segment_vocab_size=0)
        self.pool_method = pool_method

    def forward(self, token_ids_list):
        reps = []
        for token_ids in token_ids_list:
            hidden_state1, pool_cls1 = self.bert([token_ids])
            rep = get_pool_emb(hidden_state1, pool_cls1, token_ids.gt(0).long(), self.pool_method)
            reps.append(rep)
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])
        scores = 1 - torch.cosine_similarity(embeddings_a, embeddings_b)
        return scores

    def predict(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pool_cls = self.bert([token_ids])
            output = get_pool_emb(hidden_state, pool_cls, token_ids.gt(0).long(), self.pool_method)
        return output
    
    def encode(self, texts):
        token_ids_list = []
        for text in texts:
            token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
            token_ids_list.append(token_ids)
        token_ids_tensor = torch.tensor(sequence_padding(token_ids_list), dtype=torch.long)
        valid_dataloader = DataLoader(TensorDataset(token_ids_tensor), batch_size=batch_size)
        valid_sen_emb = []
        for token_ids in tqdm(valid_dataloader, desc='Evaluate'):
            token_ids = token_ids[0].to(device)
            output = self.predict(token_ids)
            valid_sen_emb.append(output.cpu())
        valid_sen_emb = torch.cat(valid_sen_emb, dim=0)
        return valid_sen_emb

model = Model().to(device)


# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=ContrastiveLoss(margin=0.8),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),
)

class Evaluator(Callback):
    def __init__(self):
        super().__init__()
        self.best_val_auc = 0

    def on_dataloader_end(self, logs=None):
        model.train_dataloader = DataLoader(MyDataset(sec_train_file), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_auc = self.evaluate(valid_dataloader)
        if val_auc >= self.best_val_auc:
            self.best_val_auc = val_auc
            model.save_weights('sec_best_weights.pt')
        print(f'val_auc: {val_auc:.5f}, best_val_auc: {self.best_val_auc:.5f}\n')
    
    def evaluate(self, data):
        embeddings1, embeddings2, labels = [], [], []
        for (batch_token1_ids, batch_token2_ids), batch_labels in tqdm(data):
            embeddings1.append(model.predict(batch_token1_ids).cpu())
            embeddings2.append(model.predict(batch_token2_ids).cpu())
            labels.append(batch_labels.cpu())
        embeddings1 = torch.cat(embeddings1).numpy()
        embeddings2 = torch.cat(embeddings2).numpy()
        labels = torch.cat(labels).numpy()
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        auc = roc_auc_score(labels, cosine_scores)
        return auc


if __name__ == '__main__':
    train_dataloader = DataLoader(MyDataset(sec_train_file), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_dataloader = DataLoader(MyDataset(sec_dev_file), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    evaluator = Evaluator()
    model.fit(train_dataloader, 
            epochs=10, 
            steps_per_epoch=None, 
            callbacks=[evaluator]
            )
else:
    model.load_weights('sec_best_weights.pt')

#! -*- coding:utf-8 -*-
# 语义相似度任务-无监督：训练集为网上pretrain数据, dev集为sts-b
# ContrastiveTensionLoss: 同一个sentence送入两个模型，pooling后的点积要大

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import pearsonr, spearmanr
import copy
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
    texts_list = [[] for _ in range(2)]
    labels = []
    pos_id = random.randint(0, len(batch)-1)
    pos_token_ids, _ = tokenizer.encode(batch[pos_id], maxlen=maxlen)
    texts_list[0].append(pos_token_ids)
    texts_list[1].append(pos_token_ids)
    labels.append(1)
    for neg_id in range(len(batch)):
        if neg_id == pos_id:
            continue
        elif random.random() < 0.5:
            neg_token_ids, _ = tokenizer.encode(batch[neg_id], maxlen=maxlen)
            texts_list[0].append(pos_token_ids)
            texts_list[1].append(neg_token_ids)
            labels.append(0)
        else:
            neg_token_ids, _ = tokenizer.encode(batch[neg_id], maxlen=maxlen)
            texts_list[0].append(neg_token_ids)
            texts_list[1].append(pos_token_ids)
            labels.append(0)
    for i, texts in enumerate(texts_list):
        texts_list[i] = torch.tensor(sequence_padding(texts), dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.float, device=device)
    return texts_list, labels

# 加载数据集
def get_data(filename):
    train_data = []
    with open(filename, encoding='utf-8') as f:
        for row, l in enumerate(f):
            if row == 0:  # 跳过首行
                continue
            text = l.strip().replace(' ', '')
            train_data.append(text)
    return train_data

train_data = get_data('F:/Projects/data/corpus/pretrain/film/film.txt')
train_dataloader = DataLoader(ListDataset(data=train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
from task_sentence_embedding_sbert_sts_b__CosineSimilarityLoss import valid_dataloader

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='mean', scale=20.0):
        super().__init__()
        self.model1, self.config = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True, return_model_config=True, segment_vocab_size=0)
        self.model2 = copy.deepcopy(self.model1)
        self.pool_method = pool_method
        self.scale = scale

    def forward(self, token_ids_list):
        token_ids1 = token_ids_list[0]
        hidden_state1, pool_cls1 = self.model1([token_ids1])
        embeddings_a = self.get_pool_emb(hidden_state1, pool_cls1, attention_mask=token_ids1.gt(0).long())

        token_ids2 = token_ids_list[1]
        hidden_state2, pool_cls2 = self.model2([token_ids2])
        embeddings_b = self.get_pool_emb(hidden_state2, pool_cls2, attention_mask=token_ids2.gt(0).long())

        return torch.matmul(embeddings_a[:, None], embeddings_b[:, :, None]).squeeze(-1).squeeze(-1)  # [btz]

    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pool_cls = self.model1([token_ids])
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
    loss=nn.BCEWithLogitsLoss(reduction='mean'),
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
            steps_per_epoch=500, 
            callbacks=[evaluator]
            )
else:
    model.load_weights('best_model.pt')

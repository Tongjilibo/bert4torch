#! -*- coding:utf-8 -*-
# 语义相似度任务：数据集LCQMC
# loss: ContrastiveLoss

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset
from bert4torch.losses import ContrastiveLoss
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

maxlen = 128
batch_size = 12
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                text1, text2, label = l.strip().split('\t')
                D.append((text1, text2, int(label)))
        return D

def collate_fn(batch):
    batch_token1_ids, batch_token2_ids, batch_labels = [], [], []
    for text1, text2, label in batch:
        token1_ids, _ = tokenizer.encode(text1, maxlen=maxlen)
        batch_token1_ids.append(token1_ids)
        token2_ids, _ = tokenizer.encode(text2, maxlen=maxlen)
        batch_token2_ids.append(token2_ids)
        batch_labels.append([label])

    batch_token1_ids = torch.tensor(sequence_padding(batch_token1_ids), dtype=torch.long, device=device)
    batch_token2_ids = torch.tensor(sequence_padding(batch_token2_ids), dtype=torch.long, device=device)

    batch_labels = torch.tensor(batch_labels, dtype=torch.float, device=device)
    return (batch_token1_ids, batch_token2_ids), batch_labels.flatten()

# 加载数据集
train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/语义相似度/LCQMC/LCQMC.train.data'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/语义相似度/LCQMC/LCQMC.valid.data'), batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/语义相似度/LCQMC/LCQMC.test.data'), batch_size=batch_size, collate_fn=collate_fn)

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='mean'):
        super().__init__()
        self.bert, self.config = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True, return_model_config=True, segment_vocab_size=0)
        self.pool_method = pool_method

    def forward(self, token1_ids, token2_ids):
        hidden_state1, pool_cls1 = self.bert([token1_ids])
        pool_emb1 = self.get_pool_emb(hidden_state1, pool_cls1, attention_mask=token1_ids.gt(0).long())
        
        hidden_state2, pool_cls2 = self.bert([token2_ids])
        pool_emb2 = self.get_pool_emb(hidden_state2, pool_cls2, attention_mask=token2_ids.gt(0).long())

        return 1- torch.cosine_similarity(pool_emb1, pool_emb2)
    
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

    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pool_cls = self.bert([token_ids])
            attention_mask = token_ids.gt(0).long()
            output = self.get_pool_emb(hidden_state, pool_cls, attention_mask)
        return output

model = Model().to(device)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=ContrastiveLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),  # 用足够小的学习率
)

# 定义评价函数
def evaluate(data):
    embeddings1, embeddings2, labels = [], [], []
    for (batch_token1_ids, batch_token2_ids), batch_labels in data:
        embeddings1.append(model.encode(batch_token1_ids))
        embeddings2.append(model.encode(batch_token2_ids))
        labels.append(batch_labels)
    embeddings1 = torch.concat(embeddings1).cpu().numpy()
    embeddings2 = torch.concat(embeddings2).cpu().numpy()
    labels = torch.concat(labels).cpu().numpy()
    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    auc = roc_auc_score(labels, cosine_scores)
    return auc


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_auc = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_auc = evaluate(valid_dataloader)
        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            # model.save_weights('best_model.pt')
        print(f'val_auc: {val_auc:.5f}, best_val_auc: {self.best_val_auc:.5f}\n')


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=20, steps_per_epoch=200, callbacks=[evaluator])
else:
    model.load_weights('best_model.pt')

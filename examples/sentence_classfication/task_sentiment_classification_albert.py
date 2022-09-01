#! -*- coding:utf-8 -*-
# 情感分类例子，加载albert_zh权重(https://github.com/brightmart/albert_zh)
# valid_acc: 94.46, test_acc: 93.98


import numpy as np
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, text_segmentate, ListDataset, seed_everything, get_pool_emb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
import random 
import os
import numpy as np

maxlen = 256
batch_size = 16
config_path = 'F:/Projects/pretrain_ckpt/albert/[brightmart_tf_small]--albert_small_zh_google/albert_config_small_google.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/albert/[brightmart_tf_small]--albert_small_zh_google/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/albert/[brightmart_tf_small]--albert_small_zh_google/vocab.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter(log_dir='./summary')  # prepare summary writer

seed_everything(42)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据
        单条格式：(文本, 标签id)
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                text, label = l.strip().split('\t')
                D.append((text, int(label)))
        return D

def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for text, label in batch:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append([label])

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], batch_labels.flatten()

# 加载数据集
train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.train.data'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.valid.data'), batch_size=batch_size, collate_fn=collate_fn) 
test_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.test.data'),  batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='cls') -> None:
        super().__init__()
        self.pool_method = pool_method
        self.bert = build_transformer_model(config_path, checkpoint_path, model='albert', with_pool=True)  # 建立模型，加载权重
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.bert.configs['hidden_size'], 2)

    def forward(self, token_ids, segment_ids):
        hidden_states, pooling = self.bert([token_ids, segment_ids])
        pooled_output = get_pool_emb(hidden_states, pooling, token_ids.gt(0).long(), self.pool_method)
        output = self.dropout(pooled_output)
        output = self.dense(output)
        return output
model = Model().to(device)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),
    metrics=['accuracy']
)

# 定义评价函数
def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        total += len(y_true)
        right += (y_true == y_pred).sum().item()
    return right / total


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    # def on_batch_end(self, global_step, local_step, logs=None):
    #     if global_step % 10 == 0:
    #         writer.add_scalar(f"train/loss", logs['loss'], global_step)
    #         val_acc = evaluate(valid_dataloader)
    #         writer.add_scalar(f"valid/acc", val_acc, global_step)

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_acc = evaluate(valid_dataloader)
        test_acc = evaluate(test_dataloader)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.pt')
        print(f'val_acc: {val_acc:.5f}, test_acc: {test_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator])
else:
    model.load_weights('best_model.pt')

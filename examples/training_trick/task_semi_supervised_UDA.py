#! -*- coding:utf-8 -*-
# 以文本分类为例的半监督学习UDA策略，https://arxiv.org/abs/1904.12848

from turtle import forward
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, text_segmentate, ListDataset
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import math

maxlen = 128
batch_size = 8
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filenames):
        """加载数据，并尽量划分为不超过maxlen的句子
        """
        D = []
        seps, strips = u'\n。！？!?；;，, ', u'；;，, '
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    text, label = l.strip().split('\t')
                    for t in text_segmentate(text, maxlen - 2, seps, strips):
                        D.append((t, int(label)))
        return D

train_dataset = MyDataset(['E:/Github/bert4torch/examples/datasets/sentiment/sentiment.train.data'])
valid_dataset = MyDataset(['E:/Github/bert4torch/examples/datasets/sentiment/sentiment.valid.data'])
test_dataset = MyDataset(['E:/Github/bert4torch/examples/datasets/sentiment/sentiment.test.data'])

# 理论上应该收集任务领域类的无监督数据，这里用所有的监督数据来作无监督数据
unsup_dataset =  [sen for sen, _ in (train_dataset.data + valid_dataset.data + test_dataset.data)]

def collate_fn(batch):
    def add_noise(token_ids, del_ratio=0.3):
        '''这里用随机删除做简单示例，实际中可以使用增删改等多种noise方案
        '''
        n = len(token_ids)
        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True # guarantee that at least one word remains
        return list(np.array(token_ids)[keep_or_not])

    # batch_token_ids包含三部分，第一部分是有监督数据，第二部分是领域类的无监督数据，第三部分是无监督数据经数据增强后的数据
    batch_token_ids, batch_labels = [[], [], []], []
    for text, label in batch:
        token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
        batch_token_ids[0].append(token_ids)
        batch_labels.append([label])
        # 无监督部分
        unsup_text = random.choice(unsup_dataset)  # 随机挑一个无监督数据
        token_ids, _ = tokenizer.encode(unsup_text, maxlen=maxlen)
        batch_token_ids[1].append(token_ids)
        batch_token_ids[2].append(token_ids[:1] + add_noise(token_ids[1:-1]) + token_ids[-1:])  # 无监督数据增强

    batch_token_ids = [j for i in batch_token_ids for j in i]
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids], batch_labels.flatten()

# 加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn) 
test_dataloader = DataLoader(test_dataset,  batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True, segment_vocab_size=0)
        self.dense = nn.Linear(768, 2)

    def forward(self, token_ids):
        _, pooled_output = self.bert(token_ids)
        output = self.dense(pooled_output)
        return output
model = Model().to(device)

class MyLoss(nn.Module):
    def __init__(self, tsa_schedule=None):
        super().__init__()
        self.loss_sup = nn.CrossEntropyLoss()
        self.loss_unsup = nn.KLDivLoss(reduction='batchmean')
        self.tsa_schedule = tsa_schedule

    def forward(self, y_pred, y_true_sup):
        sup_size = y_true_sup.size(0)
        unsup_size = (y_pred.size(0) - sup_size) // 2

        # 有监督部分, 用交叉熵损失
        y_pred_sup = y_pred[:sup_size]
        if self.tsa_schedule is None:
            loss_sup = self.loss_sup(y_pred_sup, y_true_sup)
        else:  # 使用tsa来去掉预测概率较高的有监督样本
            threshold = self.get_tsa_threshold(self.tsa_schedule, model.global_step, model.total_steps, start=0.8, end=1)  # 二分类
            true_prob = torch.gather(F.softmax(y_pred_sup, dim=-1), dim=1, index=y_true_sup[:, None])
            sel_rows = true_prob.lt(threshold).sum(dim=-1).gt(0)  # 仅保留小于阈值的样本
            loss_sup = self.loss_sup(y_pred_sup[sel_rows], y_true_sup[sel_rows]) if sel_rows.sum() > 0 else 0

        # 无监督部分，这里用KL散度，也可以用交叉熵
        y_true_unsup = y_pred[sup_size:sup_size+unsup_size]
        y_true_unsup = F.softmax(y_true_unsup.detach(), dim=-1)
        y_pred_unsup = F.log_softmax(y_pred[sup_size+unsup_size:], dim=-1)
        loss_unsup = self.loss_unsup(y_pred_unsup, y_true_unsup)
        return loss_sup + loss_unsup

    @ staticmethod
    def get_tsa_threshold(schedule, global_step, num_train_steps, start, end):
        training_progress = global_step / num_train_steps
        if schedule == "linear_schedule":
            threshold = training_progress
        elif schedule == "exp_schedule":
            scale = 5
            threshold = math.exp((training_progress - 1) * scale)
        elif schedule == "log_schedule":
            scale = 5
            threshold = 1 - math.exp((-training_progress) * scale)
        return threshold * (end - start) + start

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=MyLoss(tsa_schedule=None),  # 这里可换用不同的策略
    optimizer=optim.Adam(model.parameters(), lr=2e-5),  # 用足够小的学习率
)

# 定义评价函数
def evaluate(data):
    total, right = 0., 0.
    for inputs, y_true in data:
        inputs = [inputs[0][:y_true.size(0)]]
        y_pred = model.predict(inputs).argmax(axis=1)
        total += len(y_true)
        right += (y_true == y_pred).sum().item()
    return right / total


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_acc = evaluate(valid_dataloader)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.pt')
        print(f'val_acc: {val_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=20, steps_per_epoch=100, callbacks=[evaluator])
else:
    model.load_weights('best_model.pt')

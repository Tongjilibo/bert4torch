#! -*- coding:utf-8 -*-
# 情感分析例子，利用MLM做 Zero-Shot/Few-Shot/Semi-Supervised Learning
# 参考项目：https://github.com/bojone/Pattern-Exploiting-Training
# 指标如下，由于没有固定随机化因子，因此下述指标可能略有波动
# zero-shot1: 0.8517/0.8437
# zero-shot2: 0.8811/0.8707
# few-shot:   0.8896/0.8910
# semi-sup:   0.9024/0.8948

import torch
import torch.nn as nn
import numpy as np
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model
from torch.optim import Adam
import torch.nn.functional as F
from bert4torch.snippets import sequence_padding, ListDataset, Callback
from torch.utils.data import DataLoader

num_classes = 2
maxlen = 256
batch_size = 16
config_path = 'F:/Projects/pretrain_ckpt/robert/[hit_torch_base]--chinese-roberta-wwm-ext-base/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/robert/[hit_torch_base]--chinese-roberta-wwm-ext-base/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/robert/[hit_torch_base]--chinese-roberta-wwm-ext-base/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
choice = 'semi-sup'  # zero-shot1, zero-shot2, few-shot, semi-sup

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D

# 加载数据集
train_data = load_data('F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.train.data')
valid_data = load_data('F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.valid.data')
test_data = load_data('F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.test.data')

# 模拟标注和非标注数据
train_frac = 0.01  # 标注数据的比例
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]

if choice == 'zero-shot2':
    train_data = unlabeled_data  # 仅使用无监督数据继续mlm预训练
elif choice == 'few-shot':
    train_data = train_data[:num_labeled]  # 仅使用少量监督数据
elif choice == 'semi-sup':  # 少量监督数据和全量无监督数据做半监督
    train_data = train_data[:num_labeled]
    train_data = train_data + unlabeled_data

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述
prefix = u'很满意。'
mask_idx = 1
pos_id = tokenizer.token_to_id(u'很')
neg_id = tokenizer.token_to_id(u'不')


def random_masking(token_ids):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target

class MyDataset(ListDataset):
    def collate_fn(self, batch):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for text, label in batch:
            if label != 2:
                text = prefix + text
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            if self.kwargs['random']:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            if label == 0:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = neg_id
            elif label == 1:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = pos_id
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
        batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
        batch_output_ids = torch.tensor(sequence_padding(batch_output_ids), dtype=torch.long, device=device)
        return [batch_token_ids, batch_segment_ids], batch_output_ids

# 加载数据集
train_dataset = MyDataset(data=train_data, random=True)
valid_dataset = MyDataset(data=valid_data, random=False)
test_dataset = MyDataset(data=test_data, random=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn) 
test_dataloader = DataLoader(test_dataset,  batch_size=batch_size, collate_fn=test_dataset.collate_fn) 

# 加载预训练模型
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True).to(device)

class MyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, y_preds, y_true):
        y_pred = y_preds[1]
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        loss = super().forward(y_pred, y_true.flatten())
        return loss

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=MyLoss(ignore_index=0),
    optimizer=Adam(model.parameters(), lr=2e-5),
)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_acc = self.evaluate(valid_dataloader)
        test_acc = self.evaluate(test_dataloader)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.pt')
        print(f'[{choice}]  valid_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}, best_val_acc: {self.best_val_acc:.4f}\n')

    @staticmethod
    def evaluate(data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = F.softmax(model.predict(x_true)[1], dim=-1)
            y_pred = y_pred[:, mask_idx, [neg_id, pos_id]].argmax(axis=1)
            y_true = (y_true[:, mask_idx] == pos_id).long()
            total += len(y_true)
            right += (y_true == y_pred).sum().item()
        return right / total


if __name__ == '__main__':
    evaluator = Evaluator()
    if choice == 'zero-shot1':
        valid_acc = evaluator.evaluate(valid_dataloader)
        test_acc = evaluator.evaluate(test_dataloader)
        print(f'[{choice}]  valid_acc: {valid_acc:.4f}, test_acc: {test_acc:.4f}')
    else:
        model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator])
else:
    model.load_weights('best_model.pt')

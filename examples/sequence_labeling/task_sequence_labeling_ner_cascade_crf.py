#! -*- coding:utf-8 -*-
# bert+crf 级联方法，一阶段识别BIO，二阶段识别对应的分类
# 参考博客：https://zhuanlan.zhihu.com/p/166496466
# 数据集：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# [valid_f1]  token_level: 98.11； entity_level: 96.23

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from bert4torch.snippets import sequence_padding, Callback, ListDataset, seed_everything
from bert4torch.layers import CRF
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from tqdm import tqdm

maxlen = 256
batch_size = 16
categories = ['LOC', 'PER', 'ORG']

# BERT base
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定seed
seed_everything(42)

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            for l in f.split('\n\n'):
                if not l:
                    continue
                d = ['']
                for i, c in enumerate(l.split('\n')):
                    char, flag = c.split(' ')
                    d[0] += char
                    if flag[0] == 'B':
                        d.append([i, i, flag[2:]])
                    elif flag[0] == 'I':
                        d[-1][1] = i
                D.append(d)
        return D


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def collate_fn(batch):
    batch_token_ids, batch_labels, batch_entity_ids, batch_entity_labels = [], [], [], []
    for d in batch:
        tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
        mapping = tokenizer.rematch(d[0], tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)
        labels = np.zeros(len(token_ids))
        entity_ids, entity_labels = [], []
        for start, end, label in d[1:]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                labels[start] = 1 # 标记B
                labels[start + 1:end + 1] = 2 # 标记I
                entity_ids.append([start, end])
                entity_labels.append(categories.index(label)+1)

        if not entity_ids:  # 至少要有一个标签
            entity_ids.append([0, 0])  # 如果没有则用0填充
            entity_labels.append(0)

        batch_token_ids.append(token_ids)
        batch_labels.append(labels)
        batch_entity_ids.append(entity_ids)
        batch_entity_labels.append(entity_labels)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels), dtype=torch.long, device=device)
    batch_entity_ids = torch.tensor(sequence_padding(batch_entity_ids), dtype=torch.long, device=device)  # [btz, 实体个数，start/end]
    batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels), dtype=torch.long, device=device)  # [btz, 实体个数]
    return [batch_token_ids, batch_entity_ids], [batch_labels, batch_entity_labels]

# 转换数据集
train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/ner/china-people-daily-ner-corpus/example.train'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/ner/china-people-daily-ner-corpus/example.dev'), batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0)
        self.dense1 = nn.Linear(768, len(categories))
        self.dense2 = nn.Linear(768, len(categories)+1)  # 包含padding
        self.crf = CRF(len(categories))

    def forward(self, inputs):
        # 一阶段的输出
        token_ids, entity_ids = inputs[0], inputs[1]
        last_hidden_state = self.bert([token_ids])  # [btz, seq_len, hdsz]
        emission_score = self.dense1(last_hidden_state)  # [bts, seq_len, tag_size]
        attention_mask = token_ids.gt(0)

        # 二阶段输出
        btz, entity_count, _ = entity_ids.shape
        hidden_size = last_hidden_state.shape[-1]
        entity_ids = entity_ids.reshape(btz, -1, 1).repeat(1, 1, hidden_size)
        entity_states = torch.gather(last_hidden_state, dim=1, index=entity_ids).reshape(btz, entity_count, -1, hidden_size)
        entity_states = torch.mean(entity_states, dim=2)  # 取实体首尾hidden_states的均值
        entity_logit = self.dense2(entity_states)  # [btz, 实体个数，实体类型数]

        return emission_score, attention_mask, entity_logit

    def predict(self, token_ids):
        self.eval()
        with torch.no_grad():
            # 一阶段推理
            last_hidden_state = self.bert([token_ids])  # [btz, seq_len, hdsz]
            emission_score = self.dense1(last_hidden_state)  # [bts, seq_len, tag_size]
            attention_mask = token_ids.gt(0)
            best_path = self.crf.decode(emission_score, attention_mask)  # [bts, seq_len]

            # 二阶段推理
            batch_entity_ids = []
            for one_samp in best_path:
                entity_ids = []
                for j, item in enumerate(one_samp):
                    if item.item() == 1:  # B
                        entity_ids.append([j, j])
                    elif len(entity_ids) == 0:
                        continue
                    elif (len(entity_ids[-1]) > 0) and (item.item() == 2):  # I
                        entity_ids[-1][-1] = j
                    elif len(entity_ids[-1]) > 0:
                        entity_ids.append([])
                if not entity_ids:  # 至少要有一个标签
                    entity_ids.append([0, 0])  # 如果没有则用0填充
                batch_entity_ids.append([i for i in entity_ids if i])
            batch_entity_ids = torch.tensor(sequence_padding(batch_entity_ids), dtype=torch.long, device=device)  # [btz, 实体个数，start/end]
            
            btz, entity_count, _ = batch_entity_ids.shape
            hidden_size = last_hidden_state.shape[-1]
            gather_index = batch_entity_ids.reshape(btz, -1, 1).repeat(1, 1, hidden_size)
            entity_states = torch.gather(last_hidden_state, dim=1, index=gather_index).reshape(btz, entity_count, -1, hidden_size)
            entity_states = torch.mean(entity_states, dim=2)  # 取实体首尾hidden_states的均值
            entity_logit = self.dense2(entity_states)  # [btz, 实体个数，实体类型数]
            entity_pred = torch.argmax(entity_logit, dim=-1)  # [btz, 实体个数]

            # 每个元素为一个三元组
            entity_tulpe = trans_entity2tuple(batch_entity_ids, entity_pred)
        return best_path, entity_tulpe

model = Model().to(device)

class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss2 = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, outputs, labels):
        emission_score, attention_mask, entity_logit = outputs
        seq_labels, entity_labels = labels
        loss1 = model.crf(emission_score, attention_mask, seq_labels)
        loss2 = self.loss2(entity_logit.reshape(-1, entity_logit.shape[-1]), entity_labels.flatten())
        return {'loss': loss1+loss2, 'loss1': loss1, 'loss2': loss2}

# Loss返回的key会自动计入metrics，下述metrics不写仍可以打印loss1和loss2
model.compile(loss=Loss(), optimizer=optim.Adam(model.parameters(), lr=2e-5))

def evaluate(data):
    X1, Y1, Z1 = 1e-10, 1e-10, 1e-10
    X2, Y2, Z2 = 1e-10, 1e-10, 1e-10
    for (token_ids, entity_ids), (label, entity_labels) in tqdm(data):
        scores, entity_pred = model.predict(token_ids)  # [btz, seq_len]
        # 一阶段指标: token粒度
        attention_mask = label.gt(0)
        X1 += (scores.eq(label) * attention_mask).sum().item()
        Y1 += scores.gt(0).sum().item()
        Z1 += label.gt(0).sum().item()

        # 二阶段指标：entity粒度
        entity_true = trans_entity2tuple(entity_ids, entity_labels)
        X2 += len(entity_pred.intersection(entity_true))
        Y2 += len(entity_pred)
        Z2 += len(entity_true)

    f1, precision, recall = 2 * X1 / (Y1 + Z1), X1 / Y1, X1 / Z1
    f2, precision2, recall2 = 2 * X2 / (Y2 + Z2), X2/ Y2, X2 / Z2
    return f1, precision, recall, f2, precision2, recall2

def trans_entity2tuple(entity_ids, entity_labels):
    '''把tensor转为(样本id, start, end, 实体类型)的tuple用于计算指标
    '''
    entity_true = set()
    for i, one_sample in enumerate(entity_ids):
        for j, item in enumerate(one_sample):
            if item[0].item() * item[1].item() != 0:
                entity_true.add((i, item[0].item(), item[1].item(), entity_labels[i, j].item()))
    return entity_true

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        f1, precision, recall, f2, precision2, recall2 = evaluate(valid_dataloader)
        if f2 > self.best_val_f1:
            self.best_val_f1 = f2
            # model.save_weights('best_model.pt')
        print(f'[val-1阶段] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f}')
        print(f'[val-2阶段] f1: {f2:.5f}, p: {precision2:.5f} r: {recall2:.5f} best_f1: {self.best_val_f1:.5f}\n')


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(train_dataloader, epochs=20, steps_per_epoch=None, callbacks=[evaluator])

else: 

    model.load_weights('best_model.pt')


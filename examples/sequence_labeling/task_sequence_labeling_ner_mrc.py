#! -*- coding:utf-8 -*-
# mrc阅读理解方案
# 数据集：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# [valid_f1]: 95.75

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from bert4torch.snippets import sequence_padding, Callback, ListDataset, seed_everything
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from tqdm import tqdm
from collections import defaultdict

max_c_len = 224
max_q_len = 32
batch_size = 6  # 真实的batch_size是 batch_size * 实体类型数
categories = ['LOC', 'PER', 'ORG']
ent2query = {"LOC": "找出下述句子中的地址名",
             "PER": "找出下述句子中的人名",
             "ORG": "找出下述句子中的机构名"}

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
    batch_token_ids, batch_segment_ids, batch_start_labels, batch_end_labels = [], [], [], []
    batch_ent_type = []
    for d in batch:
        tokens_b = tokenizer.tokenize(d[0], maxlen=max_c_len)[1:]  # 不保留[CLS]
        mapping = tokenizer.rematch(d[0], tokens_b)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}

        # 按照实体类型整理实体
        label_dict = defaultdict(list)
        for start, end, label in d[1:]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                label_dict[label].append((start, end))

        # 遍历实体类型，query为tokens_a, context为tokens_b
        # 样本组成：[CLS] + tokens_a + [SEP] + tokens_b + [SEP]
        for _type in categories:
            start_ids = [0] * len(tokens_b)
            end_ids = [0] * len(tokens_b)

            text_a = ent2query[_type]
            tokens_a = tokenizer.tokenize(text_a, maxlen=max_q_len)

            for _label in label_dict[_type]:
                start_ids[_label[0]] = 1
                end_ids[_label[1]] = 1

            start_ids = [0] * len(tokens_a) + start_ids
            end_ids = [0] * len(tokens_a) + end_ids
            token_ids = tokenizer.tokens_to_ids(tokens_a) + tokenizer.tokens_to_ids(tokens_b)
            segment_ids = [0] * len(tokens_a) + [1] * len(tokens_b)
            assert len(start_ids) == len(end_ids) == len(token_ids) == len(segment_ids)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_start_labels.append(start_ids)
            batch_end_labels.append(end_ids)
            batch_ent_type.append(_type)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_start_labels = torch.tensor(sequence_padding(batch_start_labels), dtype=torch.long, device=device)
    batch_end_labels = torch.tensor(sequence_padding(batch_end_labels), dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], [batch_segment_ids, batch_start_labels, batch_end_labels, batch_ent_type]

# 转换数据集
train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/ner/china-people-daily-ner-corpus/example.train'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/ner/china-people-daily-ner-corpus/example.dev'), batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path)
        self.mid_linear = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.start_fc = nn.Linear(128, 2)
        self.end_fc = nn.Linear(128, 2)

    def forward(self, token_ids, segment_ids):
        sequence_output = self.bert([token_ids, segment_ids])  # [bts, seq_len, hdsz]
        seq_out = self.mid_linear(sequence_output)  # [bts, seq_len, mid_dims]
        start_logits = self.start_fc(seq_out)  # [bts, seq_len, 2]
        end_logits = self.end_fc(seq_out)  # [bts, seq_len, 2]

        return start_logits, end_logits


model = Model().to(device)

class Loss(nn.CrossEntropyLoss):
    def forward(self, outputs, labels):
        start_logits, end_logits = outputs
        mask, start_ids, end_ids = labels[:3]
        start_logits = start_logits.view(-1, 2)
        end_logits = end_logits.view(-1, 2)

        # 去掉 text_a 和 padding 部分的标签，计算真实 loss
        active_loss = mask.view(-1) == 1
        active_start_logits = start_logits[active_loss]
        active_end_logits = end_logits[active_loss]
        active_start_labels = start_ids.view(-1)[active_loss]
        active_end_labels = end_ids.view(-1)[active_loss]

        start_loss = super().forward(active_start_logits, active_start_labels)
        end_loss = super().forward(active_end_logits, active_end_labels)
        return start_loss + end_loss

model.compile(loss=Loss(), optimizer=optim.Adam(model.parameters(), lr=2e-5))


def evaluate(data):
    X, Y, Z = 0, 1e-10, 1e-10
    for (token_ids, segment_ids), labels in tqdm(data, desc='Evaluation'):
        start_logit, end_logit = model.predict([token_ids, segment_ids])  # [btz, seq_len, 2]
        mask, start_ids, end_ids, ent_type = labels

        # entity粒度
        entity_pred = mrc_decode(start_logit, end_logit, ent_type, mask)
        entity_true = mrc_decode(start_ids, end_ids, ent_type)

        X += len(entity_pred.intersection(entity_true))
        Y += len(entity_pred)
        Z += len(entity_true)
    f1, precision, recall = 2 * X / (Y + Z), X/ Y, X / Z
    return f1, precision, recall


# 严格解码 baseline
def mrc_decode(start_preds, end_preds, ent_type, mask=None):
    '''返回实体的start, end
    '''
    predict_entities = set()
    if mask is not None: # 预测的把query和padding部分mask掉
        start_preds = torch.argmax(start_preds, -1) * mask
        end_preds = torch.argmax(end_preds, -1) * mask

    start_preds = start_preds.cpu().numpy()
    end_preds = end_preds.cpu().numpy()

    for bt_i in range(start_preds.shape[0]):
        start_pred = start_preds[bt_i]
        end_pred = end_preds[bt_i]
        # 统计每个样本的结果
        for i, s_type in enumerate(start_pred):
            if s_type == 0:
                continue
            for j, e_type in enumerate(end_pred[i:]):
                if s_type == e_type:
                    # [样本id, 实体起点，实体终点，实体类型]
                    predict_entities.add((bt_i, i, i+j, ent_type[bt_i]))
                    break
    return predict_entities


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        f1, precision, recall = evaluate(valid_dataloader)
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            # model.save_weights('best_model.pt')
        print(f'[val] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f} best_f1: {self.best_val_f1:.5f}')


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(train_dataloader, epochs=20, steps_per_epoch=None, callbacks=[evaluator])

else: 

    model.load_weights('best_model.pt')

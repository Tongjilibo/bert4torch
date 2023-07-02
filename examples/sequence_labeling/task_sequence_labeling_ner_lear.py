#! -*- coding:utf-8 -*-
'''
数据集：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
基于`BERT`的`LEAR`实体识别模型
    模型的整体思路将实体识别问题转化为每个实体类型下的`span`预测问题
    模型的输入分为两个部分：原始的待抽取文本和所有标签对应的文本描述（先验知识）
    原始文本和标签描述文本共享`BERT`的编码器权重
    采用注意力机制融合标签信息到`token`特征中去
Reference:
    [Enhanced Language Representation with Label Knowledge for Span Extraction.](https://aclanthology.org/2021.emnlp-main.379.pdf)
    [Code](https://github.com/Akeepers/LEAR)
'''

import math
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, ListDataset, seed_everything
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from tqdm import tqdm
import numpy as np

max_c_len = 224
max_q_len = 32
batch_size = 6
nested = False
categories = {'LOC': 0, 'PER': 1, 'ORG': 2}
num_labels = len(categories)
categories_annotations = {"LOC": "找出下述句子中的地址名",
                          "PER": "找出下述句子中的人名",
                          "ORG": "找出下述句子中的机构名"}

# BERT base
config_path = 'E:/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'E:/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'E:/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
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
    batch_token_ids, batch_start_labels, batch_end_labels = [], [], []
    for d in batch:
        tokens = tokenizer.tokenize(d[0], maxlen=max_c_len)
        mapping = tokenizer.rematch(d[0], tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}

        # 按照实体类型整理实体
        start_labels = np.zeros((len(tokens), num_labels))
        end_labels = np.zeros((len(tokens), num_labels))
        for start, end, label in d[1:]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                start_labels[start, categories[label]] = 1
                end_labels[end, categories[label]] = 1

        batch_token_ids.append(tokenizer.tokens_to_ids(tokens))
        batch_start_labels.append(start_labels)
        batch_end_labels.append(end_labels)

    batch_label_token_ids = tokenizer.encode(categories_annotations.values(), maxlen=max_q_len)[0]

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)  # [btz, seg_len]
    batch_start_labels = torch.tensor(sequence_padding(batch_start_labels), dtype=torch.long, device=device)  # [btz, seg_len]
    batch_end_labels = torch.tensor(sequence_padding(batch_end_labels), dtype=torch.long, device=device)  # [btz, seg_len]
    batch_label_token_ids = torch.tensor(sequence_padding(batch_label_token_ids), dtype=torch.long, device=device)  # [c, label_len]
    batch_span_labels = None
    masks = (batch_token_ids != tokenizer._token_pad_id).long()
    return [batch_token_ids, batch_label_token_ids], [masks, batch_start_labels, batch_end_labels, batch_span_labels]

# 转换数据集
train_dataloader = DataLoader(MyDataset('E:/data/corpus/ner/china-people-daily-ner-corpus/example.train'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset('E:/data/corpus/ner/china-people-daily-ner-corpus/example.dev'), batch_size=batch_size, collate_fn=collate_fn) 

class LabelFusionForToken(nn.Module):
    def __init__(self, hidden_size):
        super(LabelFusionForToken, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, token_features, label_features, label_mask=None):
        bs, seqlen = token_features.shape[:2]
        token_features = self.linear1(token_features)  # [bts, seq_len, hdsz]
        label_features = self.linear2(label_features)  # [c, label_len, hdsz]

        # 计算注意力得分
        scores = torch.einsum('bmh, cnh->bmcn', token_features, label_features)
        scores += (1.0 - label_mask[None, None, ...]) * -10000.0
        scores = torch.softmax(scores, dim=-1)

        # 加权标签嵌入
        weighted_label_features = label_features[None, None, ...].repeat(bs, seqlen, 1, 1, 1) * scores[..., None]
        fused_features = token_features.unsqueeze(2) + weighted_label_features.sum(-2)
        return torch.tanh(self.output(fused_features))

class Classifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(Classifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_labels, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(num_labels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        x = torch.mul(input, self.weight)
        x = torch.sum(x, -1)
        return x + self.bias

class MLPForMultiLabel(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_rate=0.2):
        super(MLPForMultiLabel, self).__init__()
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = Classifier(hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, features):
        features = self.classifier1(features)
        features = self.dropout(F.gelu(features))
        return self.classifier2(features)

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0)
        hidden_size = self.bert.configs['hidden_size']
        self.label_fusion_layer = LabelFusionForToken(hidden_size)
        self.start_fc = Classifier(hidden_size, num_labels)
        self.end_fc = Classifier(hidden_size, num_labels)
        if nested:
            self.span_layer = MLPForMultiLabel(hidden_size * 2, num_labels)

    def forward(self, token_ids, label_token_ids):
        token_features = self.bert([token_ids])  # [bts, seq_len, hdsz]
        label_features = self.bert([label_token_ids])  # [c, label_len, hdsz]

        label_mask = (label_token_ids != tokenizer._token_pad_id).long()
        fused_features = self.label_fusion_layer(token_features, label_features, label_mask)
        start_logits = self.start_fc(fused_features)  # [bts, seq_len, num_labels]
        end_logits = self.end_fc(fused_features)  # [bts, seq_len, num_labels]

        span_logits = None
        if nested:
            seqlen = token_ids.shape[1]
            start_extend = fused_features.unsqueeze(2).expand(-1, -1, seqlen, -1, -1)
            end_extend = fused_features.unsqueeze(1).expand(-1, seqlen, -1, -1, -1)
            span_matrix = torch.cat((start_extend, end_extend), dim=-1)
            span_logits = self.span_layer(span_matrix)

        return start_logits, end_logits, span_logits

model = Model().to(device)

class SpanLossForMultiLabelLoss(nn.Module):
    def __init__(self, name='Span Binary Cross Entropy Loss'):
        super().__init__()
        self.name = name
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, target):
        if not nested:
            return self.flated_forward(preds, target)
        start_logits, end_logits, span_logits = preds
        masks, start_labels, end_labels, span_labels = target
        start_, end_ = start_logits > 0, end_logits > 0

        bs, seqlen, num_labels = start_logits.shape
        span_candidate = torch.logical_or(
            (start_.unsqueeze(-2).expand(-1, -1, seqlen, -1) & end_.unsqueeze(-3).expand(-1, seqlen, -1, -1)),
            (start_labels.unsqueeze(-2).expand(-1, -1, seqlen, -1).bool() & end_labels.unsqueeze(-3).expand(-1, seqlen, -1, -1).bool())
        )

        masks = masks[:, :, None].expand(-1, -1, num_labels)
        start_loss = self.loss_fct(start_logits.view(-1), start_labels.view(-1).float())
        start_loss = (start_loss * masks.reshape(-1)).view(-1, num_labels).sum(-1).sum() / (masks.sum() / num_labels)

        end_loss = self.loss_fct(end_logits.view(-1), end_labels.view(-1).float())
        end_loss = (end_loss * masks.reshape(-1)).view(-1, num_labels).sum(-1).sum() / (masks.sum() / num_labels)

        span_masks = masks.bool().unsqueeze(2).expand(-1, -1, seqlen, -1) & masks.bool().unsqueeze(1).expand(-1, seqlen,
                                                                                                             -1, -1)
        span_masks = torch.triu(span_masks.permute(0, 3, 1, 2), 0).permute(0, 2, 3,
                                                                           1) * span_candidate  # start should be less equal to end
        span_loss = self.loss_fct(span_logits.view(bs, -1), span_labels.view(bs, -1).float())
        span_loss = span_loss.reshape(-1, num_labels).sum(-1).sum() / (span_masks.view(-1, num_labels).sum() / num_labels)

        return start_loss + end_loss + span_loss

    def flated_forward(self, preds, target):
        start_logits, end_logits, _ = preds
        masks, start_labels, end_labels, _ = target
        active_loss = masks.view(-1) == 1

        active_start_logits = start_logits.view(-1, start_logits.size(-1))[active_loss]
        active_end_logits = end_logits.view(-1, start_logits.size(-1))[active_loss]

        active_start_labels = start_labels.view(-1, start_labels.size(-1))[active_loss].float()
        active_end_labels = end_labels.view(-1, end_labels.size(-1))[active_loss].float()

        start_loss = self.loss_fct(active_start_logits, active_start_labels).sum(1).mean()
        end_loss = self.loss_fct(active_end_logits, active_end_labels).sum(1).mean()
        return start_loss + end_loss

model.compile(loss=SpanLossForMultiLabelLoss(), optimizer=optim.Adam(model.parameters(), lr=2e-5))

def evaluate(data):
    X, Y, Z = 0, 1e-10, 1e-10
    for inputs, labels in tqdm(data, desc='Evaluation'):
        start_logit, end_logit, span_logits = model.predict(inputs)
        mask, start_labels, end_labels, span_labels = labels

        # entity粒度
        entity_pred = decode(start_logit, end_logit, mask)
        entity_true = decode(start_labels, end_labels)

        X += len(entity_pred.intersection(entity_true))
        Y += len(entity_pred)
        Z += len(entity_true)
    f1, precision, recall = 2 * X / (Y + Z), X/ Y, X / Z
    return f1, precision, recall

# 解码
def decode(start_logit, end_logit, mask=None, start_thresh=0.5, end_thresh=0.5):
    '''返回实体的start, end
    '''
    if not nested:
        predict_entities = set()
        if mask is not None: # 预测的把query和padding部分mask掉
            start_logit = start_logit * mask.unsqueeze(-1)
            end_logit = end_logit * mask.unsqueeze(-1)
            start_preds, end_preds = torch.sigmoid(start_logit), torch.sigmoid(end_logit)
        else:
            start_preds, end_preds = start_logit, end_logit
        start_preds, end_preds = torch.where(start_preds > start_thresh), torch.where(end_preds > end_thresh)

        for bt_i, start_i, label_i in zip(*start_preds):
            for bt_j, end_j, label_j in zip(*end_preds):
                if (bt_i == bt_j) and (start_i <= end_j) and (label_i==label_j):
                    # [样本id, 实体起点，实体终点，实体类型]
                    predict_entities.add((bt_i.item(), start_i.item(), end_j.item(), label_i.item()))
    else:
        raise NotImplementedError
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

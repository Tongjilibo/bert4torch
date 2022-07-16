#! -*- coding:utf-8 -*-
# bert+crf用来做实体识别, 测试两种方案，一种是用数据集来生成crf权重，第二种是来初始化
# 数据集：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# 初始化： [valid_f1]  token_level: 97.35； entity_level: 96.42
# 固定化： [valid_f1]  token_level: 96.92； entity_level: 95.42


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
categories = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']
categories_id2label = {i: k for i, k in enumerate(categories)}
categories_label2id = {k: i for i, k in enumerate(categories)}

# BERT base
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定seed
seed_everything(42)

# 加载数据集
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
    batch_token_ids, batch_labels = [], []
    for d in batch:
        tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
        mapping = tokenizer.rematch(d[0], tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)
        labels = np.zeros(len(token_ids))
        for start, end, label in d[1:]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                labels[start] = categories_label2id['B-'+label]
                labels[start + 1:end + 1] = categories_label2id['I-'+label]
        batch_token_ids.append(token_ids)
        batch_labels.append(labels)
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels), dtype=torch.long, device=device)
    return batch_token_ids, batch_labels

# 转换数据集
train_data = load_data('F:/Projects/data/corpus/ner/china-people-daily-ner-corpus/example.train')
valid_data = load_data('F:/Projects/data/corpus/ner/china-people-daily-ner-corpus/example.dev')
train_dataloader = DataLoader(ListDataset(data=train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(ListDataset(data=valid_data), batch_size=batch_size, collate_fn=collate_fn) 

# 根据训练数据生成权重
transition = np.zeros((len(categories), len(categories)))
start_transition = np.zeros(len(categories))
end_transition = np.zeros(len(categories))
for d in tqdm(train_data, desc='Generate init_trasitions'):
    tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
    mapping = tokenizer.rematch(d[0], tokens)
    start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
    end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
    token_ids = tokenizer.tokens_to_ids(tokens)
    labels = np.zeros(len(token_ids))
    for start, end, label in d[1:]:
        if start in start_mapping and end in end_mapping:
            start = start_mapping[start]
            end = end_mapping[end]
            labels[start] = categories_label2id['B-'+label]
            labels[start + 1:end + 1] = categories_label2id['I-'+label]
    for i in range(len(labels)-1):
        transition[int(labels[i]), int(labels[i+1])] += 1
    start_transition[int(labels[0])] += 1  # start转移到标签
    end_transition[int(labels[-1])] += 1  # 标签转移到end
max_v = np.max([np.max(transition), np.max(start_transition), np.max(end_transition)])
min_v = np.min([np.min(transition), np.min(start_transition), np.min(end_transition)])
transition = (transition - min_v) / (max_v - min_v)
start_transition = (start_transition - min_v) / (max_v - min_v)
end_transition = (end_transition - min_v) / (max_v - min_v)

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0)
        self.fc = nn.Linear(768, len(categories))
        self.crf = CRF(len(categories), init_transitions=[transition, start_transition, end_transition], freeze=True)  # 控制是否初始化，是否参加训练

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
        emission_score = self.fc(sequence_output)  # [btz, seq_len, tag_size]
        attention_mask = token_ids.gt(0).long()
        return emission_score, attention_mask

    def predict(self, token_ids):
        self.eval()
        with torch.no_grad():
            emission_score, attention_mask = self.forward(token_ids)
            best_path = self.crf.decode(emission_score, attention_mask)  # [btz, seq_len]
        return best_path

model = Model().to(device)

class Loss(nn.Module):
    def forward(self, outputs, labels):
        return model.crf(*outputs, labels)

model.compile(loss=Loss(), optimizer=optim.Adam(model.parameters(), lr=2e-5))


def evaluate(data):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    X2, Y2, Z2 = 1e-10, 1e-10, 1e-10
    for token_ids, label in tqdm(data):
        scores = model.predict(token_ids)  # [btz, seq_len]
        attention_mask = label.gt(0)

        # token粒度
        X += (scores.eq(label) * attention_mask).sum().item()
        Y += scores.gt(0).sum().item()
        Z += label.gt(0).sum().item()

        # entity粒度
        entity_pred = trans_entity2tuple(scores)
        entity_true = trans_entity2tuple(label)
        X2 += len(entity_pred.intersection(entity_true))
        Y2 += len(entity_pred)
        Z2 += len(entity_true)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    f2, precision2, recall2 = 2 * X2 / (Y2 + Z2), X2/ Y2, X2 / Z2
    return f1, precision, recall, f2, precision2, recall2


def trans_entity2tuple(scores):
    '''把tensor转为(样本id, start, end, 实体类型)的tuple用于计算指标
    '''
    batch_entity_ids = set()
    for i, one_samp in enumerate(scores):
        entity_ids = []
        for j, item in enumerate(one_samp):
            flag_tag = categories_id2label[item.item()]
            if flag_tag.startswith('B-'):  # B
                entity_ids.append([i, j, j, flag_tag[2:]])
            elif len(entity_ids) == 0:
                continue
            elif (len(entity_ids[-1]) > 0) and flag_tag.startswith('I-') and (flag_tag[2:]==entity_ids[-1][-1]):  # I
                entity_ids[-1][-2] = j
            elif len(entity_ids[-1]) > 0:
                entity_ids.append([])

        for i in entity_ids:
            if i:
                batch_entity_ids.add(tuple(i))
    return batch_entity_ids

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
        print(f'[val-token  level] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f}')
        print(f'[val-entity level] f1: {f2:.5f}, p: {precision2:.5f} r: {recall2:.5f} best_f1: {self.best_val_f1:.5f}\n')


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(train_dataloader, epochs=20, steps_per_epoch=None, callbacks=[evaluator])

else: 

    model.load_weights('best_model.pt')

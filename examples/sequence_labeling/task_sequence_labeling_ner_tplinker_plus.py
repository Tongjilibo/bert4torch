#! -*- coding:utf-8 -*-
# tplinker_plus用来做实体识别
# [valid_f1]: 95.71

import numpy as np
from bert4torch.models import build_transformer_model, BaseModel
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from bert4torch.snippets import sequence_padding, Callback, ListDataset, seed_everything
from bert4torch.tokenizers import Tokenizer
from bert4torch.losses import MultilabelCategoricalCrossentropy
from bert4torch.layers import TplinkerHandshakingKernel

maxlen = 64
batch_size = 16
categories_label2id = {"LOC": 0, "ORG": 1, "PER": 2}
categories_id2label = dict((value, key) for key,value in categories_label2id.items())

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
        data = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            for l in f.split('\n\n'):
                if not l:
                    continue
                text, label = '', []
                for i, c in enumerate(l.split('\n')):
                    char, flag = c.split(' ')
                    text += char
                    if flag[0] == 'B':
                        label.append([i, i, flag[2:]])
                    elif flag[0] == 'I':
                        label[-1][1] = i
                text_list = tokenizer.tokenize(text)[1:-1]  #不保留首位[CLS]和末位[SEP]
                tokens = [j for i in text_list for j in i][:maxlen]  # 以char为单位
                data.append((tokens, label))  # label为[[start, end, entity], ...]
        return data


def trans_ij2k(seq_len, i, j):
    '''把第i行，第j列转化成上三角flat后的序号
    '''
    if (i > seq_len - 1) or (j > seq_len - 1) or (i > j):
        return 0
    return int(0.5*(2*seq_len-i+1)*i+(j-i))
map_ij2k = {(i, j): trans_ij2k(maxlen, i, j) for i in range(maxlen) for j in range(maxlen) if j >= i}
map_k2ij = {v: k for k, v in map_ij2k.items()}

def tran_ent_rel2id():
    '''获取最后一个分类层的的映射关系
    '''
    tag2id = {}
    for p in categories_label2id.keys():
        tag2id[p] = len(tag2id)
    return tag2id
tag2id = tran_ent_rel2id()
id2tag = {v: k for k, v in tag2id.items()}


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def collate_fn(batch):
    pair_len = maxlen * (maxlen+1)//2
    # batch_head_labels: [btz, pair_len, tag2id_len]
    batch_labels = torch.zeros((len(batch), pair_len, len(tag2id)), dtype=torch.long, device=device)

    batch_token_ids = []
    for i, (tokens, labels) in enumerate(batch):
        batch_token_ids.append(tokenizer.tokens_to_ids(tokens))  # 前面已经限制了长度
        for s_i in labels:
            if s_i[1] >= len(tokens):  # 实体的结尾超过文本长度，则不标记
                continue
            batch_labels[i, map_ij2k[s_i[0], s_i[1]], tag2id[s_i[2]]] = 1
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, length=maxlen), dtype=torch.long, device=device)
    return [batch_token_ids], batch_labels

# 转换数据集
train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/ner/china-people-daily-ner-corpus/example.train'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/ner/china-people-daily-ner-corpus/example.dev'), batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0)
        self.fc = nn.Linear(768, len(tag2id))
        self.handshaking_kernel = TplinkerHandshakingKernel(768, shaking_type='cln_plus', inner_enc_type='lstm')

    def forward(self, inputs):
        last_hidden_state = self.bert(inputs)  # [btz, seq_len, hdsz]
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)
        output = self.fc(shaking_hiddens)  # [btz, pair_len, tag_size]
        return output
        
model = Model().to(device)
model.compile(loss=MultilabelCategoricalCrossentropy(), optimizer=optim.Adam(model.parameters(), lr=2e-5))

def evaluate(data, threshold=0):
    X, Y, Z, threshold = 0, 1e-10, 1e-10, 0
    for x_true, label in data:
        scores = model.predict(x_true)  # [btz, pair_len, tag_size]
        for i, score in enumerate(scores):
            R = set()
            for pair_id, tag_id in zip(*np.where(score.cpu().numpy() > threshold)):
                start, end = map_k2ij[pair_id][0], map_k2ij[pair_id][1]
                R.add((start, end, tag_id))  

            T = set()
            for pair_id, tag_id in zip(*np.where(label[i].cpu().numpy() > threshold)):
                start, end = map_k2ij[pair_id][0], map_k2ij[pair_id][1]
                T.add((start, end, tag_id))
            X += len(R & T)
            Y += len(R)
            Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


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

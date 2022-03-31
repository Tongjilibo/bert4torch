#! -*- coding:utf-8 -*-
# global_pointer用来做实体识别
# 数据集：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# 博客：https://kexue.fm/archives/8373

from turtle import forward
import numpy as np
from bert4torch.models import build_transformer_model, BaseModel
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from bert4torch.snippets import sequence_padding, Callback, ListDataset
from bert4torch.tokenizers import Tokenizer
from bert4torch.losses import MultilabelCategoricalCrossentropy
from bert4torch.layers import GlobalPointer

maxlen = 512
batch_size = 6
categories_label2id = {"LOC": 0, "ORG": 1, "PER": 2}
categories_id2label = dict((value, key) for key,value in categories_label2id.items())
ner_vocab_size = len(categories_label2id)
ner_head_size = 64

# BERT base
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                tokens = [j for i in text_list for j in i][:maxlen]
                data.append((tokens, label))  # label为[[start, end, entity], ...]
        return data


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def collate_fn(batch):
    batch_token_ids = []
    max_seq_len = min(max([len(tokens) for tokens, _ in batch]), maxlen)
    batch_labels = torch.zeros((len(batch), len(categories_label2id), max_seq_len, max_seq_len), device=device)
    for i, (tokens, labels) in enumerate(batch):
        batch_token_ids.append(tokenizer.tokens_to_ids(tokens))
        for s_i in labels:
            if s_i[1] >= max_seq_len:  # 实体的结尾超过文本长度，则不标记
                continue
            batch_labels[i, categories_label2id[s_i[-1]], s_i[0], s_i[1]] = 1
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    return batch_token_ids, batch_labels

# 转换数据集
train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/ner/china-people-daily-ner-corpus/example.train'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/ner/china-people-daily-ner-corpus/example.dev'), batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0)
        self.global_pointer = GlobalPointer(hidden_size=768, heads=ner_vocab_size, head_size=ner_head_size)

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
        logit = self.global_pointer(sequence_output, token_ids.gt(0).long())
        return logit
        
model = Model().to(device)

class MyLoss(MultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, y_pred, y_true):
        y_true = y_true.view(y_true.shape[0]*y_true.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        y_pred = y_pred.view(y_pred.shape[0]*y_pred.shape[1], -1)  # [btz*ner_vocab_size, seq_len*seq_len]
        return super().forward(y_pred, y_true)

model.compile(loss=MyLoss(), optimizer=optim.Adam(model.parameters(), lr=2e-5))

def evaluate(data, threshold=0.5):
    X, Y, Z, threshold = 1e-10, 1e-10, 1e-10, 0
    for x_true, label in data:
        scores = model.predict(x_true)
        for i, score in enumerate(scores):
            R = set()
            for l, start, end in zip(*np.where(score.cpu() > threshold)):
                R.add((start, end, categories_id2label[l]))  

            T = set()
            for l, start, end in zip(*np.where(label[i].cpu() > threshold)):
                T.add((start, end, categories_id2label[l]))
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

    model.fit(train_dataloader, epochs=50, steps_per_epoch=None, callbacks=[evaluator])

else: 

    model.load_weights('best_model.pt')

#! -*- coding: utf-8 -*-
# 用CRF做中文分词（CWS, Chinese Word Segmentation）
# 数据集 http://sighan.cs.uchicago.edu/bakeoff2005/

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, ListDataset, seed_everything
from bert4torch.layers import CRF
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from tqdm import tqdm
import re
import json
import os

maxlen = 256
batch_size = 32
num_labels = 4

# BERT base
config_path = 'E:/data/pretrain_ckpt/bert/google@chinese_L-12_H-768_A-12/bert4torch_config.json'
checkpoint_path = 'E:/data/pretrain_ckpt/bert/google@chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'E:/data/pretrain_ckpt/bert/google@chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定seed
seed_everything(42)

# 加载数据集

def load_data(filename):
    """加载数据
    单条格式：[词1, 词2, 词3, ...]
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            D.append(re.split(' +', l.strip()))
    return D

# 标注数据
data = load_data('F:/data/corpus/cws/icwb2-data/training/pku_training.utf8')

# 保存一个随机序（供划分valid用）
if not os.path.exists('./random_order.json'):
    random_order = list(range(len(data)))
    np.random.shuffle(random_order)
    json.dump(random_order, open('./random_order.json', 'w'), indent=4)
else:
    random_order = json.load(open('./random_order.json'))

# 划分valid
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def collate_fn(batch):
    """标签含义
    0: 单字词； 1: 多字词首字； 2: 多字词中间； 3: 多字词末字
    """
    batch_token_ids, batch_labels = [], []
    for item in batch:
        token_ids, labels = [tokenizer._token_start_id], [0]
        for w in item:
            w_token_ids = tokenizer.encode(w)[0][1:-1]
            if len(token_ids) + len(w_token_ids) < maxlen:
                token_ids += w_token_ids
                if len(w_token_ids) == 1:
                    labels += [0]
                else:
                    labels += [1] + [2] * (len(w_token_ids) - 2) + [3]
            else:
                break
        token_ids += [tokenizer._token_end_id]
        labels += [0]
        batch_token_ids.append(token_ids)
        batch_labels.append(labels)
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels), dtype=torch.long, device=device)
    return batch_token_ids, batch_labels

# 转换数据集
train_dataloader = DataLoader(ListDataset(data=train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0)
        self.fc = nn.Linear(768, num_labels)  # 包含首尾
        self.crf = CRF(num_labels)

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

def acc(y_pred, y_true):
    y_pred = y_pred[0]
    y_pred = torch.argmax(y_pred, dim=-1)
    acc = torch.sum(y_pred.eq(y_true)).item() / y_true.numel()
    return {'acc': acc}

# 支持多种自定义metrics = ['accuracy', acc, {acc: acc}]均可
model.compile(loss=Loss(), optimizer=optim.Adam(model.parameters(), lr=2e-5), metrics=acc)

def segmenter(text):
    tokens = tokenizer.tokenize(text)
    while len(tokens) > 512:
        tokens.pop(-2)
    mapping = tokenizer.rematch(text, tokens)
    token_ids = tokenizer.tokens_to_ids(tokens)
    token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    labels = model.predict(token_ids)[0].cpu().numpy()
    words = []
    for i, label in enumerate(labels[1:-1]):
        if label < 2 or len(words) == 0:
            words.append([i + 1])
        else:
            words[-1].append(i + 1)
    return [text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1] for w in words]

def simple_evaluate(data):
    """简单的评测
    该评测指标不等价于官方的评测指标，但基本呈正相关关系，
    可以用来快速筛选模型。
    """
    total, right = 0., 0.
    for w_true in tqdm(data):
        w_pred = segmenter(''.join(w_true))
        w_pred = set(w_pred)
        w_true = set(w_true)
        total += len(w_true)
        right += len(w_true & w_pred)
    return right / total


def predict_to_file(in_file, out_file):
    """预测结果到文件，便于用官方脚本评测
    使用示例：
    predict_to_file('/root/icwb2-data/testing/pku_test.utf8', 'myresult.txt')
    官方评测代码示例：
    data_dir="/root/icwb2-data"
    $data_dir/scripts/score $data_dir/gold/pku_training_words.utf8 $data_dir/gold/pku_test_gold.utf8 myresult.txt > myscore.txt
    （执行完毕后查看myscore.txt的内容末尾）
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file, encoding='utf-8') as fr:
        for l in tqdm(fr):
            l = l.strip()
            if l:
                l = ' '.join(segmenter(l))
            fw.write(l + '\n')
    fw.close()


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        acc = simple_evaluate(valid_data)
        # 保存最优
        if acc >= self.best_val_acc:
            self.best_val_acc = acc
            # model.save_weights('./best_model.pt')
        print('acc: %.5f, best acc: %.5f\n' % (acc, self.best_val_acc))


if __name__ == '__main__':

    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator])

else: 

    model.load_weights('best_model.pt')

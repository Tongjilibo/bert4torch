# 模型训练脚本
# 链接：https://github.com/kangyishuai/NEWS-TEXT-CLASSIFICATION
# 这里仅基于bert4torch实现了Top1解决方案中的finetune部分，直接使用了原作者的预训练权重转pytorch

import numpy as np
import pandas as pd
from bert4torch.models import build_transformer_model, BaseModel
from torch.utils.data import DataLoader
from bert4torch.snippets import sequence_padding, ListDataset, Callback
from bert4torch.tokenizers import Tokenizer
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn, optim

# BERT base
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = f'cuda' if torch.cuda.is_available() else 'cpu'

n = 5   # Cross-validation
SEED = 2020
num_classes = 14

maxlen = 512
max_segment = 2
batch_size = 4
grad_accum_steps = 64
drop = 0.2
lr = 2e-5
epochs = 100

def load_data(df):
    """加载数据。"""
    D = list()
    for _, row in df.iterrows():
        text = row['text']
        label = row['label']
        D.append((text, int(label)))
    return D


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def sentence_split(words):
    """句子截断。"""
    document_len = len(words)

    index = list(range(0, document_len, maxlen-2))
    index.append(document_len)

    segments = []
    for i in range(len(index) - 1):
        segment = words[index[i]: index[i + 1]]
        assert len(segment) > 0
        segment = tokenizer.tokens_to_ids(['[CLS]'] + segment + ['[SEP]'])
        segments.append(segment)

    assert len(segments) > 0
    if len(segments) > max_segment:
        segment_ = int(max_segment / 2)
        return segments[:segment_] + segments[-segment_:]
    else:
        return segments


def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for text, label in batch:
        token_ids = sentence_split(text)
        token_ids = sequence_padding(token_ids, length=maxlen)
        segment_ids = np.zeros_like(token_ids)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append([label])

    batch_token_ids = sequence_padding(batch_token_ids, length=max_segment)
    batch_segment_ids = sequence_padding(batch_segment_ids, length=max_segment)
    batch_labels = sequence_padding(batch_labels)
    return [batch_token_ids, batch_segment_ids], batch_labels


class Attention(nn.Module):
    """注意力层。"""
    def __init__(self, hidden_size, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)
        self.weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.query = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, x):
        x, mask = x
        mask = mask.squeeze(2)
        # linear
        key = self.weight(x) + self.bias

        # compute attention
        outputs = self.query(key).squeeze(2)
        outputs -= 1e32 * (1 - mask)

        attn_scores = F.softmax(outputs)
        attn_scores *= mask
        attn_scores = attn_scores.reshape(-1, 1, attn_scores.shape[-1])

        outputs = torch.matmul(attn_scores, key).squeeze(1)

        return outputs

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.attn = Attention(768)
        self.dense = nn.Linear(768, num_classes)

    def forward(self, inputs, input_mask):
        output = self.bert(inputs)[:, 0]  # [btz, seq_len, hdsz]
        output = output.reshape((-1, max_segment, output.shape[-1]))
        output = output * input_mask
        output = self.dropout1(output)
        output = self.attn([output, input_mask])
        output = self.dropout2(output)
        output = self.dense(output)
        return output


class Evaluator(Callback):
    def __init__(self, valid_generator):
        super().__init__()
        self.valid_generator = valid_generator
        self.best_val_f1 = 0.

    def evaluate(self):
        y_true, y_pred = list(), list()
        for x, y in self.valid_generator:
            y_true.append(y)
            y_pred.append(self.model.predict(x).argmax(axis=1))
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1

    def on_epoch_end(self, steps, epoch, logs=None):
        val_f1 = self.evaluate()
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
        logs['val_f1'] = val_f1
        print(f'val_f1: {val_f1:.5f}, best_val_f1: {self.best_val_f1:.5f}')


def do_train(df_train):
    skf = StratifiedKFold(n_splits=n, random_state=SEED, shuffle=True)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train['text'], df_train['label']), 1):
        print(f'Fold {fold}')

        train_data = load_data(df_train.iloc[trn_idx])
        valid_data = load_data(df_train.iloc[val_idx])

        train_dataloader = DataLoader(ListDataset(data=train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
        valid_dataloader = DataLoader(ListDataset(data=valid_data), batch_size=batch_size, collate_fn=collate_fn) 

        model = Model().to(device)
        model.compile(loss=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=1e-5), adversarial_train={'name': 'fgm'})

        model.fit(
            train_dataloader,
            steps_per_epoch=None,
            epochs=epochs,
            grad_accumulation_steps=grad_accum_steps,
            callbacks=[Evaluator(valid_dataloader)]
        )

        del model

if __name__ == '__main__':
    df_train = pd.read_csv('E:/Github/天池新闻分类/data/train_set.csv', sep='\t')
    df_train['text'] = df_train['text'].apply(lambda x: x.strip().split())
    do_train(df_train)

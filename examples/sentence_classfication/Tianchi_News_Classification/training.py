# 模型训练脚本
# 链接：https://github.com/kangyishuai/NEWS-TEXT-CLASSIFICATION
# 这里仅基于bert4torch实现了Top1解决方案中的finetune部分，直接使用了原作者的预训练权重转pytorch

import numpy as np
import pandas as pd
from bert4torch.models import build_transformer_model, BaseModel
from torch.utils.data import DataLoader
from bert4torch.snippets import sequence_padding, ListDataset, Callback, EarlyStopping, AdversarialTraining
from bert4torch.tokenizers import Tokenizer
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn, optim
from tqdm import tqdm

# BERT base
config_path = 'E:/Github/天池新闻分类/top1/pre_models/bert_config.json'
checkpoint_path = 'E:/Github/天池新闻分类/top1/pre_models/pytorch_model.bin'
dict_path = 'E:/Github/天池新闻分类/top1/pre_models/vocab.txt'
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
    batch_token_ids, batch_labels = [], []
    for text, label in batch:
        token_ids = sentence_split(text)
        token_ids = sequence_padding(token_ids, length=maxlen)
        batch_token_ids.append(token_ids)
        batch_labels.append(label)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, length=max_segment), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, device=device)
    return batch_token_ids, batch_labels


class Attention(nn.Module):
    """注意力层。"""
    def __init__(self, hidden_size, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)
        self.weight = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.hidden_size))
        self.query = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, x, mask):
        '''x: [btz, max_segment, hdsz]
        mask: [btz, max_segment, 1]
        '''
        mask = mask.squeeze(2)  # [btz, max_segment]

        # linear
        key = self.weight(x) + self.bias  # [btz, max_segment, hdsz]

        # compute attention
        outputs = self.query(key).squeeze(2)  # [btz, max_segment]
        outputs -= 1e32 * (1 - mask)
        attn_scores = F.softmax(outputs, dim=-1)
        attn_scores = attn_scores * mask
        attn_scores = attn_scores.reshape(-1, 1, attn_scores.shape[-1])  # [btz, 1, max_segment]

        outputs = torch.matmul(attn_scores, key).squeeze(1)  # [btz, hdsz]
        return outputs

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.attn = Attention(768)
        self.dense = nn.Linear(768, num_classes)

    def forward(self, token_ids):
        ''' token_ids: [btz, max_segment, max_len]
        '''
        input_mask = torch.any(token_ids, dim=-1, keepdim=True).long()  # [btz, max_segment, 1]
        token_ids = token_ids.reshape(-1, token_ids.shape[-1])  # [btz*max_segment, max_len]

        output = self.bert([token_ids])[:, 0]  # [btz*max_segment, hdsz]
        output = output.reshape((-1, max_segment, output.shape[-1]))  # [btz, max_segment, hdsz]
        output = output * input_mask
        output = self.dropout1(output)
        output = self.attn(output, input_mask)
        output = self.dropout2(output)
        output = self.dense(output)
        return output


class Evaluator(Callback):
    def __init__(self, model, dataloader, fold):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.best_val_f1 = 0.
        self.fold = fold

    def evaluate(self):
        y_true, y_pred = list(), list()
        for x, y in tqdm(self.dataloader, desc='evaluate'):
            y_true.append(y.cpu().numpy())
            y_pred.append(self.model.predict(x).argmax(axis=1).cpu().numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1

    def on_epoch_end(self, steps, epoch, logs=None):
        val_f1 = self.evaluate()
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.model.save_weights(f'best_model_fold{self.fold}.pt')
        logs['val_f1'] = val_f1  # 这个要设置，否则EarlyStopping不生效
        print(f'val_f1: {val_f1:.5f}, best_val_f1: {self.best_val_f1:.5f}\n')


def do_train(df_train):
    skf = StratifiedKFold(n_splits=n, random_state=SEED, shuffle=True)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(df_train['text'], df_train['label']), 1):
        print(f'[Fold {fold}]')

        train_data = load_data(df_train.iloc[trn_idx])
        valid_data = load_data(df_train.iloc[val_idx])

        train_dataloader = DataLoader(ListDataset(data=train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
        valid_dataloader = DataLoader(ListDataset(data=valid_data), batch_size=batch_size, collate_fn=collate_fn) 

        model = Model().to(device)
        model.compile(loss=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=lr), 
                      grad_accumulation_steps=grad_accum_steps)

        callbacks = [
            AdversarialTraining('fgm'),
            Evaluator(model, valid_dataloader, fold),
            EarlyStopping(monitor='val_f1', patience=5, verbose=1, mode='max'), # 需要在Evaluator后面
        ]
        model.fit(
            train_dataloader,
            steps_per_epoch=None,
            epochs=epochs,
            callbacks=callbacks
        )

        del model


if __name__ == '__main__':
    df_train = pd.read_csv('E:/Github/天池新闻分类/data/train_set.csv', sep='\t')
    df_train['text'] = df_train['text'].apply(lambda x: x.strip().split())
    do_train(df_train)

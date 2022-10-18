#! -*- coding: utf-8 -*-
# SimBERT_v2监督训练代码supervised部分
# 官方项目：https://github.com/ZhuiyiTechnology/roformer-sim

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, ListDataset, text_segmentate
from bert4torch.snippets import Callback, truncate_sequences, get_pool_emb
from bert4torch.tokenizers import Tokenizer
import json
import glob

# 基本信息
maxlen = 64
batch_size = 12
labels = ['contradiction', 'entailment', 'neutral']

# bert配置，需要加载stage2训练后的权重，这里直接加载官方最终的权重以示例
config_path = 'F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--roformer_chinese_sim_char_base/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--roformer_chinese_sim_char_base/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--roformer_chinese_sim_char_base/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def split(text):
    """分割句子
    """
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    return text_segmentate(text, maxlen * 1.2, seps, strips)

class MyDataset(ListDataset):
    def load_data(self, file_path):
        dataset1_path, dataset2_path = file_path
        D1 = self.load_data_1(dataset1_path)
        D2 = self.load_data_2(dataset2_path)
        return D1 + D2

    @staticmethod
    def load_data_1(filenames, threshold=0.5):
        """加载数据（带标签）
        单条格式：(文本1, 文本2, 标签)
        """
        D = []
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    l = l.strip().split('\t')
                    if len(l) != 3:
                        continue
                    l[0], l[1] = split(l[0])[0], split(l[1])[0]
                    D.append((l[0], l[1], int(float(l[2]) > threshold)))
        return D

    @staticmethod
    def load_data_2(dir_path):
        """加载数据（带标签）
        单条格式：(文本1, 文本2, 标签)
        """
        D = []
        for filename in glob.glob(dir_path):
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    l = json.loads(l)
                    if l['gold_label'] not in labels:
                        continue
                    text1 = split(l['sentence1'])[0]
                    text2 = split(l['sentence2'])[0]
                    label = labels.index(l['gold_label']) + 2
                    D.append((text1, text2, label))
        return D


def truncate(text):
    """截断句子
    """
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    return text_segmentate(text, maxlen - 2, seps, strips)[0]

def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for text1, text2, label in batch:
        for text in [text1, text2]:
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
        batch_labels.append([label])
    
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], batch_labels

# 加载数据集
data_path = 'F:/Projects/data/corpus/sentence_embedding/'
dataset1_path = []
for task_name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B']:
    for f in ['train', 'valid']:
        threshold = 2.5 if task_name == 'STS-B' else 0.5
        filename = '%s%s/%s.%s.data' % (data_path, task_name, task_name, f)
        dataset1_path.append(filename)
dataset2_path = 'F:/Projects/data/corpus/sentence_embedding/XNLI-MT-1.0/cnsd/cnsd-*/*.jsonl'
train_dataloader = DataLoader(MyDataset([dataset1_path, dataset2_path]), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

# 建立加载模型
class Model(BaseModel):
    def __init__(self, pool_method='cls'):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='roformer', 
                                            with_pool='linear', dropout_rate=0.2)
        self.pool_method = pool_method
        self.dense = nn.Linear(768*3, 5, bias=False)

    def forward(self, token_ids, segment_ids):
        hidden_state, pool_cls = self.bert([token_ids, segment_ids])
        sen_emb = get_pool_emb(hidden_state, pool_cls, token_ids.gt(0).long(), self.pool_method)  # [btz*2, hdsz]
        # 向量合并：a、b、|a-b|拼接
        u, v = sen_emb[::2], sen_emb[1::2]
        sen_emb_concat = torch.cat([u, v, torch.abs(u-v)], dim=-1)  # [btz, hdsz*3]
        y_pred = self.dense(sen_emb_concat)  # [btz, 5]
        return y_pred

model = Model(pool_method='cls').to(device)

class MyLoss(nn.Module):
    """loss分
    """
    def __init__(self) -> None:
        super().__init__()
        self.mask = torch.tensor([0,0,1,1,1], device=device)

    def forward(self, y_pred, y_true):
        '''如果是两分类数据，则把后三位置-inf，如果是三分类数据，把前两位置-inf
        '''
        task = (y_true < 1.5).long()
        y_pred_1 = y_pred - self.mask * 1e12
        y_pred_2 = y_pred - (1-self.mask) * 1e12
        y_pred = task * y_pred_1 + (1-task) * y_pred_2
        return F.cross_entropy(y_pred, y_true.flatten())

model.compile(loss=MyLoss(), optimizer=optim.Adam(model.parameters(), 1e-5), metrics=['seq2seq_loss', 'similarity_loss'])


class Evaluator(Callback):
    """评估模型
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, global_step, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            # model.save_weights('./best_model.pt')

if __name__ == '__main__':    
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=50, steps_per_epoch=None, callbacks=[evaluator])
else:
    model.load_weights('./best_model.pt')

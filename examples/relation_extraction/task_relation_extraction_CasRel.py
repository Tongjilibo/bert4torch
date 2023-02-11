#! -*- coding:utf-8 -*-
# 三元组抽取任务，基于“半指针-半标注”结构
# 思路：两阶段关系抽取，先抽取出句子中的主语，再通过指针网络抽取出主语对应的关系和宾语
# 文章介绍：https://kexue.fm/archives/7161
# 数据集：http://ai.baidu.com/broad/download?dataset=sked

import json
import numpy as np
from bert4torch.layers import LayerNorm
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn

maxlen = 128
batch_size = 64
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载标签字典
predicate2id, id2predicate = {}, {}

with open('F:/Projects/data/corpus/relation_extraction/BD_Knowledge_Extraction/all_50_schemas', encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 解析样本
def get_spoes(text, spo_list):
    '''单独抽出来，这样读取数据时候，可以根据spoes来选择跳过
    '''
    def search(pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1

    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    # 整理三元组 {s: [(o, p)]}
    spoes = {}
    for s, p, o in spo_list:
        s = tokenizer.encode(s)[0][1:-1]
        p = predicate2id[p]
        o = tokenizer.encode(o)[0][1:-1]
        s_idx = search(s, token_ids)
        o_idx = search(o, token_ids)
        if s_idx != -1 and o_idx != -1:
            s = (s_idx, s_idx + len(s) - 1)
            o = (o_idx, o_idx + len(o) - 1, p)
            if s not in spoes:
                spoes[s] = []
            spoes[s].append(o)
    return token_ids, segment_ids, spoes

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据
        单条格式：{'text': text, 'spo_list': [(s, p, o)]}
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in tqdm(f):
                l = json.loads(l)
                labels = [(spo['subject'], spo['predicate'], spo['object']) for spo in l['spo_list']]
                token_ids, segment_ids, spoes = get_spoes(l['text'], labels)
                if spoes:
                    D.append({'text': l['text'], 'spo_list': labels, 'token_ids': token_ids, 
                              'segment_ids': segment_ids, 'spoes': spoes})
        return D

def collate_fn(batch):
    batch_token_ids, batch_segment_ids = [], []
    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
    for d in batch:
        token_ids, segment_ids, spoes = d['token_ids'], d['segment_ids'], d['spoes']
        if spoes:
            # subject标签
            subject_labels = np.zeros((len(token_ids), 2))
            for s in spoes:
                subject_labels[s[0], 0] = 1  # subject首
                subject_labels[s[1], 1] = 1  # subject尾
            # 随机选一个subject（这里没有实现错误！这就是想要的效果！！）
            # Todo: 感觉可以对未选到的subject加个mask，这样计算loss就不会计算到，可能因为模型对prob**n正例加权重导致影响不大
            start, end = np.array(list(spoes.keys())).T
            start = np.random.choice(start)
            end = np.random.choice(end[end >= start])
            subject_ids = (start, end)
            # 对应的object标签
            object_labels = np.zeros((len(token_ids), len(predicate2id), 2))
            for o in spoes.get(subject_ids, []):
                object_labels[o[0], o[2], 0] = 1
                object_labels[o[1], o[2], 1] = 1
            # 构建batch
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_subject_labels.append(subject_labels)
            batch_subject_ids.append(subject_ids)
            batch_object_labels.append(object_labels)
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_subject_labels = torch.tensor(sequence_padding(batch_subject_labels), dtype=torch.float, device=device)
    batch_subject_ids = torch.tensor(batch_subject_ids, dtype=torch.long, device=device)
    batch_object_labels = torch.tensor(sequence_padding(batch_object_labels), dtype=torch.float, device=device)
    batch_attention_mask = (batch_token_ids != tokenizer._token_pad_id)
    return [batch_token_ids, batch_segment_ids, batch_subject_ids], [batch_subject_labels, batch_object_labels, batch_attention_mask]

train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/relation_extraction/BD_Knowledge_Extraction/train_data.json'), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataset = MyDataset('F:/Projects/data/corpus/relation_extraction/BD_Knowledge_Extraction/dev_data.json')
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn) 


# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.bert = build_transformer_model(config_path, checkpoint_path)
        self.linear1 = nn.Linear(768, 2)
        self.condLayerNorm = LayerNorm(hidden_size=768, conditional_size=768*2)
        self.linear2 = nn.Linear(768, len(predicate2id)*2)

    @staticmethod
    def extract_subject(inputs):
        """根据subject_ids从output中取出subject的向量表征
        """
        output, subject_ids = inputs
        start = torch.gather(output, dim=1, index=subject_ids[:, :1].unsqueeze(2).expand(-1, -1, output.shape[-1]))
        end = torch.gather(output, dim=1, index=subject_ids[:, 1:].unsqueeze(2).expand(-1, -1, output.shape[-1]))
        subject = torch.cat([start, end], 2)
        return subject[:, 0]

    def forward(self, *inputs):
        # 预测subject
        seq_output = self.bert(inputs[:2])  # [btz, seq_len, hdsz]
        subject_preds = (torch.sigmoid(self.linear1(seq_output)))**2  # [btz, seq_len, 2]

        # 传入subject，预测object
        # 通过Conditional Layer Normalization将subject融入到object的预测中
        subject_ids = inputs[2]
        # 理论上应该用LayerNorm前的，但是这样只能返回各个block顶层输出，这里和keras实现不一致
        subject = self.extract_subject([seq_output, subject_ids])
        output = self.condLayerNorm([seq_output, subject])
        output = (torch.sigmoid(self.linear2(output)))**4
        object_preds = output.reshape(*output.shape[:2], len(predicate2id), 2)

        return [subject_preds, object_preds]
    
    def predict_subject(self, inputs):
        self.eval()
        with torch.no_grad():
            seq_output = self.bert(inputs[:2])  # [btz, seq_len, hdsz]
            subject_preds = (torch.sigmoid(self.linear1(seq_output)))**2  # [btz, seq_len, 2]
        return [seq_output, subject_preds]
    
    def predict_object(self, inputs):
        self.eval()
        with torch.no_grad():
            seq_output, subject_ids = inputs
            subject = self.extract_subject([seq_output, subject_ids])
            output = self.condLayerNorm([seq_output, subject])
            output = (torch.sigmoid(self.linear2(output)))**4
            object_preds = output.reshape(*output.shape[:2], len(predicate2id), 2)
        return object_preds


train_model = Model().to(device)

class BCELoss(nn.BCELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, inputs, targets):
        subject_preds, object_preds = inputs
        subject_labels, object_labels, mask = targets

        # sujuect部分loss
        subject_loss = super().forward(subject_preds, subject_labels)
        subject_loss = subject_loss.mean(dim=-1)
        subject_loss = (subject_loss * mask).sum() / mask.sum()
        # object部分loss
        object_loss = super().forward(object_preds, object_labels)
        object_loss = object_loss.mean(dim=-1).sum(dim=-1)
        object_loss = (object_loss * mask).sum() / mask.sum()
        return subject_loss + object_loss

train_model.compile(loss=BCELoss(reduction='none'), optimizer=optim.Adam(train_model.parameters(), 1e-5))

def extract_spoes(text):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=device)

    # 抽取subject
    seq_output, subject_preds = train_model.predict_subject([token_ids, segment_ids])
    subject_preds[:, [0, -1]] *= 0  # 首cls, 尾sep置为0
    start = torch.where(subject_preds[0, :, 0] > 0.6)[0]
    end = torch.where(subject_preds[0, :, 1] > 0.5)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i.item(), j.item()))
    if subjects:
        spoes = []
        # token_ids = token_ids.repeat([len(subjects)]+[1]*(len(token_ids.shape)-1))
        # segment_ids = segment_ids.repeat([len(subjects)]+[1]*(len(token_ids.shape)-1))
        seq_output = seq_output.repeat([len(subjects)]+[1]*(len(seq_output.shape)-1))
        subjects = torch.tensor(subjects, dtype=torch.long, device=device)
        # 传入subject，抽取object和predicate
        object_preds = train_model.predict_object([seq_output, subjects])
        object_preds[:, [0, -1]] *= 0
        for subject, object_pred in zip(subjects, object_preds):
            start = torch.where(object_pred[:, :, 0] > 0.6)
            end = torch.where(object_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append(
                            ((mapping[subject[0]][0],
                              mapping[subject[1]][-1]), predicate1.item(),
                             (mapping[_start][0], mapping[_end][-1]))
                        )
                        break
        return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        R = set([SPO(spo) for spo in extract_spoes(d['text'])])
        T = set([SPO(spo) for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
                       ensure_ascii=False,
                       indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        # optimizer.apply_ema_weights()
        f1, precision, recall = evaluate(valid_dataset.data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            # train_model.save_weights('best_model.pt')
        # optimizer.reset_old_weights()
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


if __name__ == '__main__':
    # 训练
    if True:
        evaluator = Evaluator()
        train_model.fit(train_dataloader, steps_per_epoch=None, epochs=20, callbacks=[evaluator])
    # 预测并评估
    else:
        train_model.load_weights('best_model.pt')
        f1, precision, recall = evaluate(valid_dataset.data)


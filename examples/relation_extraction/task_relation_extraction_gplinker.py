#! -*- coding:utf-8 -*-
# 三元组抽取任务，基于GlobalPointer的仿TPLinker设计
# 文章介绍：https://kexue.fm/archives/8888
# 数据集：http://ai.baidu.com/broad/download?dataset=sked

import json
from bert4torch.layers import GlobalPointer
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset
from bert4torch.losses import SparseMultilabelCategoricalCrossentropy
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np

maxlen = 128
batch_size = 32
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

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据
        单条格式：{'text': text, 'spo_list': [(s, p, o)]}
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                D.append({'text': l['text'],
                          'spo_list': [(spo['subject'], spo['predicate'], spo['object']) for spo in l['spo_list']]})
        return D

def collate_fn(batch):
    def search(pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1

    batch_token_ids, batch_segment_ids = [], []
    batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
    for d in batch:
        token_ids, segment_ids = tokenizer.encode(d['text'], maxlen=maxlen)
        # 整理三元组 {s: [(o, p)]}
        spoes = set()
        for s, p, o in d['spo_list']:
            s = tokenizer.encode(s)[0][1:-1]
            p = predicate2id[p]
            o = tokenizer.encode(o)[0][1:-1]
            sh = search(s, token_ids)
            oh = search(o, token_ids)
            if sh != -1 and oh != -1:
                spoes.add((sh, sh + len(s) - 1, p, oh, oh + len(o) - 1))
        # 构建标签
        entity_labels = [set() for _ in range(2)]
        head_labels = [set() for _ in range(len(predicate2id))]
        tail_labels = [set() for _ in range(len(predicate2id))]
        for sh, st, p, oh, ot in spoes:
            entity_labels[0].add((sh, st))
            entity_labels[1].add((oh, ot))
            head_labels[p].add((sh, oh))
            tail_labels[p].add((st, ot))
        for label in entity_labels + head_labels + tail_labels:
            if not label:  # 至少要有一个标签
                label.add((0, 0))  # 如果没有则用0填充
        entity_labels = sequence_padding([list(l) for l in entity_labels])  # [subject/object=2, 实体个数, 实体起终点]
        head_labels = sequence_padding([list(l) for l in head_labels])  # [关系个数, 该关系下subject/object配对数, subject/object起点]
        tail_labels = sequence_padding([list(l) for l in tail_labels])  # [关系个数, 该关系下subject/object配对数, subject/object终点]
        # 构建batch
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_entity_labels.append(entity_labels)
        batch_head_labels.append(head_labels)
        batch_tail_labels.append(tail_labels)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    # batch_entity_labels: [btz, subject/object=2, 实体个数, 实体起终点]
    # batch_head_labels: [btz, 关系个数, 该关系下subject/object配对数, subject/object起点]
    # batch_tail_labels: [btz, 关系个数, 该关系下subject/object配对数, subject/object终点]
    batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2), dtype=torch.float, device=device)
    batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2), dtype=torch.float, device=device)
    batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2), dtype=torch.float, device=device)
    return [batch_token_ids, batch_segment_ids], [batch_entity_labels, batch_head_labels, batch_tail_labels]

train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/relation_extraction/BD_Knowledge_Extraction/train_data.json'), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataset = MyDataset('F:/Projects/data/corpus/relation_extraction/BD_Knowledge_Extraction/dev_data.json')
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn) 


# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.bert = build_transformer_model(config_path, checkpoint_path)
        self.entity_output = GlobalPointer(hidden_size=768, heads=2, head_size=64)
        self.head_output = GlobalPointer(hidden_size=768, heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False)
        self.tail_output = GlobalPointer(hidden_size=768, heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False)

    def forward(self, *inputs):
        hidden_states = self.bert(inputs)  # [btz, seq_len, hdsz]
        mask = inputs[0].gt(0).long()

        entity_output = self.entity_output(hidden_states, mask)  # [btz, heads, seq_len, seq_len]
        head_output = self.head_output(hidden_states, mask)  # [btz, heads, seq_len, seq_len]
        tail_output = self.tail_output(hidden_states, mask)  # [btz, heads, seq_len, seq_len]
        return entity_output, head_output, tail_output
    

model = Model().to(device)

class MyLoss(SparseMultilabelCategoricalCrossentropy):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
    def forward(self, y_preds, y_trues):
        ''' y_preds: [Tensor], shape为[btz, heads, seq_len ,seq_len]
        '''
        loss_list = []
        for y_pred, y_true in zip(y_preds, y_trues):
            shape = y_pred.shape
            # 乘以seq_len是因为(i, j)在展开到seq_len*seq_len维度对应的下标是i*seq_len+j
            y_true = y_true[..., 0] * shape[2] + y_true[..., 1]  # [btz, heads, 实体起终点的下标]
            y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))  # [btz, heads, seq_len*seq_len]
            loss = super().forward(y_pred, y_true.long())
            loss = torch.mean(torch.sum(loss, dim=1))
            loss_list.append(loss)
        return {'loss': sum(loss_list)/3, 'entity_loss': loss_list[0], 'head_loss': loss_list[1], 'tail_loss': loss_list[2]}

model.compile(loss=MyLoss(mask_zero=True), optimizer=optim.Adam(model.parameters(), 1e-5), metrics=['entity_loss', 'head_loss', 'tail_loss'])

def extract_spoes(text, threshold=0):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=device)
    outputs = model.predict([token_ids, segment_ids])
    outputs = [o[0].cpu().numpy() for o in outputs]  # [heads, seq_len, seq_len]
    # 抽取subject和object
    subjects, objects = set(), set()
    outputs[0][:, [0, -1]] -= float('inf')
    outputs[0][:, :, [0, -1]] -= float('inf')
    for l, h, t in zip(*np.where(outputs[0] > threshold)):
        if l == 0:
            subjects.add((h, t))
        else:
            objects.add((h, t))
    # 识别对应的predicate
    spoes = set()
    for sh, st in subjects:
        for oh, ot in objects:
            p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
            p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
            ps = set(p1s) & set(p2s)
            for p in ps:
                spoes.add((
                    text[mapping[sh][0]:mapping[st][-1] + 1], id2predicate[p],
                    text[mapping[oh][0]:mapping[ot][-1] + 1]
                ))
    return list(spoes)


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (tuple(tokenizer.tokenize(spo[0])), spo[1], tuple(tokenizer.tokenize(spo[2])))

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 0, 1e-10, 1e-10
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
        pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall))
        s = json.dumps({'text': d['text'], 'spo_list': list(T), 'spo_list_pred': list(R),
                        'new': list(R - T), 'lack': list(T - R)}, ensure_ascii=False, indent=4)
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
            # model.save_weights('best_model.pt')
        # optimizer.reset_old_weights()
        print('f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %(f1, precision, recall, self.best_val_f1))


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, steps_per_epoch=None, epochs=20, callbacks=[evaluator])
else:
    model.load_weights('best_model.pt')

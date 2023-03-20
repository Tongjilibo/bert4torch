# coding: utf-8
# gplinker 事件提取, 开发测试中，目前dev的最优指标是event_level f1=0.441, argument_level f1=0.728

import os
import json
import numpy as np
from itertools import groupby
from tqdm import tqdm
from bert4torch.losses import SparseMultilabelCategoricalCrossentropy
from bert4torch.tokenizers import Tokenizer
from bert4torch.layers import EfficientGlobalPointer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset
import torch
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


maxlen = 128
batch_size = 16
epochs = 200
config_path = 'F:/Projects/pretrain_ckpt/roberta/[hit_torch_base]--chinese-roberta-wwm-ext-base/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/roberta/[hit_torch_base]--chinese-roberta-wwm-ext-base/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/roberta/[hit_torch_base]--chinese-roberta-wwm-ext-base/vocab.txt'
model_name = 'Chinese_roberta_wwm_ext'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_path = 'F:/Projects/data/corpus/event_extraction/duee/duee_train.json'
valid_path = 'F:/Projects/data/corpus/event_extraction/duee/duee_dev.json'
test_path = 'F:/Projects/data/corpus/event_extraction/duee/duee_test2.json'
schema_path = 'F:/Projects/data/corpus/event_extraction/duee/duee_event_schema.json'
best_model_save_path = './'
optimizer_name = 'adam'
best_e_weigths_save_path = os.path.join(best_model_save_path, 'best_model.{}.e.weights.pt'.format(optimizer_name))
best_a_weigths_save_path = os.path.join(best_model_save_path, 'best_model.{}.a.weights.pt'.format(optimizer_name))
log_dir = './test_log'


# 读取schema
labels = []
with open(schema_path, 'r', encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)
        t = l['event_type']
        for r in [u'触发词'] + [s['role'] for s in l['role_list']]:
            labels.append((t, r))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据
        单条格式：{'text': text, 'events': [[(type, role, argument, start_index)]]}
        """
        D = []
        with open(filename, 'r', encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                d = {'text': l['text'], 'events': []}
                for e in l['event_list']:
                    d['events'].append([(
                        e['event_type'], u'触发词', e['trigger'],
                        e['trigger_start_index']
                    )])
                    for a in e['arguments']:
                        d['events'][-1].append((
                            e['event_type'], a['role'], a['argument'],
                            a['argument_start_index']
                        ))
                D.append(d)
        return D


def collate_fn(batch):
    batch_token_ids, batch_segment_ids = [], []
    batch_argu_labels, batch_head_labels, batch_tail_labels = [], [], []
    for d in batch:
        tokens = tokenizer.tokenize(d['text'], maxlen=maxlen)
        # 这个函数的是把token在原始文本中的位置计算出来，返回是个二维数组
        mapping = tokenizer.rematch(d['text'], tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        # 整理事件
        events = []
        for e in d['events']:
            events.append([])
            for t, r, a, i in e:
                label = labels.index((t, r))
                start, end = i, i + len(a) - 1
                if start in start_mapping and end in end_mapping:
                    start, end = start_mapping[start], end_mapping[end]
                    events[-1].append((label, start, end))
        # 构建标签
        argu_labels = [set() for _ in range(len(labels))]
        head_labels, tail_labels = set(), set()
        for e in events:
            for l, h, t in e:
                argu_labels[l].add((h, t))
            for i1, (_, h1, t1) in enumerate(e):
                for i2, (_, h2, t2) in enumerate(e):
                    if i2 > i1:
                        head_labels.add((min(h1, h2), max(h1, h2)))
                        tail_labels.add((min(t1, t2), max(t1, t2)))
        for label in argu_labels + [head_labels, tail_labels]:
            if not label:  # 至少要有一个标签
                label.add((0, 0))  # 如果没有则用0填充
        argu_labels = sequence_padding([list(l) for l in argu_labels])
        head_labels = sequence_padding([list(head_labels)])
        tail_labels = sequence_padding([list(tail_labels)])
        # 构建batch
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_argu_labels.append(argu_labels)
        batch_head_labels.append(head_labels)
        batch_tail_labels.append(tail_labels)
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_argu_labels = torch.tensor(sequence_padding(batch_argu_labels, seq_dims=2), dtype=torch.long, device=device)
    batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2), dtype=torch.long, device=device)
    batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2), dtype=torch.long, device=device)
    # return X, Y
    return [batch_token_ids, batch_segment_ids], [batch_argu_labels, batch_head_labels, batch_tail_labels]


train_dataloader = DataLoader(MyDataset(train_path), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataset = MyDataset(valid_path)


# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.bert = build_transformer_model(config_path, checkpoint_path)
        self.argu_output = EfficientGlobalPointer(hidden_size=768, heads=len(labels), head_size=64)
        self.head_output = EfficientGlobalPointer(hidden_size=768, heads=1, head_size=64, RoPE=False)
        self.tail_output = EfficientGlobalPointer(hidden_size=768, heads=1, head_size=64, RoPE=False)

    def forward(self, *inputs):
        hidden_states = self.bert(inputs)  # [btz, seq_len, hdsz]
        mask = inputs[0].gt(0).long()

        argu_output = self.argu_output(hidden_states, mask)  # [btz, heads, seq_len, seq_len]
        head_output = self.head_output(hidden_states, mask)  # [btz, heads, seq_len, seq_len]
        tail_output = self.tail_output(hidden_states, mask)  # [btz, heads, seq_len, seq_len]
        return argu_output, head_output, tail_output


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
        return {'loss': sum(loss_list)/3, 'argu_loss': loss_list[0], 'head_loss': loss_list[1], 'tail_loss': loss_list[2]}

model.compile(
            loss=MyLoss(mask_zero=True),
            optimizer=optim.Adam(model.parameters(), 2e-5),
            metrics=['argu_loss', 'head_loss', 'tail_loss'],
            )


class DedupList(list):
    """定义去重的list
    """
    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def neighbors(host, argus, links):
    """构建邻集（host节点与其所有邻居的集合）
    """
    results = [host]
    for argu in argus:
        if host[2:] + argu[2:] in links:
            results.append(argu)
    return list(sorted(results))


def clique_search(argus, links):
    """搜索每个节点所属的完全子图作为独立事件
    搜索思路：找出不相邻的节点，然后分别构建它们的邻集，递归处理。
    """
    Argus = DedupList()
    for i1, (_, _, h1, t1) in enumerate(argus):
        for i2, (_, _, h2, t2) in enumerate(argus):
            if i2 > i1:
                if (h1, t1, h2, t2) not in links:
                    Argus.append(neighbors(argus[i1], argus, links))
                    Argus.append(neighbors(argus[i2], argus, links))
    if Argus:
        results = DedupList()
        for A in Argus:
            for a in clique_search(A, links):
                results.append(a)
        return results
    else:
        return [list(sorted(argus))]


def extract_events(text, threshold=0, trigger=True):
    """抽取输入text所包含的所有事件
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=device)
    outputs = model.predict([token_ids, segment_ids])
    # for item in outputs:
    #     print(item.shape)
    outputs = [o[0].cpu().numpy() for o in outputs]
    # 抽取论元
    argus = set()
    # 把首尾的CLS和SEP mask掉
    outputs[0][:, [0, -1]] -= np.inf
    outputs[0][:, :, [0, -1]] -= np.inf
    for l, h, t in zip(*np.where(outputs[0] > threshold)):
        argus.add(labels[l] + (h, t))
    # 构建链接
    links = set()
    for i1, (_, _, h1, t1) in enumerate(argus):
        for i2, (_, _, h2, t2) in enumerate(argus):
            if i2 > i1:
                if outputs[1][0, min(h1, h2), max(h1, h2)] > threshold:
                    if outputs[2][0, min(t1, t2), max(t1, t2)] > threshold:
                        links.add((h1, t1, h2, t2))
                        links.add((h2, t2, h1, t1))
    # 析出事件
    events = []
    for _, sub_argus in groupby(sorted(argus), key=lambda s: s[0]):
        for event in clique_search(list(sub_argus), links):
            events.append([])
            for argu in event:
                start, end = mapping[argu[2]][0], mapping[argu[3]][-1] + 1
                events[-1].append(argu[:2] + (text[start:end], start))
            if trigger and all([argu[1] != u'触发词' for argu in event]):
                events.pop()
    return events


def evaluate(data, threshold=0):
    """评估函数，计算f1、precision、recall
    """
    ex, ey, ez = 1e-10, 1e-10, 1e-10  # 事件级别
    ax, ay, az = 1e-10, 1e-10, 1e-10  # 论元级别
    for d in tqdm(data, ncols=0):
        pred_events = extract_events(d['text'], threshold, False)
        # 事件级别
        R, T = DedupList(), DedupList()
        for event in pred_events:
            if any([argu[1] == u'触发词' for argu in event]):
                R.append(list(sorted(event)))
        for event in d['events']:
            T.append(list(sorted(event)))
        for event in R:
            if event in T:
                ex += 1
        ey += len(R)
        ez += len(T)
        # 论元级别
        R, T = DedupList(), DedupList()
        for event in pred_events:
            for argu in event:
                if argu[1] != u'触发词':
                    R.append(argu)
        for event in d['events']:
            for argu in event:
                if argu[1] != u'触发词':
                    T.append(argu)
        for argu in R:
            if argu in T:
                ax += 1
        ay += len(R)
        az += len(T)
    e_f1, e_pr, e_rc = 2 * ex / (ey + ez), ex / ey, ex / ez
    a_f1, a_pr, a_rc = 2 * ax / (ay + az), ax / ay, ax / az
    return e_f1, e_pr, e_rc, a_f1, a_pr, a_rc


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_e_f1 = 0.
        self.best_val_a_f1 = 0.
        self.writer = SummaryWriter(log_dir)

    def on_batch_end(self, global_step, local_step, logs=None):
        for k, v in logs.items():
            if 'loss' not in k:
                continue
            tag = '{}/train'.format(k)
            self.writer.add_scalar(tag, v, global_step)

    def on_epoch_end(self, steps, epoch, logs=None):
        e_f1, e_pr, e_rc, a_f1, a_pr, a_rc = evaluate(valid_dataset.data)
        if e_f1 >= self.best_val_e_f1:
            self.best_val_e_f1 = e_f1
            model.save_weights(best_e_weigths_save_path)
        if a_f1 >= self.best_val_a_f1:
            self.best_val_a_f1 = a_f1
            model.save_weights(best_a_weigths_save_path)
        print(
            '[event level] f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f'
            % (e_f1, e_pr, e_rc, self.best_val_e_f1)
        )
        print(
            '[argument level] f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n'
            % (a_f1, a_pr, a_rc, self.best_val_a_f1)
        )
        self.writer.add_scalar('f1/event_level', e_f1, steps)
        self.writer.add_scalar('precision/event_level', e_pr, steps)
        self.writer.add_scalar('recall/event_level', e_rc, steps)

        self.writer.add_scalar('f1/argument_level', a_f1, steps)
        self.writer.add_scalar('precision/event_level', a_pr, steps)
        self.writer.add_scalar('recall/event_level', a_rc, steps)

    def on_train_end(self, logs=None):
        model.load_weights(best_e_weigths_save_path)
        predict_to_file(test_path, 'duee_{}_{}.json'.format(model_name, optimizer_name))


def isin(event_a, event_b):
    """判断event_a是否event_b的一个子集
    """
    if event_a['event_type'] != event_b['event_type']:
        return False
    for argu in event_a['arguments']:
        if argu not in event_b['arguments']:
            return False
    return True


def predict_to_file(in_file, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            event_list = DedupList()
            for event in extract_events(l['text']):
                final_event = {
                    'event_type': event[0][0],
                    'arguments': DedupList()
                }
                for argu in event:
                    if argu[1] != u'触发词':
                        final_event['arguments'].append({
                            'role': argu[1],
                            'argument': argu[2]
                        })
                event_list = [
                    event for event in event_list
                    if not isin(event, final_event)
                ]
                if not any([isin(final_event, event) for event in event_list]):
                    event_list.append(final_event)
            l['event_list'] = event_list
            l = json.dumps(l, ensure_ascii=False, indent=4)
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, steps_per_epoch=None, epochs=epochs, callbacks=[evaluator])

else:
    model.load_weights(best_e_weigths_save_path)
    predict_to_file(test_path, 'duee_{}_{}.json'.format(model_name, optimizer_name))
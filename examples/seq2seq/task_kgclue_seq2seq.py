#! -*- coding: utf-8 -*-
# KgCLUE baseline
# 直接用UniLM做Seq2Seq，然后前缀树约束解码，并加入自研的“前瞻”策略；
# 基础模型为RoFormer-Sim-FT，相比直接用RoFormer/BERT/RoBERTa有2%的提升；
# 介绍链接：https://kexue.fm/archives/8802

import os, json
import numpy as np
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from bert4torch.snippets import ListDataset, sequence_padding, AutoRegressiveDecoder, Callback
from tqdm import tqdm
from collections import defaultdict
# import pylcs


def lcs(source, target):
    """最长公共子序列（source和target的最长非连续子序列）
    返回：子序列长度, 映射关系（映射对组成的list）
    注意：最长公共子序列可能不止一个，所返回的映射只代表其中一个。
    """
    c = defaultdict(int)
    for i, si in enumerate(source, 1):
        for j, tj in enumerate(target, 1):
            if si == tj:
                c[i, j] = c[i - 1, j - 1] + 1
            elif c[i, j - 1] > c[i - 1, j]:
                c[i, j] = c[i, j - 1]
            else:
                c[i, j] = c[i - 1, j]
    l, mapping = c[len(source), len(target)], []
    i, j = len(source) - 1, len(target) - 1
    while len(mapping) < l:
        if source[i] == target[j]:
            mapping.append((i, j))
            i, j = i - 1, j - 1
        elif c[i + 1, j] > c[i, j + 1]:
            j = j - 1
        else:
            i = i - 1
    return l, mapping[::-1]


def subject_split(s):
    """如果有义项，那么单独分离出来
    """
    m = ''
    if s[-1] == u'）':
        i = s.index(u'（')
        m = s[i + 1:-1]
        s = s[:i]
    return s, m


def load_data(filename):
    """读取数据集
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            s, p, o = l['answer'].split(' ||| ')
            s, m = subject_split(s)
            D.append((l['question'], (s, p, m, ' '.join(o.split()))))
    return D


class Trie(object):
    """自定义Trie树对象，用来保存知识库
    """
    def __init__(self, value_key=-1):
        self.data = {}
        self.value_key = str(value_key)

    def __setitem__(self, key, value):
        """传入一对(key, value)到前缀树中
        """
        data = self.data
        for k in key:
            k = str(k)
            if k not in data:
                data[k] = {}
            data = data[k]
        if self.value_key in data:
            if data[self.value_key] != value:
                data[self.value_key] += ('\t' + value)
        else:
            data[self.value_key] = value

    def __getitem__(self, key):
        """获取key对应的value
        """
        data = self.data
        for k in key:
            k = str(k)
            data = data[k]
        return data[self.value_key]

    def next_ones(self, prefix):
        """获取prefix后一位的容许集
        """
        data = self.data
        for k in prefix:
            k = str(k)
            data = data[k]
        return [k for k in data if k != self.value_key]

    def keys(self, prefix=None, data=None):
        """获取以prefix开头的所有key
        """
        data = data or self.data
        prefix = prefix or []
        for k in prefix:
            k = str(k)
            if k not in data:
                return []
            data = data[k]
        results = []
        for k in data:
            if k == self.value_key:
                results.append([])
            else:
                results.extend([[k] + j for j in self.keys(None, data[k])])
        return [prefix + i for i in results]

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False)

    def load(self, filename):
        with open(filename, encoding='utf-8') as f:
            self.data = json.load(f)


# 基本参数
maxlen = 128
batch_size = 32
epochs = 10

# 模型路径
config_path = 'F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--roformer_chinese_sim_char_ft_base/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--roformer_chinese_sim_char_ft_base/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--roformer_chinese_sim_char_ft_base/vocab.txt'
device =  'cuda' if torch.cuda.is_available() else 'cpu'

# 加载分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 转换知识库
KG = Trie()
if os.path.exists('../datasets/KG.json'):
    KG.load('../datasets/KG.json')
else:
    with open('F:/Projects/data/corpus/kg/KgCLUE/Knowledge_20211215.txt', 'r', encoding='utf-8') as f:
        # count = 0
        for l in tqdm(f):
            s, p, o = l.split('\t')
            s, m = subject_split(s)
            ids = tokenizer.encode(s, p)[0][1:]
            ids += tokenizer.encode(m)[0][1:-1]
            KG[ids] = ' '.join(o.split())
            # count += 1
            # if count > 10000:
            #     break
    KG.save('../datasets/KG.json')


def collate_fn(batch):
    """数据生成器
    单条样本：[CLS] Q [SEP] S [SEP] P [SEP] M [SEP]
    """
    batch_token_ids, batch_segment_ids = [], []
    for (q, a) in batch:
        q_ids = tokenizer.encode(q, maxlen=maxlen // 2 + 1)[0]
        a_ids = tokenizer.encode(a[0], a[1])[0]
        a_ids += tokenizer.encode(a[2])[0][1:]
        token_ids = (q_ids + a_ids[1:])[:maxlen]
        segment_ids = [0] * len(q_ids)
        segment_ids += [1] * (len(token_ids) - len(q_ids))
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], [batch_token_ids, batch_segment_ids]

# 读取数据集
train_data = load_data('F:/Projects/data/corpus/kg/KgCLUE/train.json')
train_dataloader = DataLoader(ListDataset(train_data), shuffle=True, collate_fn=collate_fn)
valid_data = load_data('F:/Projects/data/corpus/kg/KgCLUE/dev.json')
test_data = load_data('F:/Projects/data/corpus/kg/KgCLUE/test_public.json')

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, outputs, target):
        '''
        y_pred: [btz, seq_len, vocab_size]
        targets: y_true, y_segment
        unilm式样，需要手动把非seq2seq部分mask掉
        '''
        _, y_pred = outputs
        y_true, y_mask = target
        y_true = y_true[:, 1:]# 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1, :]  # 预测序列，错开一位
        
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = (y_true*y_mask).flatten()
        return super().forward(y_pred, y_true)


model = build_transformer_model(config_path, checkpoint_path, model='roformer', application='unilm', add_trainer=True).to(device)

model.compile(loss=CrossEntropyLoss(ignore_index=0), optimizer=optim.Adam(model.parameters(), 5e-6))

class AutoQA(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        all_token_ids = torch.cat([token_ids, output_ids], 1)
        segment_ids = torch.cat([segment_ids, torch.ones_like(output_ids, device=device)], 1)
        _, y_pred = model.predict([all_token_ids, segment_ids])
        probas = F.softmax(y_pred[:, -1, :], dim=-1)
        new_probas = torch.zeros_like(probas)
        for i, ids in enumerate(output_ids):
            ids = ids.cpu().numpy()
            next_ids = [int(j) for j in KG.next_ones(ids)]  # 下一位容许集
            # ===========如果t时刻为Pt的前缀树中的短句，带来的信息增益越大，则增加Pt的概率
            if len(next_ids) > 1 and self.end_id in ids:  # 容许集大于1且已解码出S
                candidates = KG.keys(list(ids))  # 可能解码结果
                weights = torch.ones_like(probas[i])  # 默认权重为1
                lcs0 = lcs(ids, token_ids[i])[0]  # 当前已经覆盖的token数
                for c in candidates:
                    if len(c) > len(ids):
                        c = [int(j) for j in c]
                        w = lcs(c, token_ids[i])[0] - lcs0  # 未来还可能覆盖的token数
                        weights[c[len(ids)]] = max(w + 1, weights[c[len(ids)]].cpu().numpy())
                probas[i] = torch.pow(probas[i], 1. / weights)  # 按 p^(1/n) 来增大权重
            if not next_ids:  # 如果容许集为空，意味着要结束了
                next_ids.append(self.end_id)
            new_probas[i, next_ids] += probas[i, next_ids]  # 只保留容许集概率
        new_probas /= new_probas.sum(axis=1, keepdims=True)  # 重新归一化
        return new_probas

    def generate(self, text, topk=1):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.beam_search([token_ids, segment_ids], topk=topk, min_ends=3)  # 基于beam search
        end_idxs = [i for i, j in enumerate(output_ids) if j == self.end_id]
        subject_ids = output_ids[:end_idxs[0]]
        predicate_ids = output_ids[end_idxs[0]:end_idxs[1]]
        meaning_ids = output_ids[end_idxs[1]:]
        return (
            tokenizer.decode(subject_ids.cpu().numpy()), tokenizer.decode(predicate_ids.cpu().numpy()),
            tokenizer.decode(meaning_ids.cpu().numpy()), KG[output_ids[:-1].cpu().numpy()]
        )


autoqa = AutoQA(start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen, device=device)


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_score = 0

    def on_epoch_end(self, steps, epoch, logs=None):
        # 保存最优
        em, f1, score = self.evaluate(valid_data, topk=3)
        if score >= self.best_score:
            self.best_score = score
            # model.save_weights('./best_model.weights')
        print(
            u'[VALID] em: %.5f, f1: %.5f, score: %.5f, best_score: %.5f\n' %
            (em, f1, score, self.best_score)
        )

    def f1sim(self, text_a, text_b):
        """计算两个文本之间的f1相似度
        说明：算出两个文本的最长公共子序列长度，然后乘2并处以两者
        长度之和。推荐用pylcs算，速度较快。
        """
        if not text_a and not text_b:
            return 0.
        else:
            lcs_len = lcs(text_a, text_b)[0]
            return 2. * lcs_len / (len(text_a) + len(text_b))

    def evaluate(self, data, topk=1):
        """评估函数
        注意：同一(S, P)对应的O可能有多个，但标注数据只保留了
        一个，为了跟标注数据对齐来提高分数，这里也只保留第一个。
        """
        em, f1, total = 0., 0., 0.
        for d in tqdm(data, ncols=0):
            a = autoqa.generate(d[0], topk=topk)
            o = a[3].split('\t')[0]  # 如果有多个，只保留第一个
            em += float(o == d[1][3])
            f1 += self.f1sim(o, d[1][3])
            total += 1
        em /= total
        f1 /= total
        return em, f1, (em + f1) / 2


def test_predict(in_file, out_file, topk=1):
    """输出测试结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    fw = open(out_file, 'w')

    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            s, p, m, o = autoqa.generate(l['question'], topk=topk)
            if m:
                s += u'（%s）' % m
            l['answer'] = '%s ||| %s ||| %s' % (s, p, o.split('\t')[0])
            l = json.dumps(l, ensure_ascii=False)
            fw.write(l + '\n')

    fw.close()


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_dataloader,
        steps_per_epoch=None,
        epochs=epochs,
        callbacks=[evaluator]
    )

    model.load_weights('./best_model.weights')
    em, f1, score = evaluator.evaluate(test_data, topk=1)
    print(u'[TEST] topk=1, em: %.5f, f1: %.5f, score: %.5f' % (em, f1, score))
    em, f1, score = evaluator.evaluate(test_data, topk=3)
    print(u'[TEST] topk=3, em: %.5f, f1: %.5f, score: %.5f' % (em, f1, score))
    em, f1, score = evaluator.evaluate(test_data, topk=5)
    print(u'[TEST] topk=5, em: %.5f, f1: %.5f, score: %.5f' % (em, f1, score))

else:

    model.load_weights('./best_model.weights')
    # test_predict('../datasets/test.json', 'kgclue_predict.json', topk=3)
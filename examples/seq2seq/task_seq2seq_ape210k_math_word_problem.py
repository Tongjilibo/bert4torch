#! -*- coding: utf-8 -*-
# 用Seq2Seq做小学数学应用题
# 数据集为ape210k：https://github.com/Chenny0808/ape210k
# 介绍链接：https://kexue.fm/archives/7809

from __future__ import division
import json, re
from tqdm import tqdm
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from bert4torch.snippets import sequence_padding, Callback, ListDataset
from bert4torch.snippets import AutoRegressiveDecoder
from sympy import Integer
import warnings
warnings.filterwarnings("ignore")

# 基本参数
maxlen = 192
batch_size = 16
epochs = 100

# bert配置
config_path = 'F:/Projects/pretrain_ckpt/bert/[hit_torch_base]--chinese-bert-wwm-ext/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[hit_torch_base]--chinese-bert-wwm-ext/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[hit_torch_base]--chinese-bert-wwm-ext/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


def is_equal(a, b):
    """比较两个结果是否相等
    """
    a = round(float(a), 6)
    b = round(float(b), 6)
    return a == b


def remove_bucket(equation):
    """去掉冗余的括号
    """
    l_buckets, buckets = [], []
    for i, c in enumerate(equation):
        if c == '(':
            l_buckets.append(i)
        elif c == ')':
            buckets.append((l_buckets.pop(), i))
    eval_equation = eval(equation)
    for l, r in buckets:
        new_equation = '%s %s %s' % (equation[:l], equation[l + 1:r], equation[r + 1:])
        try:
            if is_equal(eval(new_equation.replace(' ', '')), eval_equation):
                equation = new_equation
        except:
            pass
    return equation.replace(' ', '')

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """读取训练数据，并做一些标准化，保证equation是可以eval的
        参考：https://kexue.fm/archives/7809
        """
        D = []
        for l in open(filename, 'r', encoding='utf-8'):
            l = json.loads(l)
            question, equation, answer = l['original_text'], l['equation'], l['ans']
            # 处理带分数
            question = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', question)
            equation = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', equation)
            answer = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+\\2)', answer)
            equation = re.sub('(\d+)\(', '\\1+(', equation)
            answer = re.sub('(\d+)\(', '\\1+(', answer)
            # 分数去括号
            question = re.sub('\((\d+/\d+)\)', '\\1', question)
            # 处理百分数
            equation = re.sub('([\.\d]+)%', '(\\1/100)', equation)
            answer = re.sub('([\.\d]+)%', '(\\1/100)', answer)
            # 冒号转除号、剩余百分号处理
            equation = equation.replace(':', '/').replace('%', '/100')
            answer = answer.replace(':', '/').replace('%', '/100')
            if equation[:2] == 'x=':
                equation = equation[2:]
            try:
                if is_equal(eval(equation), eval(answer)):
                    D.append((question, remove_bucket(equation), answer))
            except:
                continue
        return D


def collate_fn(batch):
    batch_token_ids, batch_segment_ids = [], []
    for question, equation, answer in batch:
        token_ids, segment_ids = tokenizer.encode(question, equation, maxlen=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], [batch_token_ids, batch_segment_ids]

# 加载数据集
train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/seq2seq/ape210k/train.ape.json'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_dataset = MyDataset('F:/Projects/data/corpus/seq2seq/ape210k/valid.ape.json')
# valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# test_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/seq2seq/ape210k/test.ape.json'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


model = build_transformer_model(
    config_path,
    checkpoint_path,
    with_mlm=True,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    add_trainer=True
).to(device)


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
model.compile(loss=CrossEntropyLoss(ignore_index=0), optimizer=optim.Adam(model.parameters(), 1e-5))


class AutoSolve(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = torch.cat([token_ids, output_ids], 1)
        segment_ids = torch.cat([segment_ids, torch.ones_like(output_ids, device=device)], 1)
        _, y_pred = model.predict([token_ids, segment_ids])
        return y_pred[:, -1, :]

    def generate(self, text, topk=1):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.beam_search([token_ids, segment_ids], topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids.cpu().numpy()).replace(' ', '')


autosolve = AutoSolve(start_id=None, end_id=tokenizer._token_end_id, maxlen=64, device=device)


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_acc = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        metrics = self.evaluate(valid_dataset.data[:200])  # 评测模型
        if metrics['acc'] >= self.best_acc:
            self.best_acc = metrics['acc']
            # model.save_weights('./best_model_math.pt')  # 保存模型
        metrics['best_acc'] = self.best_acc
        print('valid_data:', metrics)
        print()

    def evaluate(self, data, topk=1):
        total, right = 0.0, 0.0
        for question, equation, answer in tqdm(data, desc='Evaluate'):
            total += 1
            pred_equation = autosolve.generate(question, topk)
            try:
                right += int(is_equal(eval(pred_equation), eval(answer)))
            except:
                pass
        return {'acc': right / total}


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, steps_per_epoch=None, epochs=epochs, callbacks=[evaluator])
else:
    model.load_weights('./best_model.weights')
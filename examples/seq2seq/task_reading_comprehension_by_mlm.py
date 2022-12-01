#! -*- coding: utf-8 -*-
# 用MLM的方式做阅读理解任务
# 数据集和评测同 https://github.com/bojone/dgcnn_for_reading_comprehension

import json, os
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.snippets import sequence_padding
from bert4torch.snippets import Callback, ListDataset
from tqdm import tqdm
import torch
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import re
import torch.nn.functional as F

# 基本参数
max_p_len = 256
max_q_len = 64
max_a_len = 32
batch_size = 12
epochs = 10

# bert配置
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def process_data():
    if os.path.exists('F:/Projects/data/corpus/qa/CIPS-SOGOU/train_data.json'):
        return

    # 标注数据
    webqa_data = json.load(open('F:/Projects/data/corpus/qa/WebQA.json', encoding='utf-8'))
    sogou_data = json.load(open('F:/Projects/data/corpus/qa/SogouQA.json', encoding='utf-8'))

    # 保存一个随机序（供划分valid用）
    random_order = list(range(len(sogou_data)))
    np.random.seed(2022)
    np.random.shuffle(random_order)

    # 划分valid
    train_data = [sogou_data[j] for i, j in enumerate(random_order) if i % 3 != 0]
    valid_data = [sogou_data[j] for i, j in enumerate(random_order) if i % 3 == 0]
    train_data.extend(train_data)
    train_data.extend(webqa_data)  # 将SogouQA和WebQA按2:1的比例混合

    json.dump(train_data, open('F:/Projects/data/corpus/qa/CIPS-SOGOU/train_data.json', 'w', encoding='utf-8'), indent=4)
    json.dump(valid_data, open('F:/Projects/data/corpus/qa/CIPS-SOGOU/valid_data.json', 'w', encoding='utf-8'), indent=4)

process_data()

class MyDataset(ListDataset):
    @staticmethod
    def load_data(file_path):
        return json.load(open(file_path))


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


def collate_fn(batch):
    """单条样本格式为
    输入: [CLS][MASK][MASK][SEP]问题[SEP]篇章[SEP]
    输出: 答案
    """
    batch_token_ids, batch_segment_ids, batch_a_token_ids = [], [], []
    for D in batch:
        question = D['question']
        answers = [p['answer'] for p in D['passages'] if p['answer']]
        passage = np.random.choice(D['passages'])['passage']
        passage = re.sub(u' |、|；|，', ',', passage)
        final_answer = ''
        for answer in answers:
            if all([a in passage[:max_p_len - 2] for a in answer.split(' ')]):
                final_answer = answer.replace(' ', ',')
                break
        a_token_ids, _ = tokenizer.encode(final_answer, maxlen=max_a_len + 1)
        q_token_ids, _ = tokenizer.encode(question, maxlen=max_q_len + 1)
        p_token_ids, _ = tokenizer.encode(passage, maxlen=max_p_len + 1)
        token_ids = [tokenizer._token_start_id]
        token_ids += ([tokenizer._token_mask_id] * max_a_len)
        token_ids += [tokenizer._token_end_id]
        token_ids += (q_token_ids[1:] + p_token_ids[1:])
        segment_ids = [0] * len(token_ids)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_a_token_ids.append(a_token_ids[1:])

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_a_token_ids = torch.tensor(sequence_padding(batch_a_token_ids, max_a_len), dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], batch_a_token_ids

train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/qa/CIPS-SOGOU/train_data.json'), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataset = MyDataset('F:/Projects/data/corpus/qa/CIPS-SOGOU/valid_data.json')
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn) 

model = build_transformer_model(
    config_path,
    checkpoint_path,
    with_mlm=True,
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    add_trainer=True
).to(device)
summary(model, input_data=[next(iter(train_dataloader))[0]])

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, outputs, y_true):
        '''
        y_pred: [btz, seq_len, hdsz]
        y_true: [btz, max_a_len]
        '''
        _, y_pred = outputs
        y_pred = y_pred[:, 1:max_a_len+1, :]  # 预测序列，错开一位
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.flatten()
        return super().forward(y_pred, y_true)
model.compile(loss=CrossEntropyLoss(ignore_index=0), optimizer=optim.Adam(model.parameters(), 1e-5))

def get_ngram_set(x, n):
    """生成ngram合集，返回结果格式是:
    {(n-1)-gram: set([n-gram的第n个字集合])}
    """
    result = {}
    for i in range(len(x) - n + 1):
        k = tuple(x[i:i + n])
        if k[:-1] not in result:
            result[k[:-1]] = set()
        result[k[:-1]].add(k[-1])
    return result


def gen_answer(question, passages):
    """由于是MLM模型，所以可以直接argmax解码。
    """
    all_p_token_ids, token_ids, segment_ids = [], [], []
    for passage in passages:
        passage = re.sub(u' |、|；|，', ',', passage)
        p_token_ids, _ = tokenizer.encode(passage, maxlen=max_p_len + 1)
        q_token_ids, _ = tokenizer.encode(question, maxlen=max_q_len + 1)
        all_p_token_ids.append(p_token_ids[1:])
        token_ids.append([tokenizer._token_start_id])
        token_ids[-1] += ([tokenizer._token_mask_id] * max_a_len)
        token_ids[-1] += [tokenizer._token_end_id]
        token_ids[-1] += (q_token_ids[1:] + p_token_ids[1:])
        segment_ids.append([0] * len(token_ids[-1]))
    token_ids = torch.tensor(sequence_padding(token_ids), device=device)
    segment_ids = torch.tensor(sequence_padding(segment_ids), device=device)
    logit = model.predict([token_ids, segment_ids])[-1][:, 1:max_a_len+1, :]
    probas = F.softmax(logit, dim=-1)
    results = {}
    for t, p in zip(all_p_token_ids, probas):
        a, score = tuple(), 0.
        for i in range(max_a_len):
            idxs = list(get_ngram_set(t, i + 1)[a])
            if tokenizer._token_end_id not in idxs:
                idxs.append(tokenizer._token_end_id)
            # pi是将passage以外的token的概率置零
            pi = torch.zeros_like(p[i])
            pi[idxs] = p[i, idxs]
            a = a + (pi.argmax().item(),)
            score += pi.max().item()
            if a[-1] == tokenizer._token_end_id:
                break
        score = score / (i + 1)
        a = tokenizer.decode(a)
        if a:
            results[a] = results.get(a, []) + [score]
    results = {
        k: (np.array(v)**2).sum() / (sum(v) + 1)
        for k, v in results.items()
    }
    return results


def max_in_dict(d):
    if d:
        return sorted(d.items(), key=lambda s: -s[1])[0][0]


def predict_to_file(data, filename):
    """将预测结果输出到文件，方便评估
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for d in tqdm(iter(data), desc=u'正在预测(共%s条样本)' % len(data)):
            q_text = d['question']
            p_texts = [p['passage'] for p in d['passages']]
            a = gen_answer(q_text, p_texts)
            a = max_in_dict(a)
            if a:
                s = u'%s\t%s\n' % (d['id'], a)
            else:
                s = u'%s\t\n' % (d['id'])
            f.write(s)
            f.flush()


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, steps, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            # model.save_weights('./best_model.pt')
        predict_to_file(valid_dataset.data[:100], 'qa.csv')

if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_dataloader,
        steps_per_epoch=None,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    model.load_weights('./best_model.pt')
    # predict_to_file(valid_data, 'qa.csv')

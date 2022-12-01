#! -*- coding: utf-8 -*-
# 用seq2seq的方式做阅读理解任务
# 数据集和评测同 https://github.com/bojone/dgcnn_for_reading_comprehension

import json, os
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.snippets import sequence_padding, text_segmentate
from bert4torch.snippets import AutoRegressiveDecoder, Callback, ListDataset
from tqdm import tqdm
import torch
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import re 

# 基本参数
max_p_len = 256
max_q_len = 64
max_a_len = 32
max_qa_len = max_q_len + max_a_len
batch_size = 8
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
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


def collate_fn(batch):
    """单条样本格式: [CLS]篇章[SEP]问题[SEP]答案[SEP]
    """
    batch_token_ids, batch_segment_ids = [], []
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
        qa_token_ids, qa_segment_ids = tokenizer.encode(question, final_answer, maxlen=max_qa_len + 1)
        p_token_ids, p_segment_ids = tokenizer.encode(passage, maxlen=max_p_len + 1)
        token_ids = p_token_ids + qa_token_ids[1:]
        segment_ids = p_segment_ids + qa_segment_ids[1:]
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], [batch_token_ids, batch_segment_ids]

train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/qa/CIPS-SOGOU/train_data.json'), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataset = MyDataset('F:/Projects/data/corpus/qa/CIPS-SOGOU/valid_data.json')
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn) 

model = build_transformer_model(
    config_path,
    checkpoint_path,
    with_mlm=True,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    add_trainer=True
).to(device)
summary(model, input_data=[next(iter(train_dataloader))[0]])

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, outputs, target):
        '''
        y_pred: [btz, seq_len, hdsz]
        targets: y_true, y_segment
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

class ReadingComprehension(AutoRegressiveDecoder):
    """beam search解码来生成答案
    passages为多篇章组成的list，从多篇文章中自动决策出最优的答案，
    如果没答案，则返回空字符串。
    mode是extractive时，按照抽取式执行，即答案必须是原篇章的一个片段。
    """
    def __init__(self, mode='extractive', **kwargs):
        super(ReadingComprehension, self).__init__(**kwargs)
        self.mode = mode

    def get_ngram_set(self, x, n):
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

    @AutoRegressiveDecoder.wraps(default_rtype='probas', use_states=True)
    def predict(self, inputs, output_ids, states):
        inputs = [i for i in inputs if i[0, 0].item() > -1]  # 过滤掉无答案篇章
        topk = len(inputs[0])
        all_token_ids, all_segment_ids = [], []
        for token_ids in inputs:  # inputs里每个元素都代表一个篇章
            token_ids = torch.cat([token_ids, output_ids], 1)
            segment_ids = torch.zeros_like(token_ids)
            if states > 0:
                segment_ids[:, -output_ids.shape[1]:] = 1
            all_token_ids.extend(token_ids)
            all_segment_ids.extend(segment_ids)
        padded_all_token_ids = sequence_padding(all_token_ids)
        padded_all_segment_ids = sequence_padding(all_segment_ids)
        _, logits = model.predict([padded_all_token_ids, padded_all_segment_ids])
        probas = nn.Softmax(dim=-1)(logits)
        # 这里改成用torch.gather来做了
        # probas = [probas[i, len(ids) - 1] for i, ids in enumerate(all_token_ids)]
        # probas = torch.stack(probas).reshape((len(inputs), topk, -1))
        index_ = torch.tensor([[len(i)-1] for i in all_token_ids], device=probas.device).view(-1, 1, 1).expand(-1, 1, probas.shape[-1])
        probas = torch.gather(probas, dim=1, index=index_).reshape((len(inputs), topk, -1))
        
        if states == 0:
            # 这一步主要是排除没有答案的篇章
            # 如果一开始最大值就为end_id，那说明该篇章没有答案
            argmax = probas[:, 0].argmax(dim=1)
            available_idxs = torch.where(argmax != self.end_id)[0]
            if len(available_idxs) == 0:
                scores = torch.zeros_like(probas[0])
                scores[:, self.end_id] = 1
                return scores, states + 1
            else:
                for i in torch.where(argmax == self.end_id)[0]:
                    inputs[i][:, 0] = -1  # 无答案篇章首位标记为-1
                probas = probas[available_idxs]
                inputs = [i for i in inputs if i[0, 0] > -1]  # 过滤掉无答案篇章
        if self.mode == 'extractive':
            # 如果是抽取式，那么答案必须是篇章的一个片段
            # 那么将非篇章片段的概率值全部置0
            new_probas = torch.zeros_like(probas)
            ngrams = {}
            for token_ids in inputs:
                token_ids = token_ids[0]
                sep_idx = torch.where(token_ids == tokenizer._token_end_id)[0][0]
                p_token_ids = token_ids[1:sep_idx]
                for k, v in self.get_ngram_set(p_token_ids.cpu().numpy(), states + 1).items():  # 这里要放到.cpu().numpy()，否则会出现nrams.get不到
                    ngrams[k] = ngrams.get(k, set()) | v
            for i, ids in enumerate(output_ids):
                available_idxs = ngrams.get(tuple(ids.cpu().numpy()), set())
                available_idxs.add(tokenizer._token_end_id)
                available_idxs = list(available_idxs)
                new_probas[:, i, available_idxs] = probas[:, i, available_idxs]
            probas = new_probas
        return (probas**2).sum(0) / (probas.sum(0) + 1), states + 1  # 某种平均投票方式

    def answer(self, question, passages, topk=1):
        token_ids = []
        for passage in passages:
            passage = re.sub(u' |、|；|，', ',', passage)
            p_token_ids = tokenizer.encode(passage, maxlen=max_p_len)[0]
            q_token_ids = tokenizer.encode(question, maxlen=max_q_len + 1)[0]
            token_ids.append(p_token_ids + q_token_ids[1:])
        output_ids = self.beam_search(token_ids, topk=topk, states=0)  # 基于beam search
        return tokenizer.decode(output_ids.cpu().numpy())


reader = ReadingComprehension(
    start_id=None,
    end_id=tokenizer._token_end_id,
    maxlen=max_a_len,
    mode='extractive',
    device=device
)

def predict_to_file(data, filename, topk=1):
    """将预测结果输出到文件，方便评估
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for d in tqdm(iter(data), desc=u'正在预测(共%s条样本)' % len(data)):
            q_text = d['question']
            p_texts = [p['passage'] for p in d['passages']]
            a = reader.answer(q_text, p_texts, topk)
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
    predict_to_file(valid_dataset.data, 'qa.csv')

#! -*- coding: utf-8- -*-
# 用Seq2Seq做阅读理解构建
# 根据篇章先采样生成答案，然后采样生成问题
# 数据集同 https://github.com/bojone/dgcnn_for_reading_comprehension

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
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 基本参数
max_p_len = 128
max_q_len = 64
max_a_len = 16
batch_size = 24
epochs = 100

# bert配置
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def process_data():
    if os.path.exists('F:/Projects/data/corpus/qa/CIPS-SOGOU/train_data_list_format.json'):
        return

    # 标注数据
    webqa_data = json.load(open('F:/Projects/data/corpus/qa/WebQA.json', encoding='utf-8'))
    sogou_data = json.load(open('F:/Projects/data/corpus/qa/SogouQA.json', encoding='utf-8'))

    # 筛选数据
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    data = []
    for d in webqa_data + sogou_data:
        for p in d['passages']:
            if p['answer']:
                for t in text_segmentate(p['passage'], max_p_len - 2, seps, strips):
                    if p['answer'] in t:
                        data.append((t, d['question'], p['answer']))

    del webqa_data
    del sogou_data

    # 保存一个随机序（供划分valid用）
    random_order = list(range(len(data)))
    np.random.seed(2022)
    np.random.shuffle(random_order)

    # 划分valid
    train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
    valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]
    json.dump(train_data, open('F:/Projects/data/corpus/qa/CIPS-SOGOU/train_data_list_format.json', 'w'), indent=4)
    json.dump(valid_data, open('F:/Projects/data/corpus/qa/CIPS-SOGOU/valid_data_list_format.json', 'w'), indent=4)

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
    """单条样本格式：[CLS]篇章[SEP]答案[SEP]问题[SEP]
    """
    batch_token_ids, batch_segment_ids = [], []
    for (p, q, a) in batch:
        p_token_ids, _ = tokenizer.encode(p, maxlen=max_p_len + 1)
        a_token_ids, _ = tokenizer.encode(a, maxlen=max_a_len)
        q_token_ids, _ = tokenizer.encode(q, maxlen=max_q_len)
        token_ids = p_token_ids + a_token_ids[1:] + q_token_ids[1:]  # 去掉answer和question的cls位
        segment_ids = [0] * len(p_token_ids)
        segment_ids += [1] * (len(token_ids) - len(p_token_ids))
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], [batch_token_ids, batch_segment_ids]

train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/qa/CIPS-SOGOU/train_data_list_format.json'), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataset = MyDataset('F:/Projects/data/corpus/qa/CIPS-SOGOU/valid_data_list_format.json')
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

class QuestionAnswerGeneration(AutoRegressiveDecoder):
    """随机生成答案，并且通过beam search来生成问题
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = torch.cat([token_ids, output_ids], 1)
        segment_ids = torch.cat([segment_ids, torch.ones_like(output_ids, device=device)], 1)
        _, y_pred = model.predict([token_ids, segment_ids])
        return y_pred[:, -1, :]

    def generate(self, passage, topk=1, topp=0.95):
        token_ids, segment_ids = tokenizer.encode(passage, maxlen=max_p_len)
        a_ids = self.random_sample([token_ids, segment_ids], 1, topp=topp)[0]  # 基于随机采样
        token_ids += list(a_ids)
        segment_ids += [1] * len(a_ids)
        q_ids = self.beam_search([token_ids, segment_ids], topk=topk)  # 基于beam search
        return (tokenizer.decode(q_ids.cpu().numpy()), tokenizer.decode(a_ids.cpu().numpy()))


qag = QuestionAnswerGeneration(start_id=None, end_id=tokenizer._token_end_id, maxlen=max_q_len, device=device)


def predict_to_file(data, filename, topk=1):
    """将预测结果输出到文件，方便评估
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for d in tqdm(iter(data), desc=u'正在预测(共%s条样本)' % len(data)):
            q, a = qag.generate(d[0])
            s = '%s\t%s\t%s\n' % (q, a, d[0])
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

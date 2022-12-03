#! -*- coding: utf-8 -*-
# bert做Seq2Seq任务，采用UNILM方案
# 介绍链接：https://kexue.fm/archives/6933
# 数据集：https://github.com/CLUEbenchmark/CLGE 中的CSL数据集
# 补充了评测指标bleu、rouge-1、rouge-2、rouge-l

import json, os
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.snippets import sequence_padding, text_segmentate
from bert4torch.snippets import AutoRegressiveDecoder, Callback, ListDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 基本参数
maxlen = 256
batch_size = 16
epochs = 50
steps_per_epoch = None

# bert配置
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据
        单条格式：(标题, 正文)
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                title, content = l['title'], l['abst']
                D.append((title, content))
        return D


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

def collate_fn(batch):
    """单条样本格式：[CLS]文章[SEP]标题[SEP]
    """
    batch_token_ids, batch_segment_ids = [], []
    for title, content in batch:
        token_ids, segment_ids = tokenizer.encode(content, title, maxlen=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], [batch_token_ids, batch_segment_ids]

train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/seq2seq/summary/csl_title_public/csl_title_train.json'), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataset = MyDataset('F:/Projects/data/corpus/seq2seq/summary/csl_title_public/csl_title_dev.json')
test_dataset = MyDataset('F:/Projects/data/corpus/seq2seq/summary/csl_title_public/csl_title_test.json')

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

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = torch.cat([token_ids, output_ids], 1)
        segment_ids = torch.cat([segment_ids, torch.ones_like(output_ids, device=device)], 1)
        _, y_pred = model.predict([token_ids, segment_ids])
        return y_pred[:, -1, :]

    def generate(self, text, topk=1, topp=0.95):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids], topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids.cpu().numpy())


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=32, device=device)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        just_show()
        metrics = self.evaluate(valid_dataset.data)  # 评测模型
        metrics_test = self.evaluate(test_dataset.data)  # 评测模型
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            # model.save_weights('./best_model.pt')  # 保存模型
        metrics['best_bleu'] = self.best_bleu
        print('valid_data:', metrics)
        print('test_data:', metrics_test)
    
    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for title, content in tqdm(data):
            total += 1
            title = ' '.join(title).lower()
            pred_title = ' '.join(autotitle.generate(content, topk)).lower()
            if pred_title.strip():
                scores = self.rouge.get_scores(hyps=pred_title, refs=title)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(references=[title.split(' ')], hypothesis=pred_title.split(' '),
                                      smoothing_function=self.smooth)
        rouge_1, rouge_2, rouge_l, bleu = rouge_1/total, rouge_2/total, rouge_l/total, bleu/total
        return {'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l': rouge_l, 'bleu': bleu}

def just_show():
    s1 = u'抽象了一种基于中心的战术应用场景与业务,并将网络编码技术应用于此类场景的实时数据多播业务中。在分析基于中心网络与Many-to-all业务模式特性的基础上,提出了仅在中心节点进行编码操作的传输策略以及相应的贪心算法。分析了网络编码多播策略的理论增益上界,仿真试验表明该贪心算法能够获得与理论相近的性能增益。最后的分析与仿真试验表明,在这种有中心网络的实时数据多播应用中,所提出的多播策略的实时性能要明显优于传统传输策略。'
    s2 = u'普适计算环境中未知移动节点的位置信息是定位服务要解决的关键技术。在普适计算二维空间定位过程中,通过对三角形定位单元区域的误差分析,提出了定位单元布局(LUD)定理。在此基础上,对多个定位单元布局进行了研究,定义了一个新的描述定位单元中定位参考点覆盖效能的物理量——覆盖基,提出了在误差最小情况下定位单元布局的覆盖基定理。仿真实验表明定位单元布局定理能更好地满足对普适终端实时定位的需求,且具有较高的精度和最大覆盖效能。'
    for s in [s1, s2]:
        print(u'生成标题:', autotitle.generate(s))

if __name__ == '__main__':
    just_show()
    evaluator = Evaluator()

    model.fit(
        train_dataloader,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    model.load_weights('./best_model.pt')

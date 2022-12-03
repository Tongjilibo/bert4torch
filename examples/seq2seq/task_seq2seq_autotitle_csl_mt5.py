#! -*- coding: utf-8 -*-
# 微调多国语言版T5做Seq2Seq任务
# 介绍链接：https://kexue.fm/archives/7867
# 数据集：https://github.com/CLUEbenchmark/CLGE 中的CSL数据集
# 补充了评测指标bleu、rouge-1、rouge-2、rouge-l
# mt5主要特点：gated-gelu, decoder的最后的dense层独立权重，rmsnorm

import json, os
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import SpTokenizer, load_vocab
from bert4torch.snippets import sequence_padding, seed_everything
from bert4torch.snippets import AutoRegressiveDecoder, Callback, ListDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 基本参数
max_c_len = 256
max_t_len = 32
batch_size = 16
epochs = 50
steps_per_epoch = None
token_pad_ids = -100

# bert配置
config_path = 'F:/Projects/pretrain_ckpt/t5/[google_mt5_torch_base]/bert4torch_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/t5/[google_mt5_torch_base]/pytorch_model.bin'
# 下面两个config是从bert4keras中拿的，项目连接https://github.com/bojone/t5_in_bert4keras
spm_path = 'F:/Projects/pretrain_ckpt/t5/[google_mt5_bert4keras]/sentencepiece_cn.model'
keep_tokens_path = 'F:/Projects/pretrain_ckpt/t5/[google_mt5_bert4keras]/sentencepiece_cn_keep_tokens.json'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)

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


tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>')
keep_tokens = json.load(open(keep_tokens_path))

def collate_fn(batch):
    """单条样本格式：content：[CLS]文章[SEP]  tgt: [CLS]标题[SEP]
    """
    batch_content_ids, batch_titile_ids = [], []
    for title, content in batch:
        token_ids, _ = tokenizer.encode(content, maxlen=max_c_len)
        batch_content_ids.append(token_ids)

        token_ids, _ = tokenizer.encode(title, maxlen=max_t_len)
        batch_titile_ids.append([0] + token_ids)

    batch_content_ids = torch.tensor(sequence_padding(batch_content_ids, value=token_pad_ids), dtype=torch.long, device=device)
    batch_titile_ids = torch.tensor(sequence_padding(batch_titile_ids, value=token_pad_ids), dtype=torch.long, device=device)
    return [[batch_content_ids], [batch_titile_ids[:, :-1]]], batch_titile_ids[:, 1:].flatten()

train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/seq2seq/summary/csl_title_public/csl_title_train.json'), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataset = MyDataset('F:/Projects/data/corpus/seq2seq/summary/csl_title_public/csl_title_dev.json')
test_dataset = MyDataset('F:/Projects/data/corpus/seq2seq/summary/csl_title_public/csl_title_test.json')

model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='mt5.1.1',
    segment_vocab_size=0,
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    token_pad_ids=token_pad_ids,  # 也可以指定custom_attention_mask并传入attention_mask来实现
    add_trainer=True
).to(device)

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, outputs, y_true):
        _, _, y_pred = outputs
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        return super().forward(y_pred, y_true)
model.compile(loss=CrossEntropyLoss(ignore_index=token_pad_ids), optimizer=optim.Adam(model.parameters(), 1e-4))

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        # inputs中包含了[decoder_ids, encoder_hidden_state, encoder_attention_mask]
        return model.decoder.predict([output_ids] + inputs)[-1][:, -1, :]  # 保留最后一位

    def generate(self, text, topk=1):
        token_ids, _ = tokenizer.encode(text, maxlen=max_c_len)
        token_ids = torch.tensor([token_ids], device=device)
        encoder_output = model.encoder.predict([token_ids])
        output_ids = self.beam_search(encoder_output, topk=topk)  # 基于beam search
        return tokenizer.decode([int(i) for i in output_ids.cpu().numpy()])

autotitle = AutoTitle(start_id=0, end_id=tokenizer._token_end_id, maxlen=max_t_len, device=device)

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
    evaluator = Evaluator()
    print(u'生成标题:', autotitle.generate(u'中国的首都是extra0京'))  # 和huggingface的结果一致
    model.fit(
        train_dataloader,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    model.load_weights('./best_model.pt')

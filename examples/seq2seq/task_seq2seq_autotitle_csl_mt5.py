#! -*- coding: utf-8 -*-
# 微调多国语言版T5做Seq2Seq任务
# 介绍链接：https://kexue.fm/archives/7867
# 细节请看：https://github.com/bojone/t5_in_bert4keras
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
torch.manual_seed(42)

# 基本参数
max_c_len = 256
max_t_len = 32
batch_size = 16
steps_per_epoch = 1000
epochs = 10000

# bert配置
# config_path = 'F:/Projects/pretrain_ckpt/t5/[uer_t5_torch]--t5-small-chinese-cluecorpussmall/bert4torch_config.json'
# checkpoint_path = 'F:/Projects/pretrain_ckpt/t5/[uer_t5_torch]--t5-small-chinese-cluecorpussmall/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/t5/[uer_t5_torch]--t5-small-chinese-cluecorpussmall/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_path = 'E:/Github/T5-Finetuning-PyTorch/config.json'
checkpoint_path = 'E:/Github/T5-Finetuning-PyTorch/pytorch_model.bin'

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
token_dict = load_vocab(
    dict_path=dict_path,
    simplified=False,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

def collate_fn(batch):
    """单条样本格式：content：[CLS]文章[SEP]  tgt: [CLS]标题[SEP]
    """
    batch_content_ids, batch_titile_ids = [], []
    for title, content in batch:
        token_ids, _ = tokenizer.encode(content, maxlen=max_c_len)
        batch_content_ids.append(token_ids)  # src用[CLS]开头，[SEP]结尾

        token_ids, _ = tokenizer.encode(title, maxlen=max_t_len)
        batch_titile_ids.append(token_ids)  # tgt用[CLS]开头，[SEP]结尾

    batch_content_ids = torch.tensor(sequence_padding(batch_content_ids), dtype=torch.long, device=device)
    batch_titile_ids = torch.tensor(sequence_padding(batch_titile_ids), dtype=torch.long, device=device)
    return [[batch_content_ids], [batch_titile_ids[:, :-1]]], batch_titile_ids[:, 1:].flatten()

train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/文本生成/文本摘要/csl_title_public/csl_title_train.json'), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataset = MyDataset('F:/Projects/data/corpus/文本生成/文本摘要/csl_title_public/csl_title_dev.json')

model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='t5.1.0',
    segment_vocab_size=0,
    attention_scale=False,
    is_dropout=True,
    # keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
).to(device)

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, outputs, y_true):
        _, _, y_pred = outputs
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        return super().forward(y_pred, y_true)
model.compile(loss=CrossEntropyLoss(ignore_index=0), optimizer=optim.Adam(model.parameters(), 1.5e-5))

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids = inputs[0] 
        return model.predict([[token_ids], [output_ids]])[-1][:, -1, :]  # 保留最后一位

    def generate(self, text, topk=1, topp=0.95):
        token_ids, _ = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids], topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids.cpu().numpy())

autotitle = AutoTitle(start_id=tokenizer._token_start_id, end_id=tokenizer._token_end_id, maxlen=max_t_len, device=device)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        # metrics = self.evaluate(valid_dataset)  # 评测模型
        # if metrics['bleu'] > self.best_bleu:
        #     self.best_bleu = metrics['bleu']
        #     # model.save_weights('./best_model.pt')  # 保存模型
        # metrics['best_bleu'] = self.best_bleu
        # print('valid_data:', metrics)
        just_show()
    
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
                bleu += sentence_bleu(
                    references=[title.split(' ')],
                    hypothesis=pred_title.split(' '),
                    smoothing_function=self.smooth
                )
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }


def just_show():
    s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及 时 就医 。'
    s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午 ，华 住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'
    for s in [s1, s2]:
        print(u'生成标题:', autotitle.generate(s))

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

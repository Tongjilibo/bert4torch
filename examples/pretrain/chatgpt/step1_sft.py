#! -*- coding: utf-8 -*-
# instructgpt中stage1: Supervised Finetune

from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding, Callback, text_segmentate
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.snippets import AutoRegressiveDecoder, ListDataset
import json

# 基本参数
maxlen = 512
batch_size = 8
epochs = 10000
mask_prompt = False  # 是否把prompt部分mask掉

# 模型配置
root_path = 'F:/Projects/pretrain_ckpt/gpt2/[uer_gpt2_torch_base]--gpt2-chinese-cluecorpussmall/'
config_path = root_path + 'bert4torch_config.json'
checkpoint_path = root_path + 'bert4torch_pytorch_model.bin'
dict_path = root_path + 'vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer(dict_path, token_start=None, token_end=None, do_lower_case=True)  # 建立分词器

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据，并尽量分为不超过maxlen的句子
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                prompt, answer = l['prompt'], l['answer']
                D.append((prompt, answer))
        return D

def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for prompt, answer in batch:
        token_ids, segment_ids = tokenizer.encode(prompt, answer, maxlen=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids], [batch_token_ids, batch_segment_ids]

train_dataloader = DataLoader(MyDataset('./data/prompt_examples.json'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='gpt2',
    segment_vocab_size=0,
    add_trainer=True
).to(device)  # 建立模型，加载权重

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, y_pred, labels):
        '''
        y_pred: [btz, seq_len, vocab_size]
        labels: token_ids: [btz, seq_len], segment_ids: [btz, seq_len]
        '''
        y_true, y_mask = labels
        y_true = y_true[:, 1:]# 目标token_ids
        y_pred = y_pred[:, :-1, :]  # 预测序列，错开一位
        
        if mask_prompt:
            y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
            y_true = y_true*y_mask

        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.flatten()
        return super().forward(y_pred, y_true)
model.compile(loss=CrossEntropyLoss(ignore_index=0), optimizer=optim.Adam(model.parameters(), 1e-5))

class ArticleCompletion(AutoRegressiveDecoder):
    """基于随机采样的文章续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids = torch.cat([inputs[0], output_ids], 1)
        logits = model.predict([token_ids])
        return logits[:, -1, :]

    def generate(self, text, n=1, topp=0.95):
        token_ids, _ = tokenizer.encode(text)
        results = self.random_sample([token_ids], n, topp=topp)  # 基于随机采样
        return [text + tokenizer.decode(ids.cpu().numpy()) for ids in results]


article_completion = ArticleCompletion(
    start_id=None,
    end_id=511,  # 511是中文句号
    maxlen=100,
    minlen=50,
    device=device
)

def just_show():
    s1 = u'别爱我没结果'
    s2 = u'你这样会失去我的'
    for s in [s1, s2]:
        print(u'生成标题:', article_completion.generate(s))

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
        # 演示效果
        just_show()

if __name__ == '__main__':
    just_show()
    evaluator = Evaluator()

    model.fit(
        train_dataloader,
        steps_per_epoch=None,
        epochs=epochs,
        callbacks=[]
    )

else:
    model.load_weights('./best_model.pt')

#! -*- coding: utf-8 -*-
# 数据集在examples/datasets目录下
# 基于苏神的NEZHA模型做闲聊任务Finetune
# 参考苏神博客 https://kexue.fm/archives/7718
# base测试: https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_nezha_gpt_dialog.py

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from bert4torch.models import build_transformer_model
from bert4torch.snippets import AutoRegressiveDecoder, ListDataset, sequence_padding, Callback
from bert4torch.tokenizers import Tokenizer

# 一些基础配置
base_path = './pt_nezha_gpt_dialog'
config_path = os.path.join(base_path, 'config.json')
checkpoint_path = os.path.join(base_path, 'pytorch_model.bin')
dict_path = os.path.join(base_path, 'vocab.txt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

maxlen = 128
batch_size = 32
epochs = 30

# 分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 建立模型
model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='nezha',
    application='lm',
    add_trainer=True
).to(device)


# fn函数
def collate_fn(batch):
    """
    [
        ['哈哈', '哦', '你是猪', '不是']
    ]

    [CLS]text1[SEP]text2[SEP]

    """
    batch_token_ids, batch_segment_ids = [], []
    for texts in batch:
        token_ids, segment_ids = [tokenizer._token_start_id], [0]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:]
            # 这里做了截断
            if len(token_ids) + len(ids) <= maxlen:
                token_ids.extend(ids)
                segment_ids.extend([i % 2] * len(ids))

            else:
                break

        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)

    return [[batch_token_ids, batch_segment_ids]], [batch_token_ids, batch_segment_ids]


# dataloader
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        d = []

        with open(filename, 'r') as fr:
            for l in fr:
                l = json.loads(l)
                d.append(l)
        return d


# 损失函数
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, outputs, target):
        _, y_pred = outputs
        y_true, y_mask = target
        y_true = y_true[:, 1:]
        # mask指预测的token
        y_mask = y_mask[:, 1:]
        # 注意错开一位
        y_pred = y_pred[:, :-1, :]

        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = (y_true * y_mask).flatten()

        return super().forward(y_pred, y_true)


model.compile(loss=CrossEntropyLoss(ignore_index=0), optimizer=optim.Adam(model.parameters(), 2e-5))


# 解码器
class ChatBot(AutoRegressiveDecoder):
    """基于随机采样对话机器人
    """

    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = torch.concat([token_ids, output_ids], 1)
        curr_segment_ids = torch.ones_like(output_ids) - segment_ids[0, -1]
        segment_ids = torch.concat([segment_ids, curr_segment_ids], 1)
        return model.predict([token_ids, segment_ids])[-1][:, -1]

    def response(self, texts, topk=5):
        token_ids, segment_ids = [tokenizer._token_start_id], [0]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:]
            token_ids.extend(ids)
            segment_ids.extend([i % 2] * len(ids))
        results = self.random_sample([token_ids, segment_ids], 1, topk)
        return tokenizer.decode(results[0].cpu().numpy())


chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen, device=device)


def just_show():
    texts = ["你什么时候开始实习"]
    print('just show {0}'.format(chatbot.response(texts)))


class Evaluator(Callback):
    def on_epoch_end(self, steps, epoch, logs=None):
        # 保存最优
        model.save_weights('./best_nezha_dialogpt.pt')
        # 演示效果
        just_show()


if __name__ == '__main__':
    evaluator = Evaluator()

    train_dataloader = DataLoader(MyDataset('../datasets/LCCD-large-shuf.json'),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)

    model.fit(train_dataloader=train_dataloader,
              steps_per_epoch=None,
              epochs=epochs,
              callbacks=[evaluator])

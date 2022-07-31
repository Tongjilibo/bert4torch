#! -*- coding:utf-8 -*-
# 情感分析例子，利用MLM+P-tuning，目前示例是全部一起finetune未冻结
# 官方项目：https://github.com/THUDM/P-tuning
# 参考项目：https://github.com/bojone/P-tuning
# few-shot: 0.8953/0.8953

import torch
import torch.nn as nn
import numpy as np
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from torch.optim import Adam
from bert4torch.snippets import sequence_padding, ListDataset, Callback
from torch.utils.data import DataLoader
from torchinfo import summary

maxlen = 256
batch_size = 16
config_path = 'F:/Projects/pretrain_ckpt/robert/[hit_torch_base]--chinese-roberta-wwm-ext-base/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/robert/[hit_torch_base]--chinese-roberta-wwm-ext-base/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/robert/[hit_torch_base]--chinese-roberta-wwm-ext-base/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
choice = 'finetune_all'  # finetune_all finetune_few

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D

# 加载数据集
train_data = load_data('F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.train.data')
valid_data = load_data('F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.valid.data')
test_data = load_data('F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.test.data')

# 模拟标注和非标注数据
train_frac = 0.01  # 标注数据的比例
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
train_data = train_data[:num_labeled]
# train_data = train_data + unlabeled_data

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述
mask_idx = 5
desc = ['[unused%s]' % i for i in range(1, 9)]
desc.insert(mask_idx - 1, '[MASK]')
desc_ids = [tokenizer.token_to_id(t) for t in desc]
pos_id = tokenizer.token_to_id(u'很')
neg_id = tokenizer.token_to_id(u'不')


def random_masking(token_ids):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target


class MyDataset(ListDataset):
    def collate_fn(self, batch):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for text, label in batch:
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            if label != 2:
                token_ids = token_ids[:1] + desc_ids + token_ids[1:]
                segment_ids = [0] * len(desc_ids) + segment_ids
            if self.kwargs['random']:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            if label == 0:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = neg_id
            elif label == 1:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = pos_id
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
        batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
        batch_output_ids = torch.tensor(sequence_padding(batch_output_ids), dtype=torch.long, device=device)
        return [batch_token_ids, batch_segment_ids], batch_output_ids

# 加载数据集
train_dataset = MyDataset(data=train_data, random=True)
valid_dataset = MyDataset(data=valid_data, random=False)
test_dataset = MyDataset(data=test_data, random=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn) 
test_dataloader = DataLoader(test_dataset,  batch_size=batch_size, collate_fn=test_dataset.collate_fn) 

class MyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, y_preds, y_true):
        y_pred = y_preds[1]
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        loss = super().forward(y_pred, y_true.flatten())
        return loss

if choice == 'finetune_few':
    # 只训练这几个tokens权重这部分尚未调试好
    class PtuningBERT(BaseModel):
        def __init__(self):
            super().__init__()
            self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True, tie_emb_prj_weight=True, custom_attention_mask=True)
            for name, param in self.bert.named_parameters():
                if ('word_embeddings' not in name) and ('mlmDecoder' not in name):
                    param.requires_grad = False  # 冻结除了word_embedding层意外的其他层
        def forward(self, token_ids, segment_ids):
            embedding = self.bert.embeddings.word_embeddings(token_ids)
            embedding_no_grad = embedding.detach()
            mask = torch.ones(token_ids.shape[1], dtype=torch.long, device=token_ids.device)
            mask[1:9] -= 1  # 只优化id为1～8的token
            embedding[:, mask.bool()] = embedding_no_grad[:, mask.bool()]
            attention_mask = (token_ids != tokenizer._token_pad_id)
            return self.bert([embedding, segment_ids, attention_mask])
    model = PtuningBERT().to(device)
    summary(model, input_data=next(iter(train_dataloader))[0])
elif choice == 'finetune_all':
    # 全部权重一起训练
    model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True).to(device)
    summary(model, input_data=[next(iter(train_dataloader))[0]])

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=MyLoss(ignore_index=0),
    optimizer=Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=6e-4),
)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_acc = self.evaluate(valid_dataloader)
        test_acc = self.evaluate(test_dataloader)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.pt')
        print(f'valid_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}, best_val_acc: {self.best_val_acc:.4f}\n')

    @staticmethod
    def evaluate(data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = model.predict(x_true)[1]
            y_pred = y_pred[:, mask_idx, [neg_id, pos_id]].argmax(axis=1)
            y_true = (y_true[:, mask_idx] == pos_id).long()
            total += len(y_true)
            right += (y_true == y_pred).sum().item()
        return right / total


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator])
else:
    model.load_weights('best_model.pt')

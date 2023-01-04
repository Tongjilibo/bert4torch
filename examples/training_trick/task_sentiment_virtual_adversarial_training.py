#! -*- coding:utf-8 -*-
# 以文本分类为例的半监督学习，虚拟对抗训练策略
# 监督数据部分只计算监督Loss, 有监督+无监督数据计算对抗训练的Loss

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, text_segmentate, ListDataset, seed_everything, get_pool_emb, AdversarialTraining
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

maxlen = 256
batch_size = 16
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filenames):
        """加载数据，并尽量划分为不超过maxlen的句子
        """
        D = []
        seps, strips = u'\n。！？!?；;，, ', u'；;，, '
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    text, label = l.strip().split('\t')
                    for t in text_segmentate(text, maxlen - 2, seps, strips):
                        D.append((t, int(label)))
        return D

train_dataset = MyDataset(['F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.train.data'])
valid_dataset = MyDataset(['F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.valid.data'])
test_dataset = MyDataset(['F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.test.data'])

# 理论上应该收集任务领域类的无监督数据，这里用所有的监督数据来作无监督数据
unsup_dataset =  [sen for sen, _ in (train_dataset.data + valid_dataset.data + test_dataset.data)]

def collate_fn(batch):
    # batch_token_ids包含两部部分，第一部分是有监督数据，第二部分是无监督数据
    batch_token_ids, batch_labels = [[], []], []
    for text, label in batch:
        token_ids = tokenizer.encode(text, maxlen=maxlen)[0]
        batch_token_ids[0].append(token_ids)
        batch_labels.append([label])
        # 无监督部分
        unsup_text = random.choice(unsup_dataset)  # 随机挑一个无监督数据
        token_ids, _ = tokenizer.encode(unsup_text, maxlen=maxlen)
        batch_token_ids[1].append(token_ids)

    batch_token_ids = [j for i in batch_token_ids for j in i]
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return batch_token_ids, batch_labels.flatten()

# 加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn) 
test_dataloader = DataLoader(test_dataset,  batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='cls'):
        super().__init__()
        self.pool_method = pool_method
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True, segment_vocab_size=0)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.bert.configs['hidden_size'], 2)

    def forward(self, token_ids):
        hidden_states, pooling = self.bert([token_ids])
        pooled_output = get_pool_emb(hidden_states, pooling, token_ids[0].gt(0).long(), self.pool_method)
        output = self.dropout(pooled_output)
        output = self.dense(output)
        return output
model = Model().to(device)

class MyLoss(nn.Module):
    def forward(self, y_pred, y_true_sup):
        y_pred_sup = y_pred[:y_true_sup.shape[0]]  # 仅计算监督部分loss
        return F.cross_entropy(y_pred_sup, y_true_sup)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=MyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),
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
        print(f'val_acc: {val_acc:.5f}, test_acc: {test_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')

    # 定义评价函数
    def evaluate(self, data):
        total, right = 0., 0.
        for inputs, y_true in data:
            inputs = [inputs[0][:y_true.size(0)]]  # 仅计算有监督部分
            y_pred = model.predict(inputs).argmax(axis=1)
            total += len(y_true)
            right += (y_true == y_pred).sum().item()
        return right / total

if __name__ == '__main__':
    evaluator = Evaluator()
    adversarial_train = AdversarialTraining('vat', adversarial={'adv_alpha': 1})  # 虚拟对抗

    model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator, adversarial_train])
else:
    model.load_weights('best_model.pt')

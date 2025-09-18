#! -*- coding:utf-8 -*-
# 情感分类任务, 加载bert权重
# valid_acc: 94.72, test_acc: 94.11

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, DeepSpeedTrainer
from bert4torch.callbacks import Callback, Logger
from bert4torch.snippets import sequence_padding, text_segmentate, ListDataset, seed_everything, get_pool_emb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


maxlen = 256
config_path = '/mnt/e/pretrain_ckpt/google-bert/bert-base-chinese/bert4torch_config.json'
checkpoint_path = '/mnt/e/pretrain_ckpt/google-bert/bert-base-chinese/pytorch_model.bin'
dict_path = '/mnt/e/pretrain_ckpt/google-bert/bert-base-chinese/vocab.txt'
choice = 'train'  # train表示训练，infer表示推理

# 固定seed
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

def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for text, label in batch:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append([label])

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)
    return [batch_token_ids, batch_segment_ids], batch_labels.flatten()

# 定义bert上的模型结构
class Model(nn.Module):
    def __init__(self, pool_method='cls') -> None:
        super().__init__()
        self.pool_method = pool_method
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.bert.config['hidden_size'], 2)

    def forward(self, token_ids, segment_ids):
        hidden_states, pooling = self.bert([token_ids, segment_ids])
        pooled_output = get_pool_emb(hidden_states, pooling, token_ids.gt(0).long(), self.pool_method)
        output = self.dropout(pooled_output)
        output = self.dense(output)
        return output
net = Model()
model = DeepSpeedTrainer(net)
model.move_to_model_device = True

# 加载数据集
batch_size = model.config.train_micro_batch_size_per_gpu
train_dataloader = DataLoader(MyDataset(['/mnt/e/data/corpus/sentence_classification/sentiment/sentiment.train.data']), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset(['/mnt/e/data/corpus/sentence_classification/sentiment/sentiment.valid.data']), batch_size=batch_size, collate_fn=collate_fn) 
test_dataloader = DataLoader(MyDataset(['/mnt/e/data/corpus/sentence_classification/sentiment/sentiment.test.data']),  batch_size=batch_size, collate_fn=collate_fn) 

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=None,
    metrics=['accuracy']
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
            # 需要所有的RANK执行，因此不能设置self.run_callback=False
            model.save_to_checkpoint(save_dir='./ckpt/')
        if model.deepspeed_engine.local_rank == 0:
            print(f'val_acc: {val_acc:.5f}, test_acc: {test_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')

    # 定义评价函数
    def evaluate(self, data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1).cpu()
            total += len(y_true)
            right += (y_true == y_pred).sum().item()
        return right / total

def inference(texts):
    '''单条样本推理
    '''
    for text in texts:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        token_ids = torch.tensor(token_ids, dtype=torch.long)[None, :]
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)[None, :]

        logit = model.predict([token_ids, segment_ids])
        y_pred = torch.argmax(torch.softmax(logit, dim=-1)).cpu().numpy()
        print(text, ' ----> ', y_pred)

if __name__ == '__main__':
    if choice == 'train':
        evaluator = Evaluator()
        log = Logger('./ckpt/log.log')
        log.run_callback = model.deepspeed_engine.local_rank == 0
        model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator, log])
    else:
        model.load_weights('best_model.pt')
        inference(['我今天特别开心', '我今天特别生气'])

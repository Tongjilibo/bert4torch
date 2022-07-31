#! -*- coding:utf-8 -*-
# 情感分类任务: xlnet
# transformer包中tokenizer是padding在前面的
# 这里可以使用transformer的tokenizer，也可以使用SpTokenizer，注意取最后一位时候取非padding的最后一位
# valid_acc: 95.00, test_acc: 94.24


from bert4torch.tokenizers import SpTokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, text_segmentate, ListDataset, seed_everything
import torch.nn as nn
import torch
import torch.optim as optim
import random, os, numpy as np
from torch.utils.data import DataLoader

maxlen = 256
batch_size = 16
pretrain_model = 'F:/Projects/pretrain_ckpt/xlnet/[hit_torch_base]--chinese-xlnet-base/'
config_path = pretrain_model + 'bert4torch_config.json'
checkpoint_path = pretrain_model + 'pytorch_model.bin'
spm_path = pretrain_model + 'spiece.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定seed
seed_everything(42)

# 建立分词器
tokenizer = SpTokenizer(spm_path, token_start=None, token_end='<cls>')

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
    batch_token_ids, batch_labels = [], []
    for text, label in batch:
        token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
        batch_token_ids.append(token_ids)
        batch_labels.append([label])

    # 用tokenizer的pad_id来做padding
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, value=tokenizer._token_pad_id), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return batch_token_ids, batch_labels.flatten()

# 加载数据集
train_dataloader = DataLoader(MyDataset(['F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.train.data']), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset(['F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.valid.data']), batch_size=batch_size, collate_fn=collate_fn) 
test_dataloader = DataLoader(MyDataset(['F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.test.data']),  batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='xlnet', 
                                            token_pad_ids=tokenizer._token_pad_id, segment_vocab_size=0)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(768, 2)

    def forward(self, token_ids):
        last_hidden_state = self.bert([token_ids])
        # 取最后一位<cls>位的隐含层状态
        last_token_idx = token_ids.not_equal(tokenizer._token_pad_id).sum(dim=1) - 1
        last_token_idx = last_token_idx[:, None, None].expand(last_hidden_state.shape[0], 1, last_hidden_state.shape[-1])
        pooling = torch.gather(last_hidden_state, dim=1, index=last_token_idx).squeeze(1)

        output = self.dropout(pooling)
        output = self.dense(output)
        return output
model = Model().to(device)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),
    metrics=['accuracy']
)

# 定义评价函数
def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        total += len(y_true)
        right += (y_true == y_pred).sum().item()
    return right / total


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_acc = evaluate(valid_dataloader)
        test_acc = evaluate(test_dataloader)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.pt')
        print(f'val_acc: {val_acc:.5f}, test_acc: {test_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator])
else:
    model.load_weights('best_model.pt')

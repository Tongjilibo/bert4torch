#! -*- coding:utf-8 -*-
# 情感分类例子，RoPE相对位置编码
# 官方项目：https://github.com/ZhuiyiTechnology/roformer-v2
# pytorch参考项目：https://github.com/JunnYu/RoFormer_pytorch

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, text_segmentate, ListDataset
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

maxlen = 128
batch_size = 16
config_path = 'F:/Projects/pretrain_ckpt/roformer/[sushen_torch_base]--roformer_v2_char_base/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/roformer/[sushen_torch_base]--roformer_v2_char_base/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/roformer/[sushen_torch_base]--roformer_v2_char_base/vocab.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], batch_labels.flatten()

# 加载数据集
train_dataloader = DataLoader(MyDataset(['E:/Github/bert4torch/examples/datasets/sentiment/sentiment.train.data']), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset(['E:/Github/bert4torch/examples/datasets/sentiment/sentiment.valid.data']), batch_size=batch_size, collate_fn=collate_fn) 
test_dataloader = DataLoader(MyDataset(['E:/Github/bert4torch/examples/datasets/sentiment/sentiment.test.data']),  batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        # 指定好model=nezha和对应的ckpt地址
        self.bert, self.config = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='roformer_v2', return_model_config=True)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.config['hidden_size'], 2)

    def forward(self, token_ids, segment_ids):
        last_hidden_state = self.bert([token_ids, segment_ids])
        output = self.dropout(last_hidden_state[:, 0, :])
        output = self.dense(output)
        return output
model = Model().to(device)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),  # 用足够小的学习率
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
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.pt')
        print(f'val_acc: {val_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=20, steps_per_epoch=500, callbacks=[evaluator])
else:
    model.load_weights('best_model.pt')

#! -*- coding:utf-8 -*-
# 通过对抗训练增强模型的泛化性能
# 数据集：IFLYTEK' 长文本分类 (https://github.com/CLUEbenchmark/CLUE)
# 博客：https://kexue.fm/archives/7234

import json
from bert4torch.models import build_transformer_model, BaseModel
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from bert4torch.snippets import sequence_padding, Callback, ListDataset
from bert4torch.tokenizers import Tokenizer
from tqdm import tqdm
from torchinfo import summary

num_classes = 119
maxlen = 128
batch_size = 32

# BERT base
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据
        单条格式: (文本, 标签id)
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for i, l in enumerate(f):
                l = json.loads(l)
                text, label = l['sentence'], l['label']
                D.append((text, int(label)))
        return D


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

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

# 转换数据集
train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/文本分类/CLUEdataset/iflytek/train.json'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/文本分类/CLUEdataset/iflytek/dev.json'), batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.bert, self.config = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, return_model_config=True)
        self.dense = nn.Linear(self.config['hidden_size'], num_classes)

    def forward(self, token_ids, segment_ids):
        encoded_layers = self.bert([token_ids, segment_ids])
        output = self.dense(encoded_layers[:, 0, :])  # 取第1个位置
        return output
model = Model().to(device)
summary(model, input_data=next(iter(train_dataloader))[0])

model.compile(loss=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=2e-5), 
              metrics=['accuracy'], adversarial_train={'name': 'fgm'})

def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        val_acc = evaluate(valid_dataloader)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.pt')
        print(u'val_acc: %.5f, best_val_acc: %.5f\n' %(val_acc, self.best_val_acc))


def predict_to_file(in_file, out_file):
    """输出预测结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            text = l['sentence']
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            label = model.predict([[token_ids], [segment_ids]])[0].argmax()
            l = json.dumps({'id': str(l['id']), 'label': str(label)})
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(train_dataloader, epochs=50, steps_per_epoch=None, callbacks=[evaluator])

else: 

    model.load_weights('best_model.pt')
    # predict_to_file('/root/CLUE-master/baselines/CLUEdataset/iflytek/test.json', 'iflytek_predict.json')

#! -*- coding:utf-8 -*-
# 文本分类例子下的模型压缩
# 方法为BERT-of-Theseus
# 论文：https://arxiv.org/abs/2002.02925
# 博客：https://kexue.fm/archives/7575

import json
from bert4torch.models import build_transformer_model, BaseModel, BERT
from bert4torch.snippets import sequence_padding, Callback, ListDataset
from bert4torch.tokenizers import Tokenizer
from bert4torch.layers import BertLayer
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchinfo import summary
import copy
from torch.distributions.bernoulli import Bernoulli

num_classes = 119
maxlen = 128
batch_size = 32
replacing_rate = 0.5
steps_for_replacing = 2000

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
train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/sentence_classification/CLUEdataset/iflytek/train.json'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/sentence_classification/CLUEdataset/iflytek/dev.json'), batch_size=batch_size, collate_fn=collate_fn) 

class BERT_THESEUS(BERT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layer = BertLayer(self.hidden_size, self.num_attention_heads, self.dropout_rate, self.attention_probs_dropout_prob, self.intermediate_size, self.hidden_act, is_dropout=False, conditional_size=self.conditional_size)
        self.encoderLayer = nn.ModuleList(nn.ModuleList([copy.deepcopy(layer) for _ in range(self.num_hidden_layers)]))
        self.scc_n_layer = 6  # 蒸馏到6层
        self.scc_layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.scc_n_layer)])
        self.compress_ratio = self.num_hidden_layers // self.scc_n_layer
        self.bernoulli = None

    def set_replacing_rate(self, replacing_rate):
        if not 0 < replacing_rate <= 1:
            raise Exception('Replace rate must be in the range (0, 1]!')
        self.bernoulli = Bernoulli(torch.tensor([replacing_rate]))

    def apply_main_layers(self, **model_kwargs):
        """BERT的主体是基于Self-Attention的模块
        顺序:Att --> Add --> LN --> FFN --> Add --> LN
        v0.2.8以后输入输出是以字典形式，这里进行修改
        """
        hidden_states, attention_mask, conditional_emb = model_kwargs['hidden_states'], model_kwargs['attention_mask'], model_kwargs['conditional_emb']
        encoded_layers = [hidden_states] # 添加embedding的输出

        if self.training:
            inference_layers = []
            for i in range(self.scc_n_layer):
                if self.bernoulli.sample() == 1:  # REPLACE
                    inference_layers.append(self.scc_layer[i])
                else:  # KEEP the original
                    for offset in range(self.compress_ratio):
                        inference_layers.append(self.encoderLayer[i * self.compress_ratio + offset])

        else:  # inference with compressed model
            inference_layers = self.scc_layer

        # forward
        for i, layer_module in enumerate(inference_layers):
            outputs = layer_module(hidden_states, attention_mask, conditional_emb)
            hidden_states = outputs['hidden_states']
            model_kwargs.update(outputs)
            if self.output_all_encoded_layers:
                encoded_layers.append(hidden_states)
        if not self.output_all_encoded_layers:
            encoded_layers.append(hidden_states)
        model_kwargs['encoded_layers'] = encoded_layers
        return model_kwargs

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model=BERT_THESEUS)
        self.dense = nn.Linear(self.bert.configs['hidden_size'], num_classes)

    def forward(self, token_ids, segment_ids):
        encoded_layers = self.bert([token_ids, segment_ids])
        output = self.dense(encoded_layers[:, 0, :])  # 取第1个位置
        return output
model = Model().to(device)
summary(model, input_data=next(iter(train_dataloader))[0])

# replacing策略
class ConstantReplacementScheduler:
    def __init__(self, bert_encoder, replacing_rate, replacing_steps=None):
        self.bert_encoder = bert_encoder
        self.replacing_rate = replacing_rate
        self.replacing_steps = replacing_steps
        self.step_counter = 0
        self.bert_encoder.set_replacing_rate(replacing_rate)

    def step(self):
        self.step_counter += 1
        if self.replacing_steps is None or self.replacing_rate == 1.0:
            return self.replacing_rate
        else:
            if self.step_counter >= self.replacing_steps:
                self.bert_encoder.set_replacing_rate(1.0)
                self.replacing_rate = 1.0
            return self.replacing_rate

class LinearReplacementScheduler:
    def __init__(self, bert_encoder, base_replacing_rate, k):
        self.bert_encoder = bert_encoder
        self.base_replacing_rate = base_replacing_rate
        self.step_counter = 0
        self.k = k
        self.bert_encoder.set_replacing_rate(base_replacing_rate)

    def step(self):
        self.step_counter += 1
        current_replacing_rate = min(self.k * self.step_counter + self.base_replacing_rate, 1.0)
        self.bert_encoder.set_replacing_rate(current_replacing_rate)
        return current_replacing_rate

replacing_rate_scheduler = ConstantReplacementScheduler(bert_encoder=model.bert, replacing_rate=replacing_rate, replacing_steps=steps_for_replacing)
model.compile(loss=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=2e-5), scheduler=replacing_rate_scheduler,
              metrics=['accuracy'])


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

    model.fit(train_dataloader, epochs=50, callbacks=[evaluator])

else: 

    model.load_weights('best_model.pt')
    # predict_to_file('/root/CLUE-master/baselines/CLUEdataset/iflytek/test.json', 'iflytek_predict.json')

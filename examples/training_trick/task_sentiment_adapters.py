#! -*- coding:utf-8 -*-
# 情感分类任务, 加载bert权重
# valid_acc: 94.72, test_acc: 94.11

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, text_segmentate, ListDataset, seed_everything, get_pool_emb
from bert4torch.activations import get_activation
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


maxlen = 256
batch_size = 16
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], batch_labels.flatten()


# 加载数据集
train_dataloader = DataLoader(
    MyDataset(['F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.train.data']),
    batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(
    MyDataset(['F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.valid.data']),
    batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(
    MyDataset(['F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.test.data']), batch_size=batch_size,
    collate_fn=collate_fn)

class BottleneckAdapterLayer(nn.Module):
    def __init__(self, adapter_input_size, bottleneck_size, adapter_non_linearity='gelu'):
        super().__init__()
        self.adapter_input_size = adapter_input_size
        self.bottleneck_size = bottleneck_size
        self.non_linearity = get_activation(adapter_non_linearity)

        # down proj
        self.down_proj = nn.Linear(self.adapter_input_size, self.bottleneck_size)
        # up proj
        self.up_proj = nn.Linear(self.bottleneck_size, self.adapter_input_size)

        self.init_weights()

    def init_weights(self, init_mean=0.0, init_std=0.01):
        self.down_proj.weight.data.normal_(mean=init_mean, std=init_std)
        self.down_proj.bias.data.zero_()
        self.up_proj.weight.data.normal_(mean=init_mean, std=init_std)
        self.up_proj.bias.data.zero_()

    def forward(self, x):
        output = self.up_proj(self.non_linearity(self.down_proj(x)))
        output = x + output
        return output
    
def add_adapter(model, adapter_method='bottleneck', **kwargs):
    """
    使模型可用adapter模式进行训练
    """
    num_trainable_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 冻结模型参数
    for param in model.parameters():
        param.requires_grad = False

    if adapter_method == 'bottleneck':
        bottlenect_size = kwargs.get('bottlenect_size', 64)
        # 顺序为: Attention --> Adapter --> Add --> LN --> FeedForward --> Adapter --> Add --> LayerNorm
        for layer_id in range(model.num_hidden_layers):
            transformer_layer = model.encoderLayer[layer_id].multiHeadAttention.o
            out_featuers = transformer_layer.out_features
            adapter1 = BottleneckAdapterLayer(out_featuers, bottleneck_size=bottlenect_size)
            model.encoderLayer[layer_id].dropout1 = nn.Sequential(transformer_layer, adapter1)

            transformer_layer = model.encoderLayer[layer_id].feedForward
            out_featuers = transformer_layer.outputDense.out_features
            adapter2 = BottleneckAdapterLayer(out_featuers, bottleneck_size=bottlenect_size)
            model.encoderLayer[layer_id].feedForward = nn.Sequential(transformer_layer, adapter2)

    # 待新增其余类型adapter
    else:
        pass

    num_trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable params: {0}({1:.2f}%)'.format(num_trainable_params_after,
                                                   100 * num_trainable_params_after / num_trainable_params_before))
    return model


# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='cls') -> None:
        super().__init__()
        self.pool_method = pool_method
        bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True)
        # adapter 模式
        self.bert = add_adapter(bert)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.bert.configs['hidden_size'], 2)

    def forward(self, token_ids, segment_ids):
        hidden_states, pooling = self.bert([token_ids, segment_ids])
        pooled_output = get_pool_emb(hidden_states, pooling, token_ids.gt(0).long(), self.pool_method)
        output = self.dropout(pooled_output)
        output = self.dense(output)
        return output

model = Model().to(device)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),
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
            # model.save_weights('best_model.pt')
        print(f'val_acc: {val_acc:.5f}, test_acc: {test_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')

    # 定义评价函数
    def evaluate(self, data):
        total, right = 0., 0.
        for x_true, y_true in data:
            y_pred = model.predict(x_true).argmax(axis=1)
            total += len(y_true)
            right += (y_true == y_pred).sum().item()
        return right / total


def inference(texts):
    """单条样本推理
    """
    for text in texts:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)[None, :]
        segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=device)[None, :]

        logit = model.predict([token_ids, segment_ids])
        y_pred = torch.argmax(torch.softmax(logit, dim=-1)).cpu().numpy()
        print(text, ' ----> ', y_pred)


if __name__ == '__main__':
    if choice == 'train':
        evaluator = Evaluator()
        model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator])
    else:
        model.load_weights('best_model.pt')
        inference(['我今天特别开心', '我今天特别生气'])

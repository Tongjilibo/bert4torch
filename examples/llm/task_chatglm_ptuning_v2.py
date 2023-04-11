#! -*- coding: utf-8 -*-
# chatglm的指令微调, 还在调试中

from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding, Callback, text_segmentate
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from bert4torch.models import build_transformer_model, BaseModel
from transformers import AutoTokenizer
from bert4torch.snippets import AutoRegressiveDecoder, ListDataset, SeqGeneration
import json

# 基本参数
max_source_length = 64
max_target_length = 64
batch_size = 1
epochs = 4

# 模型配置
dir_path = "F:/Projects/pretrain_ckpt/chatglm/6B"
config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = [dir_path + f'/bert4torch_pytorch_model_{i}.bin' for i in range(1,9)]  # 可加载单个，也可以加载多个
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path.replace('/', '\\'), trust_remote_code=True)

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
                prompt, answer = l['content'], l['summary']
                D.append((prompt, answer))
        return D

def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for prompt, answer in batch:
        token_ids = tokenizer.encode(prompt, maxlen=max_source_length, truncation=True)[0]
        labels = tokenizer(answer, max_length=max_target_length, truncation=True)[0]
        batch_token_ids.append(token_ids+labels)
        batch_labels.append([-100]*len(token_ids)+labels)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels), dtype=torch.long, device=device)
    return [batch_token_ids], [batch_token_ids]

train_dataloader = DataLoader(MyDataset('./data/prompt_examples.json'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, config.num_layers * config.hidden_size * 2)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_layers * config.hidden_size * 2)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
    
class Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='glm')
        for param in self.parameters():
            param.requires_grad = False
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(self.configs)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, token_ids, segment_ids):
        hidden_state, pool_cls, seq_logit = self.encoder([token_ids, segment_ids])
        return
model = Model().to(device)

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
        

        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.flatten()
        return super().forward(y_pred, y_true)
model.compile(loss=CrossEntropyLoss(ignore_index=0), optimizer=optim.Adam(model.parameters(), 1e-5))

class Chat(SeqGeneration):
    def pre_process(self, text):
        return [tokenizer.encode(text)]
    def post_process(self, input_text, output_ids):
        return tokenizer.decode(output_ids[0].cpu().numpy())
generation = Chat(model, tokenizer, start_id=None, end_id=tokenizer.encode(['<eop>'])[0], mode='random_sample',
                  maxlen=2048, default_rtype='logits', use_states=True)

def just_show():
    s1 = u'别爱我没结果'
    s2 = u'你这样会失去我的'
    for s in [s1, s2]:
        print(u'生成标题:', generation.generate(s))

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

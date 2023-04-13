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
pre_seq_len = 128
max_seq_length = max_source_length + max_target_length
ignore_pad_token_for_loss = True
batch_size = 2
epochs = 4
prefix = ''
prompt_column = 'content'
response_column = 'summary'
history_column = None

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
                prompt, response = l[prompt_column], l[response_column]
                history = l.get('history_column', None)
                D.append((prompt, response, history))
        return D

def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for query, answer, history in batch:
        if history_column is None:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, answer) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, answer)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

        prompt = prefix + prompt
        a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
        b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

        if len(a_ids) > max_source_length - 1:
            a_ids = a_ids[:max_source_length - 1]

        if len(b_ids) > max_target_length - 2:
            b_ids = b_ids[:max_target_length - 2]

        input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
        context_length = input_ids.index(tokenizer.bos_token_id)
        mask_position = context_length - 1
        labels = [-100] * context_length + input_ids[mask_position+1:]
        batch_token_ids.append(input_ids)
        batch_labels.append(labels)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, value=tokenizer.pad_token_id), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels, value=tokenizer.pad_token_id), dtype=torch.long, device=device)
    return [batch_token_ids], [batch_labels]

train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/prompt/AdvertiseGen/train.json'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
dev_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/prompt/AdvertiseGen/dev.json'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

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
                torch.nn.Linear(config.hidden_size, config.num_hidden_layers * config.hidden_size * 2)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * config.hidden_size * 2)

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
        self.encoder = build_transformer_model(config_path=config_path, checkpoint_path=None, model='glm', token_pad_ids=tokenizer.pad_token_id, num_hidden_layers=3).half().quantize(4)
        self.config = self.encoder.configs
        self.config.pre_seq_len = 128
        self.config.prefix_projection = False
        for param in self.parameters():
            param.requires_grad = False
        self.prefix_tokens = torch.arange(self.config.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(self.config)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, token_ids):
        batch_size = token_ids.shape[0]
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(token_ids.device)
        past_key_values = self.prefix_encoder(prefix_tokens).type(torch.float16)
        past_key_values = past_key_values.view(
            batch_size,
            self.config.pre_seq_len,
            self.config.num_hidden_layers * 2,
            self.config.num_attention_heads,
            self.config.hidden_size // self.config.num_attention_heads
        )
        # b, seq_len, nh, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 1, 3, 4]).split(2)
        past_key_values = [(v[0], v[1]) for v in past_key_values]

        logits = self.encoder([token_ids], past_key_values=past_key_values)
        return
model = Model().to(device)

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, y_pred, labels):
        '''
        y_pred: [btz, seq_len, vocab_size]
        labels: token_ids: [btz, seq_len]
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
    evaluator = Evaluator()

    model.fit(
        train_dataloader,
        steps_per_epoch=None,
        epochs=epochs,
        callbacks=[]
    )

else:
    model.load_weights('./best_model.pt')

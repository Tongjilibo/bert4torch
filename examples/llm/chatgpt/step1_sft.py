#! -*- coding: utf-8 -*-
# instructgpt中stage1: Supervised Finetune

from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding, text_segmentate, ListDataset
from bert4torch.callbacks import Callback, Logger
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.generation import SeqGeneration
import json
from glob import glob


# 基本参数
max_source_length = 256
max_target_length = 256
maxlen = max_source_length + max_target_length
batch_size = 8
epochs = 10000
mask_prompt = False  # 是否把prompt部分mask掉

# 模型配置
data_path = 'E:/Github/MedicalGPT/data/finetune/**/*.txt'
root_path = 'E:/pretrain_ckpt/bloom/bloom-560m/'
config_path = root_path + 'bert4torch_config.json'
checkpoint_path = root_path + 'bert4torch_pytorch_model.bin'
dict_path = root_path + 'vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer(dict_path, token_start=None, token_end=None, do_lower_case=True)  # 建立分词器

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
                prompt, answer = l['prompt'], l['answer']
                D.append((prompt, answer))
        return D

def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for prompt, answer in batch:
        token_ids, segment_ids = tokenizer.encode(prompt, answer, maxlen=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids], [batch_token_ids, batch_segment_ids]

train_dataloader = DataLoader(MyDataset(glob(data_path, recursive=True)), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='gpt2',
    segment_vocab_size=0,
    add_trainer=True
).to(device)  # 建立模型，加载权重

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
        
        if mask_prompt:
            y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
            y_true = y_true*y_mask

        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.flatten()
        return super().forward(y_pred, y_true)
model.compile(loss=CrossEntropyLoss(ignore_index=0), optimizer=optim.Adam(model.parameters(), 1e-5))

generation = SeqGeneration(model, tokenizer, start_id=None, end_id=tokenizer.eos_token_id, mode='random_sample', tokenizer_config=tokenizer_config,
                           maxlen=maxlen, default_rtype='logits', use_states=True)

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

if __name__ == '__main__':
    logger = Logger('./log_sft.log')
    evaluator = Evaluator()
    model.fit(train_dataloader, steps_per_epoch=None, epochs=epochs, callbacks=[evaluator, logger])

else:
    model.load_weights('./best_model.pt')

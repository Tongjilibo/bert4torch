#! -*- coding:utf-8 -*-
# 搜狐2022实体情感分类Top1方案复现，https://www.biendata.xyz/competition/sohu_2022/
# 链接：https://zhuanlan.zhihu.com/p/533808475
# 复现方案：类似Prompt，拼接方案：[CLS]+sentence+[SEP]+ent1+[MASK]+ent2+[MASK]+[SEP]，取[MASK]位置进行

import numpy as np
import json
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from bert4torch.snippets import sequence_padding, Callback, ListDataset, text_segmentate, seed_everything
from bert4torch.optimizers import get_linear_schedule_with_warmup
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# 配置设置
config_path = 'F:/Projects/pretrain_ckpt/robert/[hit_torch_base]--chinese-roberta-wwm-ext-base/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/robert/[hit_torch_base]--chinese-roberta-wwm-ext-base/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/robert/[hit_torch_base]--chinese-roberta-wwm-ext-base/vocab.txt'
data_dir = 'E:/Github/Sohu2022/Sohu2022_data/nlp_data'

choice = 'train'
prefix = f'_char_512'
save_path = f'./section1{prefix}.txt'
save_path_dev = f'./dev{prefix}.txt'
ckpt_path = f'./best_model{prefix}.pt'
device = f'cuda' if torch.cuda.is_available() else 'cpu'
use_swa = True

# 模型设置
epochs = 10
steps_per_epoch = None
total_eval_step = None
num_warmup_steps = 4000
maxlen = 512
batch_size = 7
batch_size_eval = 64
categories = [-2, -1, 0, 1, 2]

seed_everything(42) # 估计随机数

# 加载数据集
def load_data(filename):
    D = []
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    with open(filename, encoding='utf-8') as f:
        for l in tqdm(f.readlines(), desc="Loading data"):
            taskData = json.loads(l.strip())
            text2 = ''.join([ent+'[MASK]' for ent in taskData['entity'].keys()]) + '[SEP]'
            text2_len = sum([len(ent)+1 for ent in taskData['entity'].keys()]) + 1
            for t in text_segmentate(taskData['content'], maxlen-text2_len-2, seps, strips):
                D.append((t, text2, taskData['entity']))
    return D

def search(tokens, start_idx=0):
    mask_idxs = []
    for i in range(len(tokens)):
        if tokens[i] == '[MASK]':
            mask_idxs.append(i+start_idx)
    return mask_idxs


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def collate_fn(batch):
    batch_token_ids, batch_entity_ids, batch_entity_labels = [], [], []
    for text1, text2, entity in batch:
        token_ids1 = tokenizer.encode(text1)[0]
        tokens2 = tokenizer.tokenize(text2)[1:-1]
        token_ids2 = tokenizer.tokens_to_ids(tokens2)
        ent_ids_raw = search(tokens2, start_idx=len(token_ids1))
        # 不在原文中的实体，其[MASK]标记不用于计算loss
        ent_labels, ent_ids = [], []
        for i, (ent, label) in enumerate(entity.items()):
            if ent in text1:
                assert tokens2[ent_ids_raw[i]-len(token_ids1)] == '[MASK]'
                ent_ids.append(ent_ids_raw[i])
                ent_labels.append(categories.index(label))

        batch_token_ids.append(token_ids1 + token_ids2)
        batch_entity_ids.append(ent_ids)
        batch_entity_labels.append(ent_labels)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_entity_ids = torch.tensor(sequence_padding(batch_entity_ids), dtype=torch.long, device=device)
    batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, value=-1), dtype=torch.long, device=device)  # [btz, 实体个数]
    return [batch_token_ids, batch_entity_ids], batch_entity_labels

# 转换数据集
all_data = load_data(f'{data_dir}/train.txt')
split_index = int(len(all_data)*0.9)
train_dataloader = DataLoader(ListDataset(data=all_data[:split_index]), batch_size=batch_size, collate_fn=collate_fn) 
valid_dataloader = DataLoader(ListDataset(data=all_data[split_index:]), batch_size=batch_size_eval, collate_fn=collate_fn)

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0)
        hidden_size = self.bert.configs['hidden_size']
        self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 5)
                )

    def forward(self, inputs):
        token_ids, entity_ids = inputs[0], inputs[1]
        last_hidden_state = self.bert([token_ids])  # [btz, seq_len, hdsz]

        hidden_size = last_hidden_state.shape[-1]
        entity_ids = entity_ids.unsqueeze(2).repeat(1, 1, hidden_size)
        entity_states = torch.gather(last_hidden_state, dim=1, index=entity_ids)
        entity_logits = self.classifier(entity_states)
        return entity_logits
model = Model().to(device)

class Loss(nn.CrossEntropyLoss):
    def forward(self, entity_logit, labels):
        loss = super().forward(entity_logit.reshape(-1, entity_logit.shape[-1]), labels.flatten())
        return loss
optimizer = optim.Adam(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps=len(train_dataloader)*epochs, last_epoch=-1)
model.compile(loss=Loss(ignore_index=-1), optimizer=optimizer, scheduler=scheduler, adversarial_train={'name': 'fgm'})
# swa
if use_swa:
    def average_function(ax: torch.Tensor, x: torch.Tensor, num: int) -> torch.Tensor:
        return ax + (x - ax) / (num + 1)
    swa_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=average_function)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        f1, acc, pred_result = self.evaluate(valid_dataloader)
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(ckpt_path)
        print(f'[val-entity] f1: {f1:.5f}, acc: {acc:.5f} best_f1: {self.best_val_f1:.5f}\n')
        if use_swa:
            swa_model.update_parameters(model)

    @staticmethod
    def evaluate(data):
        valid_true, valid_pred = [], []
        eval_step = 0
        result = dict()
        for (token_ids, entity_ids), entity_labels in tqdm(data):
            if use_swa:
                swa_model.eval()
                with torch.no_grad():
                    entity_logit = F.softmax(swa_model([token_ids, entity_ids]), dim=-1)  # [btz, 实体个数, 实体类别数]
            else:
                entity_logit = F.softmax(model.predict([token_ids, entity_ids]), dim=-1)  # [btz, 实体个数, 实体类别数]
            _, entity_pred = torch.max(entity_logit, dim=-1)  # [btz, 实体个数]
            # v_pred和v_true是实体的预测结果
            valid_index = (entity_ids.flatten()>0).nonzero().squeeze(-1)
            valid_pred.extend(entity_pred.flatten()[valid_index].cpu().tolist())
            valid_true.extend(entity_labels.flatten()[valid_index].cpu().tolist())
                
            eval_step += 1
            if (total_eval_step is not None) and (eval_step >= total_eval_step):
                break
        
        valid_true = np.array(valid_true)
        valid_pred = np.array(valid_pred)
        f1 = f1_score(valid_true, valid_pred, average='macro')
        acc = accuracy_score(valid_true, valid_pred)
        print(classification_report(valid_true, valid_pred))
        # 只保留label，不需要prob
        for k, v in result.items():
            result[k] = {i: j[0] for i, j in v.items()}
        return f1, acc, result

if __name__ == '__main__':
    if choice == 'train':
        evaluator = Evaluator()
        model.fit(train_dataloader, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[evaluator])

    model.load_weights(ckpt_path)
    f1, acc, pred_result = Evaluator.evaluate(valid_dataloader)

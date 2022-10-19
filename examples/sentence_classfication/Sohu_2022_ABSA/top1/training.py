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
from bert4torch.tokenizers import Tokenizer, SpTokenizer
from bert4torch.models import build_transformer_model, BaseModel
from tqdm import tqdm
import transformers
import random
from sklearn.metrics import f1_score, classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# 配置设置
pretrain_model = 'F:/Projects/pretrain_ckpt/xlnet/[hit_torch_base]--chinese-xlnet-base'
config_path = pretrain_model + '/bert4torch_config.json'
checkpoint_path = pretrain_model + '/pytorch_model.bin'
data_dir = 'E:/Github/Sohu2022/Sohu2022_data/nlp_data'

choice = 'train'
prefix = f'_char_512'
save_path = f'./section1{prefix}.txt'
save_path_dev = f'./dev{prefix}.txt'
ckpt_path = f'./best_model{prefix}.pt'
device = f'cuda' if torch.cuda.is_available() else 'cpu'
use_swa = False
use_adv_train = False

# 模型设置
epochs = 10
steps_per_epoch = None
total_eval_step = None
num_warmup_steps = 4000
maxlen = 900
batch_size = 6
batch_size_eval = 64
grad_accumulation_steps = 3
categories = [-2, -1, 0, 1, 2]
mask_symbol = '<mask>'

seed_everything(19260817) # 估计随机数

# 加载数据集
def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in tqdm(f.readlines(), desc="Loading data"):
            taskData = json.loads(l.strip())
            text2 = ''.join([ent+mask_symbol for ent in taskData['entity'].keys()])
            D.append((taskData['content'], text2, taskData['entity']))
    return D

def search(tokens, search_token, start_idx=0):
    mask_idxs = []
    for i in range(len(tokens)):
        if tokens[i] == search_token:
            mask_idxs.append(i+start_idx)
    return mask_idxs


# 建立分词器，这里使用transformer自带的
tokenizer = transformers.XLNetTokenizerFast.from_pretrained(pretrain_model)

def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_entity_ids, batch_entity_labels = [], [], [], []
    for text, prompt, entity in batch:
        inputs = tokenizer.__call__(text=text, text_pair=prompt, add_special_tokens=True, max_length=maxlen, truncation="only_first")
        token_ids, segment_ids = inputs['input_ids'], inputs['token_type_ids']
        ent_ids = search(token_ids, tokenizer.mask_token_id)

        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_entity_ids.append(ent_ids)
        batch_entity_labels.append([categories.index(label) for label in entity.values()])

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_entity_ids = torch.tensor(sequence_padding(batch_entity_ids), dtype=torch.long, device=device)
    batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, value=-1), dtype=torch.long, device=device)  # [btz, 实体个数]
    return [batch_token_ids, batch_segment_ids, batch_entity_ids], batch_entity_labels

# 转换数据集
all_data = load_data(f'{data_dir}/train.txt')
random.shuffle(all_data)
split_index = 2000 # int(len(all_data)*0.9)
train_dataloader = DataLoader(ListDataset(data=all_data[split_index:]), batch_size=batch_size, shuffle=False, collate_fn=collate_fn) 
valid_dataloader = DataLoader(ListDataset(data=all_data[:split_index]), batch_size=batch_size_eval, collate_fn=collate_fn)

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__() 
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='xlnet')
        hidden_size = self.bert.configs['hidden_size']
        self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, 5)
                )

    def forward(self, inputs):
        token_ids, segment_ids, entity_ids = inputs
        last_hidden_state = self.bert([token_ids, segment_ids])  # [btz, seq_len, hdsz]

        entity_ids = entity_ids.unsqueeze(2).repeat(1, 1, last_hidden_state.shape[-1])
        entity_states = torch.gather(last_hidden_state, dim=1, index=entity_ids)
        entity_logits = self.classifier(entity_states)
        return entity_logits
model = Model().to(device)

class Loss(nn.CrossEntropyLoss):
    def forward(self, entity_logit, labels):
        loss = super().forward(entity_logit.reshape(-1, entity_logit.shape[-1]), labels.flatten())
        return loss
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps=len(train_dataloader)*epochs, last_epoch=-1)
model.compile(loss=Loss(ignore_index=-1), optimizer=optimizer, scheduler=scheduler, clip_grad_norm=1.0, 
               grad_accumulation_steps=grad_accumulation_steps, adversarial_train={'name': 'fgm' if use_adv_train else ''})

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

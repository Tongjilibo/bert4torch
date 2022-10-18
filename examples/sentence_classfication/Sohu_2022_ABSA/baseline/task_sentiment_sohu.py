#! -*- coding:utf-8 -*-
# 搜狐2022实体情感分类baseline，https://www.biendata.xyz/competition/sohu_2022/
# 方案：用实体在句子中首次出现的首尾平均池化，fgm + multi_dropout + cv，f1=0.67176
import numpy as np
import random
import json
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from bert4torch.snippets import sequence_padding, Callback, ListDataset, text_segmentate
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.losses import FocalLoss
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score
import random
import os
import argparse
import pickle
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='交叉验证')
parser.add_argument('--fileid', default=0)
parser.add_argument('--gpuid', default=0)
args = parser.parse_args()
fileid = args.fileid
gpuid = args.gpuid

# 配置设置
config_path = 'F:/Projects/pretrain_ckpt/robert/[hit_torch_base]--chinese-roberta-wwm-ext-base/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/robert/[hit_torch_base]--chinese-roberta-wwm-ext-base/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/robert/[hit_torch_base]--chinese-roberta-wwm-ext-base/vocab.txt'
data_dir = 'E:/Github/Sohu2022/Sohu2022_data/nlp_data'

choice = 'train'
prefix = f'_char_512_cv_{fileid}'
save_path = f'./output/section1{prefix}.txt'
save_path_dev = f'./output/dev{prefix}.txt'
ckpt_path = f'./ckpt/best_model{prefix}.pt'
device = f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu'
seed = 42

# 模型设置
epochs = 10
steps_per_epoch = None
total_eval_step = None
maxlen = 512
batch_size = 7
batch_size_eval = 64
categories = [-2, -1, 0, 1, 2]
categories_count = {k+1:0 for k in range(len(categories))}


# 固定seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# 加载数据集
class MyDataset(ListDataset):
    def __init__(self, file_path=None, data=None, mode='train'):
        self.mode = mode
        super().__init__(file_path, data)

    def load_data(self, filename):
        D = []
        seps, strips = u'\n。！？!?；;，, ', u'；;，, '
        with open(filename, encoding='utf-8') as f:
            for l in tqdm(f, desc=f'[Loading {self.mode} data]'):
                taskData = json.loads(l.strip())
                id = taskData['id']
                # 按照最长长度和标点符号切分
                for t in text_segmentate(taskData['content'], maxlen - 2, seps, strips):                       
                    entitys = []
                    # train
                    if isinstance(taskData['entity'], dict):
                        for ent, label in taskData['entity'].items():
                            start = self.search(ent, t)
                            if start != -1:
                                label = categories.index(label)+1
                                entitys.append((ent, start, start+len(ent)-1, label)) # +1是为了padding
                                categories_count[label] += 1
                    # test
                    elif isinstance(taskData['entity'], list):
                        for ent in taskData['entity']:
                            start = self.search(ent, t)
                            if start != -1:
                                entitys.append((ent, start, start+len(ent)-1, 0))
                    if entitys:  # 如果存在实体
                        D.append((id, t, *entitys))
        return D
    
    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def collate_fn(batch):
    batch_extra, batch_token_ids, batch_entity_ids, batch_entity_labels = [], [], [], []
    for d in batch:
        id, contents, entities = d[0], d[1], d[2:]
        tokens = tokenizer.tokenize(contents, maxlen=maxlen)[1:-1]
        tokens = ['[CLS]'] + [j for i in tokens for j in i] + ['[SEP]'] # 转成char为单位的
        mapping = tokenizer.rematch(contents, tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)
        entity_ids, entity_labels, extra_map = [], [], {}
        for ent, start, end, label in entities:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                entity_ids.append([start, end])
                # # 验证边界id没有问题
                # if ''.join(tokenizer.ids_to_tokens(token_ids[start:end+1])) != ent.lower():
                #     print(''.join(tokenizer.ids_to_tokens(token_ids[start:end+1])), ent)
                entity_labels.append(label)
                extra_map[(start, end)] = (ent, label)

        if not entity_ids:  # 至少要有一个标签
            entity_ids.append([0, 0])  # 如果没有则用0填充
            entity_labels.append(0)

        batch_extra.append((id, extra_map))
        batch_token_ids.append(token_ids)
        batch_entity_ids.append(entity_ids)
        batch_entity_labels.append(entity_labels)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_entity_ids = torch.tensor(sequence_padding(batch_entity_ids), dtype=torch.long, device=device)  # [btz, 实体个数，start/end]
    batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels), dtype=torch.long, device=device)  # [btz, 实体个数]
    return [batch_token_ids, batch_entity_ids, batch_extra], batch_entity_labels

# 转换数据集
train_dataloader = DataLoader(MyDataset(f'{data_dir}/train_{fileid}.txt'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset(f'{data_dir}/dev_{fileid}.txt', mode='dev'), batch_size=batch_size_eval, collate_fn=collate_fn) 
test_dataloader = DataLoader(MyDataset(f'{data_dir}/test.txt', mode='test'), batch_size=batch_size_eval, collate_fn=collate_fn) 


# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0)
        self.dropout = [nn.Dropout(0.1), nn.Dropout(0.3), nn.Dropout(0.5), nn.Dropout(0.7)]
        self.dense = nn.Linear(768, 5+1)  # 包含padding

    def forward(self, inputs):
        token_ids, entity_ids = inputs[0], inputs[1]
        last_hidden_state = self.bert([token_ids])  # [btz, seq_len, hdsz]

        btz, entity_count, _ = entity_ids.shape
        hidden_size = last_hidden_state.shape[-1]
        entity_ids = entity_ids.reshape(btz, -1, 1).repeat(1, 1, hidden_size)
        entity_states = torch.gather(last_hidden_state, dim=1, index=entity_ids).reshape(btz, entity_count, -1, hidden_size)
        entity_states = torch.mean(entity_states, dim=2)  # 取实体首尾hidden_states的均值
        entity_logits = []
        for dropout in self.dropout:
            entity_logits.append(self.dense(dropout(entity_states)))
        return entity_logits


model = Model().to(device)

print(categories_count)
class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = FocalLoss(ignore_index=0)

    def forward(self, entity_logits, labels):
        loss = 0
        for entity_logit in entity_logits:
            loss += self.loss_fn(entity_logit.reshape(-1, entity_logit.shape[-1]), labels.flatten())
        return loss

model.compile(loss=Loss(), optimizer=optim.Adam(model.parameters(), lr=1e-5), adversarial_train={'name': 'fgm'})

def evaluate(data):
    valid_true, valid_pred = [], []
    eval_step = 0
    result, result_prob = dict(), dict()
    for (token_ids, entity_ids, extra), entity_labels in tqdm(data):
        entity_logit = model.predict([token_ids, entity_ids])[0]  # [btz, 实体个数, 实体类别数]
        entity_logit = F.softmax(entity_logit, dim=-1)
        entity_prob, entity_pred = torch.max(entity_logit, dim=-1)  # [btz, 实体个数]
        # v_pred和v_true是实体的预测结果，entity_tuple是(smp_id, ent_id, start, end, label, prob)的列表
        v_pred, entity_tuple = trans_entity2tuple(entity_ids, entity_pred, entity_prob)
        v_true, _ = trans_entity2tuple(entity_ids, entity_labels)
        valid_pred.extend(v_pred)
        valid_true.extend(v_true)

        # generate submit result
        for id_, ent_id_, start, end, label_, prob in entity_tuple:
            label_ = label_-3
            smp_id, s_e_ents = extra[id_][0], extra[id_][1]

            if (start, end) not in s_e_ents:
                raise ValueError('entity missing')
            if smp_id not in result:
                result[smp_id], result_prob[smp_id] = {}, {}
            ent_name = s_e_ents[(start, end)][0]
            if ent_name in result[smp_id] and prob < result[smp_id][ent_name][-1]:
                # 如果同一个实体
                continue
            else:
                result[smp_id].update({ent_name: (label_, prob)})
                ent_prob = entity_logit[id_][ent_id_].cpu().numpy()
                result_prob[smp_id].update({ent_name: ent_prob})
                assert prob == ent_prob[label_+3]
                

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
    return f1, acc, result, result_prob

def trans_entity2tuple(entity_ids, entity_labels, entity_probs=None):
    '''把tensor转为(样本id, start, end, 实体类型, 实体概率值)的tuple用于计算指标
    '''
    y, ent_tuple = [], []
    for i, one_sample in enumerate(entity_ids):  # 遍历样本
        for j, item in enumerate(one_sample):  # 遍历实体
            if item[0].item() * item[1].item() != 0:
                tmp = (i, j, item[0].item(), item[1].item(), entity_labels[i, j].item())
                y.append(entity_labels[i, j].item())
                ent_tuple.append(tmp if entity_probs is None else tmp + (entity_probs[i, j].item(),))
    return y, ent_tuple


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        f1, acc, pred_result, pred_result_prob = evaluate(valid_dataloader)
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(ckpt_path)
            # save_result(pred_result, pred_result_prob, save_path=save_path_dev)
        print(f'[val-entity] f1: {f1:.5f}, acc: {acc:.5f} best_f1: {self.best_val_f1:.5f}\n')

def save_result(result, result_prob, save_path):
    result = [(key, value) for key, value in result.items()]
    result.sort(key=lambda x: x[0])
    result_str = 'id\tresult\n'
    for key, value in result:
        result_str += f'{key}\t{value}\n'
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(result_str)
    # 保存概率
    with open(save_path[:-4] + '_prob.pkl', 'wb') as f:
        pickle.dump(result_prob, f)

if __name__ == '__main__':
    if choice == 'train':
        evaluator = Evaluator()
        model.fit(train_dataloader, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[evaluator])

    model.load_weights(ckpt_path)
    f1, acc, pred_result, pred_result_prob = evaluate(test_dataloader)
    save_result(pred_result, pred_result_prob, save_path=save_path)
#! -*- coding: utf-8 -*-
# ESimCSE 中文测试
# |     solution    |   ATEC  |  BQ  |  LCQMC  |  PAWSX  |  STS-B  |
# |      ESimCSE    |  34.05  | 50.54|  71.58  |  12.53  |  71.27  |

from bert4torch.snippets import sequence_padding
from tqdm import tqdm
import numpy as np
import scipy.stats
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.tokenizers import Tokenizer
from bert4torch.snippets import sequence_padding, Callback, get_pool_emb
from torch.utils.data import DataLoader
from torch import optim, nn
import torch
import random
import copy
import sys
from bert4torch.snippets import ListDataset
import jieba
jieba.initialize()

class CollateFunc(object):
    '''对句子进行复制，和抽取负对
    '''
    def __init__(self, tokenizer, max_len=256, q_size=160, dup_rate=0.15):
        self.q = []
        self.q_size = q_size
        self.max_len = max_len
        self.dup_rate = dup_rate
        self.tokenizer = tokenizer

    def word_repetition(self, batch_text, pre_tokenize=False):
        dst_text = list()
        for text in batch_text:
            if pre_tokenize:
                cut_text = jieba.cut(text, cut_all=False)
                text = list(cut_text)
            actual_len = len(text)
            dup_len = random.randint(a=0, b=max(2, int(self.dup_rate * actual_len)))
            try:
                dup_word_index = random.sample(list(range(1, actual_len)), k=dup_len)
            except:
                dup_word_index = set()

            dup_text = ''
            for index, word in enumerate(text):
                dup_text += word
                if index in dup_word_index:
                    dup_text += word
            dst_text.append(dup_text)
        return dst_text

    def negative_samples(self, batch_src_text):
        batch_size = len(batch_src_text)
        negative_samples = None
        if len(self.q) > 0:
            negative_samples = self.q[:self.q_size]
            # print("size of negative_samples", len(negative_samples))

        if len(self.q) + batch_size >= self.q_size:
            del self.q[:batch_size]
        self.q.extend(batch_src_text)

        return negative_samples

    def __call__(self, batch_text):
        '''
        input: batch_text: [batch_text,]
        output: batch_src_text, batch_dst_text, batch_neg_text
        '''
        batch_pos_text = self.word_repetition(batch_text)
        batch_neg_text = self.negative_samples(batch_text)
        # print(len(batch_pos_text))

        batch_tokens_list, batch_pos_tokens_list = [], []
        for text, text_pos in zip(batch_text, batch_pos_text):
            batch_tokens_list.append(self.tokenizer.encode(text, maxlen=maxlen)[0])
            batch_pos_tokens_list.append(self.tokenizer.encode(text_pos, maxlen=maxlen)[0])

        batch_neg_tokens_list = []
        if batch_neg_text:
            for text in batch_neg_text:
                batch_neg_tokens_list.append(self.tokenizer.encode(text, maxlen=maxlen)[0])

        batch_tokens_list = torch.tensor(sequence_padding(batch_tokens_list), dtype=torch.long, device=device)
        batch_pos_tokens_list = torch.tensor(sequence_padding(batch_pos_tokens_list), dtype=torch.long, device=device)
        
        labels = torch.arange(batch_tokens_list.size(0), device=batch_tokens_list.device)
        if batch_neg_tokens_list:
            batch_neg_tokens_list = torch.tensor(sequence_padding(batch_neg_tokens_list), dtype=torch.long, device=device)
            return [batch_tokens_list, batch_pos_tokens_list, batch_neg_tokens_list], labels
        else:
            return [batch_tokens_list, batch_pos_tokens_list], labels


# =============================基本参数=============================
model_type, pooling, task_name, dropout_rate = sys.argv[1:]  # 传入参数
# model_type, pooling, task_name, dropout_rate = 'BERT', 'cls', 'STS-B', 0.3  # debug使用
print(model_type, pooling, task_name, dropout_rate)

# 选用NEZHA和RoFormer选哟修改build_transformer_model的model参数
assert model_type in {'BERT', 'RoBERTa', 'NEZHA', 'RoFormer', 'SimBERT'}
assert pooling in {'first-last-avg', 'last-avg', 'cls', 'pooler'}
assert task_name in {'ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B'}
if model_type in {'BERT', 'RoBERTa', 'SimBERT'}:
    model_name = 'bert'
elif model_type in {'RoFormer'}:
    model_name = 'roformer'
elif model_type in {'NEZHA'}:
    model_name = 'nezha'

dropout_rate = float(dropout_rate)
batch_size = 32

if task_name == 'PAWSX':
    maxlen = 128
else:
    maxlen = 64

# bert配置
model_dir = {
    'BERT': 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12',
    'RoBERTa': 'F:/Projects/pretrain_ckpt/robert/[hit_torch_base]--chinese-roberta-wwm-ext-base',
    'NEZHA': 'F:/Projects/pretrain_ckpt/nezha/[huawei_noah_torch_base]--nezha-cn-base',
    'RoFormer': 'F:/Projects/pretrain_ckpt/roformer/[sushen_torch_base]--roformer_v1_base',
    'SimBERT': 'F:/Projects/pretrain_ckpt/simbert/[sushen_torch_base]--simbert_chinese_base',
}[model_type]

config_path = f'{model_dir}/bert_config.json' if model_type == 'BERT' else f'{model_dir}/config.json'
checkpoint_path = f'{model_dir}/pytorch_model.bin'
dict_path = f'{model_dir}/vocab.txt'
data_path = 'F:/Projects/data/corpus/sentence_embedding/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================加载数据集=============================
# 建立分词器
if model_type in ['RoFormer']:
    tokenizer = Tokenizer(dict_path, do_lower_case=True, pre_tokenize=lambda s: jieba.lcut(s, HMM=False))
else:
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

all_names = [f'{data_path}{task_name}/{task_name}.{f}.data' for f in ['train', 'valid', 'test']]
print(all_names)

def load_data(filenames):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = l.strip().split('\t')
                if len(l) == 3:
                    D.append((l[0], l[1], float(l[2])))
    return D

all_texts = load_data(all_names)
train_texts = [j for i in all_texts for j in i[:2]]

if task_name != 'PAWSX':
    np.random.shuffle(train_texts)
    train_texts = train_texts[:10000]

train_call_func = CollateFunc(tokenizer, max_len=maxlen, q_size=64, dup_rate=0.15)
train_dataloader = DataLoader(ListDataset(data=train_texts), shuffle=True, batch_size=batch_size, collate_fn=train_call_func)

def collate_fn_eval(batch):
    texts_list = [[] for _ in range(2)]
    labels = []
    for text1, text2, label in batch:
        texts_list[0].append(tokenizer.encode(text1, maxlen=maxlen)[0])
        texts_list[1].append(tokenizer.encode(text2, maxlen=maxlen)[0])
        labels.append(label)
    for i, texts in enumerate(texts_list):
        texts_list[i] = torch.tensor(sequence_padding(texts), dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.float, device=device)
    return texts_list, labels
valid_dataloader = DataLoader(ListDataset(data=all_texts), batch_size=batch_size, collate_fn=collate_fn_eval)

# 建立模型
class Model(BaseModel):
    def __init__(self, pool_method='cls', scale=20.0):
        super().__init__()
        self.pool_method = pool_method
        with_pool = 'linear' if pool_method == 'pooler' else True
        output_all_encoded_layers = True if pool_method == 'first-last-avg' else False
        self.encoder = build_transformer_model(config_path, checkpoint_path, model=model_name, segment_vocab_size=0, dropout_rate=dropout_rate,
                                               with_pool=with_pool, output_all_encoded_layers=output_all_encoded_layers)
        self.momentum_encoder = copy.deepcopy(self.encoder)
        self.scale = scale
    
    def forward(self, token_ids_list):
        reps = []
        for token_ids in token_ids_list[:2]:
            hidden_state1, pooler = self.encoder([token_ids])
            rep = get_pool_emb(hidden_state1, pooler, token_ids.gt(0).long(), self.pool_method)
            reps.append(rep)
        if len(token_ids_list) == 3:  # 负样本
            hidden_state1, pooler = self.momentum_encoder([token_ids_list[2]])
            rep = get_pool_emb(hidden_state1, pooler, token_ids.gt(0).long(), self.pool_method)
            reps.append(rep)
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])
        scores = self.cos_sim(embeddings_a, embeddings_b) * self.scale  # [btz, btz]
        return scores
    
    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pooler = self.encoder([token_ids])
            output = get_pool_emb(hidden_state, pooler, token_ids.gt(0).long(), self.pool_method)
        return output

    @staticmethod
    def cos_sim(a, b):
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

model = Model(pool_method=pooling).to(device)

class Momentum(object):
    ''' 动量更新，这里用scheduler来实现，因为是在optimizer.step()后来调用的
    '''
    def __init__(self, gamma=0.95) -> None:
        self.gamma = gamma
    def step(self):
        for encoder_param, moco_encoder_param in zip(model.encoder.parameters(), model.momentum_encoder.parameters()):
            moco_encoder_param.data = self.gamma * moco_encoder_param.data  + (1. - self.gamma) * encoder_param.data

model.compile(loss=nn.CrossEntropyLoss(), 
              optimizer=optim.Adam(model.parameters(), 1e-5),
              scheduler=Momentum(gamma=0.95))

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_consine = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_consine = evaluate(valid_dataloader)
        if val_consine > self.best_val_consine:
            self.best_val_consine = val_consine
            # model.save_weights('best_model.pt')
        print(f'val_consine: {val_consine:.5f}, best_val_consine: {self.best_val_consine:.5f}\n')

def evaluate(dataloader):
    # 模型预测
    # 标准化，相似度，相关系数
    sims_list, labels = [], []
    for (a_token_ids, b_token_ids), label in tqdm(dataloader):
        a_vecs = model.encode(a_token_ids)
        b_vecs = model.encode(b_token_ids)
        a_vecs = torch.nn.functional.normalize(a_vecs, p=2, dim=1).cpu().numpy()
        b_vecs = torch.nn.functional.normalize(b_vecs, p=2, dim=1).cpu().numpy()
        sims = (a_vecs * b_vecs).sum(axis=1)
        sims_list.append(sims)
        labels.append(label.cpu().numpy())

    corrcoef = scipy.stats.spearmanr(np.concatenate(labels), np.concatenate(sims_list)).correlation
    return corrcoef

if  __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, steps_per_epoch=None, epochs=5, callbacks=[evaluator])


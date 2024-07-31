#! -*- coding:utf-8 -*-
# 语义相似度任务-无监督
# ContrastiveTensionLoss: 同一个sentence送入两个模型，pooling后的点积要大
# |     solution    |   ATEC  |  BQ  |  LCQMC  |  PAWSX  |  STS-B  |
# |        CT       |  30.65  | 44.50|  68.67  |  16.20  |  69.27  |

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, ListDataset, get_pool_emb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import pearsonr, spearmanr
import copy
import random
from tqdm import tqdm
import numpy as np
import argparse
import jieba
jieba.initialize()


# =============================基本参数=============================
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default='BERT', choices=['BERT', 'RoBERTa', 'NEZHA', 'RoFormer', 'SimBERT'])
parser.add_argument('--pooling', default='cls', choices=['first-last-avg', 'last-avg', 'cls', 'pooler'])
parser.add_argument('--task_name', default='ATEC', choices=['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B'])
parser.add_argument('--dropout_rate', default=0.1, type=float)
args = parser.parse_args()
model_type = args.model_type
pooling = args.pooling
task_name = args.task_name
dropout_rate = args.dropout_rate

model_name = {'BERT': 'bert', 'RoBERTa': 'bert', 'SimBERT': 'bert', 'RoFormer': 'roformer', 'NEZHA': 'nezha'}[model_type]
batch_size = 32
maxlen = 128 if task_name == 'PAWSX' else 64

# bert配置
model_dir = {
    'BERT': '/data/pretrain_ckpt/bert/google@chinese_L-12_H-768_A-12',
    'RoBERTa': '/data/pretrain_ckpt/roberta/hfl@chinese-roberta-wwm-ext-base',
    'NEZHA': '/data/pretrain_ckpt/nezha/huawei_noah@nezha-cn-base',
    'RoFormer': '/data/pretrain_ckpt/roformer/sushen@roformer_v1_base',
    'SimBERT': '/data/pretrain_ckpt/simbert/sushen@simbert_chinese_base',
}[model_type]

config_path = f'{model_dir}/bert4torch_config.json' if model_type == 'BERT' else f'{model_dir}/config.json'
checkpoint_path = f'{model_dir}/pytorch_model.bin'
dict_path = f'{model_dir}/vocab.txt'
data_path = '/data/corpus/sentence_embedding/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================加载数据集=============================
# 建立分词器
if model_type in ['RoFormer']:
    tokenizer = Tokenizer(dict_path, do_lower_case=True, pre_tokenize=lambda s: jieba.lcut(s, HMM=False))
else:
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 读数据
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

# 加载训练数据集
def collate_fn(batch):
    texts_list = [[] for _ in range(2)]
    labels = []
    pos_id = random.randint(0, len(batch)-1)
    pos_token_ids, _ = tokenizer.encode(batch[pos_id], maxlen=maxlen)
    texts_list[0].append(pos_token_ids)
    texts_list[1].append(pos_token_ids)
    labels.append(1)
    for neg_id in range(len(batch)):
        if neg_id == pos_id:
            continue
        elif random.random() < 0.5:
            neg_token_ids, _ = tokenizer.encode(batch[neg_id], maxlen=maxlen)
            texts_list[0].append(pos_token_ids)
            texts_list[1].append(neg_token_ids)
            labels.append(0)
        else:
            neg_token_ids, _ = tokenizer.encode(batch[neg_id], maxlen=maxlen)
            texts_list[0].append(neg_token_ids)
            texts_list[1].append(pos_token_ids)
            labels.append(0)
    for i, texts in enumerate(texts_list):
        texts_list[i] = torch.tensor(sequence_padding(texts), dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.float, device=device)
    return texts_list, labels
train_dataloader = DataLoader(ListDataset(data=train_texts), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

# 加载测试数据集
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

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='cls'):
        super().__init__()
        with_pool = 'linear' if pool_method == 'pooler' else True
        output_all_encoded_layers = True if pool_method == 'first-last-avg' else False
        self.model1 = build_transformer_model(config_path, checkpoint_path, model=model_name, segment_vocab_size=0, dropout_rate=dropout_rate,
                                            with_pool=with_pool, output_all_encoded_layers=output_all_encoded_layers)
        self.model2 = copy.deepcopy(self.model1)
        self.pool_method = pool_method

    def forward(self, *token_ids_list):
        token_ids1 = token_ids_list[0]
        hidden_state1, pool_cls1 = self.model1([token_ids1])
        embeddings_a = get_pool_emb(hidden_state1, pool_cls1, token_ids1.gt(0).long(), self.pool_method)

        token_ids2 = token_ids_list[1]
        hidden_state2, pool_cls2 = self.model2([token_ids2])
        embeddings_b = get_pool_emb(hidden_state2, pool_cls2, token_ids2.gt(0).long(), self.pool_method)

        return torch.matmul(embeddings_a[:, None], embeddings_b[:, :, None]).squeeze(-1).squeeze(-1)  # [btz]

    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pool_cls = self.model1([token_ids])
            output = get_pool_emb(hidden_state, pool_cls, token_ids.gt(0).long(), self.pool_method)
        return output
    
model = Model(pool_method=pooling).to(device)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=nn.BCEWithLogitsLoss(reduction='mean'),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),  # 用足够小的学习率
)

# 定义评价函数
def evaluate(data):
    cosine_scores, labels = [], []
    for (batch_token1_ids, batch_token2_ids), label in tqdm(data):
        embeddings1 = model.encode(batch_token1_ids).cpu().numpy()
        embeddings2 = model.encode(batch_token2_ids).cpu().numpy()
        cosine_score = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        cosine_scores.append(cosine_score)
        labels.append(label)

    cosine_scores = np.concatenate(cosine_scores)
    labels = torch.cat(labels).cpu().numpy()
    eval_pearson_cosine, _ = spearmanr(labels, cosine_scores)
    return eval_pearson_cosine


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


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, 
            epochs=5, 
            steps_per_epoch=None, 
            callbacks=[evaluator]
            )
else:
    model.load_weights('best_model.pt')

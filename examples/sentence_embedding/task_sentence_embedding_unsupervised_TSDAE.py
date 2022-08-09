#! -*- coding:utf-8 -*-
# 语义相似度任务-无监督
# 一个encoder输入删减后的句子生成句向量，decoder依据这个句子向量来恢复原句
# |     solution    |   ATEC  |  BQ  |  LCQMC  |  PAWSX  |  STS-B  |      comment       |
# |       TSDAE     |    ——   | 46.65|  65.30  |  12.54  |    ——   | ——表示该指标异常未记录 |

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset, get_pool_emb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
import re
from tqdm import tqdm
import sys
import jieba
jieba.initialize()


# =============================基本参数=============================
model_type, pooling, task_name, dropout_rate = sys.argv[1:]  # 传入参数
# model_type, pooling, task_name, dropout_rate = 'BERT', 'cls', 'ATEC', 0.1  # debug使用
print(model_type, pooling, task_name, dropout_rate)

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
    def add_noise(token_ids, del_ratio=0.6):
        n = len(token_ids)
        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True # guarantee that at least one word remains
        return list(np.array(token_ids)[keep_or_not])

    texts_list = [[] for _ in range(3)]
    
    for text in batch:
        token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
        texts_list[0].append([tokenizer._token_start_id] + add_noise(token_ids[1:-1]) + [tokenizer._token_end_id])
        texts_list[1].append(token_ids[:-1])
        texts_list[2].append(token_ids[1:])

    for i, texts in enumerate(texts_list):
        texts_list[i] = torch.tensor(sequence_padding(texts), dtype=torch.long, device=device)
    
    return texts_list[:2], texts_list[2].flatten()
train_dataloader = DataLoader(ListDataset(data=train_texts), shuffle=True, batch_size=batch_size, collate_fn=collate_fn)

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
        self.encoder = build_transformer_model(config_path, checkpoint_path, model=model_name, segment_vocab_size=0, dropout_rate=dropout_rate,
                                               with_pool=with_pool, output_all_encoded_layers=output_all_encoded_layers)
        # 用bert的权重来初始化decoder，crossAttn部分是随机初始化的
        self.decoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model=model_name, application='lm', dropout_rate=dropout_rate, 
                                               output_all_encoded_layers=output_all_encoded_layers, is_decoder=True, segment_vocab_size=0)
        self.pool_method = pool_method

        # 绑定encoder和decoder的权重
        decoder_names = {k for k, _ in self.decoder.named_parameters()}
        for enc_k, v in self.encoder.named_parameters():
            dec_k = enc_k
            if dec_k in decoder_names:
                rep_str = f'self.encoder.{enc_k} = self.decoder.{dec_k}'
                if re.search('\.[0-9]+\.', rep_str):
                    temp = '[' + re.findall('\.[0-9]+\.', rep_str)[0][1:-1] + '].'
                    rep_str = re.sub('\.[0-9]+\.', temp, rep_str)
                exec(rep_str)
            else:
                print(enc_k, dec_k)

    def forward(self, token_ids_list):
        token_ids1 = token_ids_list[0]
        hidden_state1, pool_cls1 = self.encoder([token_ids1])
        embeddings_a = get_pool_emb(hidden_state1, pool_cls1, token_ids1.gt(0).long(), self.pool_method)

        token_ids2 = token_ids_list[1]
        encoder_embedding = embeddings_a.unsqueeze(1)
        encoder_attention_mask = torch.ones_like(token_ids1)[:, 0:1][:, None, None, :]
        _, logits = self.decoder([token_ids2, encoder_embedding, encoder_attention_mask])

        return logits.reshape(-1, logits.shape[-1])

    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pool_cls = self.encoder([token_ids])
            output = get_pool_emb(hidden_state, pool_cls, token_ids.gt(0).long(), self.pool_method)
        return output
    
model = Model(pool_method=pooling).to(device)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=nn.CrossEntropyLoss(ignore_index=0),
    optimizer=optim.Adam(model.parameters(), lr=2e-4),
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

#! -*- coding:utf-8 -*-
# bert_whitening
# 官方项目：https://github.com/bojone/BERT-whitening
# cls+不降维
# |     solution    |   ATEC  |  BQ  |  LCQMC  |  PAWSX  |  STS-B  |
# | Bert-whitening  |  26.79  | 31.81|  56.34  |  17.22  |  67.45  |

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, ListDataset, get_pool_emb
from bert4torch.layers import BERT_WHITENING
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import scipy.stats
import argparse
import jieba
jieba.initialize()


# =============================基本参数=============================
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default='BERT', choices=['BERT', 'RoBERTa', 'NEZHA', 'RoFormer', 'SimBERT'])
parser.add_argument('--pooling', default='cls', choices=['first-last-avg', 'last-avg', 'cls', 'pooler'])
parser.add_argument('--task_name', default='ATEC', choices=['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B'])
parser.add_argument('--n_components', default=-1, type=int)
args = parser.parse_args()
model_type = args.model_type
pooling = args.pooling
task_name = args.task_name
n_components = args.n_components

if n_components < 0:
    if model_type.endswith('large'):
        n_components = 1024
    elif model_type.endswith('tiny'):
        n_components = 312
    elif model_type.endswith('small'):
        n_components = 384
    else:
        n_components = 768

model_name = {'BERT': 'bert', 'RoBERTa': 'bert', 'SimBERT': 'bert', 'RoFormer': 'roformer', 'NEZHA': 'nezha'}[model_type]
batch_size = 128
maxlen = 128 if task_name == 'PAWSX' else 64

# bert配置
model_dir = {
    'BERT': 'E:/data/pretrain_ckpt/google-bert/bert-base-chinese',
    'RoBERTa': 'E:/data/pretrain_ckpt/hfl/chinese-roberta-wwm-ext',
    'NEZHA': 'E:/data/pretrain_ckpt/sijunhe/nezha-cn-base',
    'RoFormer': 'E:/data/pretrain_ckpt/junnyu/roformer_chinese_base',
    'SimBERT': 'E:/data/pretrain_ckpt/Tongjilibo/simbert-chinese-base',
}[model_type]

config_path = f'{model_dir}/bert4torch_config.json'
checkpoint_path = f'{model_dir}/pytorch_model.bin'
dict_path = f'{model_dir}/vocab.txt'
data_path = 'F:/data/corpus/sentence_embedding/'
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

class MyDataset(ListDataset):
    @staticmethod
    def load_data(filenames):
        """加载数据
        单条格式：(文本1, 文本2, 标签id)
        """
        D = []
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    l = l.strip().split('\t')
                    if len(l) == 3:
                        D.append((l[0], l[1], float(l[2])))
                    # if len(D) > 1000:
                    #     break
        return D

def collate_fn(batch):
    batch_token1_ids, batch_token2_ids, batch_labels = [], [], []
    for text1, text2, label in batch:
        token1_ids, _ = tokenizer.encode(text1, maxlen=maxlen)
        batch_token1_ids.append(token1_ids)
        token2_ids, _ = tokenizer.encode(text2, maxlen=maxlen)
        batch_token2_ids.append(token2_ids)
        batch_labels.append([label])

    batch_token1_ids = torch.tensor(sequence_padding(batch_token1_ids), dtype=torch.long, device=device)
    batch_token2_ids = torch.tensor(sequence_padding(batch_token2_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.float, device=device)
    return (batch_token1_ids, batch_token2_ids), batch_labels.flatten()

# 加载数据集
train_dataloader = DataLoader(MyDataset(all_names), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self, pool_method='mean'):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True, segment_vocab_size=0)
        self.pool_method = pool_method
   
    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pool_cls = self.bert([token_ids])
            attention_mask = token_ids.gt(0).long()
            output = get_pool_emb(hidden_state, pool_cls, attention_mask, self.pool_method)
        return output

model = Model().to(device)

# 提取训练集的所有句向量
sen_emb_list, sen_labels = [], []
for token_ids, labels in tqdm(train_dataloader, desc='Encoding'):
    sen1_emb = model.encode(token_ids[0])
    sen2_emb = model.encode(token_ids[1])
    sen_emb_list.append((sen1_emb, sen2_emb))
    sen_labels.append(labels)

# 调用bert_whitening模块
bert_whitening = BERT_WHITENING()
if n_components > 0:
    bert_whitening.compute_kernel_bias([v for vecs in sen_emb_list for v in vecs])
    bert_whitening.kernel = bert_whitening.kernel[:, :n_components]

# 变换，标准化，相似度，相关系数
all_sims = []
for (a_vecs, b_vecs) in tqdm(sen_emb_list, desc='Transform'):
    a_vecs = bert_whitening.transform_and_normalize(a_vecs)
    b_vecs = bert_whitening.transform_and_normalize(b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    all_sims.append(sims)
all_sims = torch.cat(all_sims, dim=0)
sen_labels = torch.cat(sen_labels, dim=0)
corrcoef = scipy.stats.spearmanr(sen_labels.cpu().numpy(), all_sims.cpu().numpy()).correlation
print(f'{task_name} corrcoefs: ', corrcoef)
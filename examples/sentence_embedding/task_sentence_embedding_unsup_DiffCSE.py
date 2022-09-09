#! -*- coding: utf-8 -*-
# DiffCSE中文测试：model, electra部分的gennerator和discriminator都是用的同样的bert模型
# 源项目: https://github.com/voidism/DiffCSE
# 原项目是btz *2 来做mask

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
from bert4torch.snippets import ListDataset
import torch.nn.functional as F
import sys
import jieba
jieba.initialize()


# =============================基本参数=============================
# model_type, pooling, task_name, dropout_rate = sys.argv[1:]  # 传入参数
model_type, pooling, task_name, dropout_rate = 'BERT', 'cls', 'ATEC', 0.3  # debug使用
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

lambda_weight = 0.05  # electra部分loss权重

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

def mask_tokens(inputs, special_tokens_mask=None):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    mlm_probability = 0.3
    special_tokens = {tokenizer._token_start_id, tokenizer._token_end_id, tokenizer._token_pad_id, 
                      tokenizer._token_unk_id, tokenizer._token_mask_id}

    inputs = inputs.clone()
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [[val in special_tokens for val in smp] for smp in labels.tolist()]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer._token_mask_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(tokenizer._vocab_size, labels.shape, dtype=torch.long, device=device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

# 加载训练数据集
def collate_fn(batch):
    input_ids = []
    for text in batch:
        token_ids = tokenizer.encode(text, maxlen=maxlen)[0]
        input_ids.append(token_ids)
    input_ids.extend(input_ids)
    input_ids = torch.tensor(sequence_padding(input_ids), dtype=torch.long, device=device)
    labels = torch.arange(len(batch), device=device)

    # mlm_inputs和mlm_outputs
    mlm_inputs, mlm_labels = mask_tokens(input_ids)
    attention_mask = input_ids.gt(0).long()
    return [input_ids, mlm_inputs], [labels, mlm_labels, attention_mask]
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

# 定义generator
generator = build_transformer_model(config_path, checkpoint_path, model=model_name, segment_vocab_size=0, dropout_rate=dropout_rate, with_mlm=True)
generator.to(device)
generator.eval()

class ProjectionMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        in_dim = hidden_size
        hidden_dim = hidden_size * 2
        out_dim = hidden_size
        affine=False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False),
                       nn.BatchNorm1d(hidden_dim),
                       nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False),
                        nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.record = None
        self.pos_avg = 0.0
        self.neg_avg = 0.0

    def forward(self, x, y):
        sim = self.cos(x, y)
        self.record = sim.detach()
        min_size = min(self.record.shape[0], self.record.shape[1])
        num_item = self.record.shape[0] * self.record.shape[1]
        self.pos_avg = self.record.diag().sum() / min_size
        self.neg_avg = (self.record.sum() - self.record.diag().sum()) / (num_item - min_size)
        return sim / self.temp

# 建立模型
class Model(BaseModel):
    def __init__(self, pool_method='cls'):
        super().__init__()
        self.pool_method = pool_method
        with_pool = 'linear' if pool_method == 'pooler' else True
        output_all_encoded_layers = True if pool_method == 'first-last-avg' else False
        self.bert = build_transformer_model(config_path, checkpoint_path, model=model_name, segment_vocab_size=0, dropout_rate=dropout_rate,
                                            with_pool=with_pool, output_all_encoded_layers=output_all_encoded_layers)
        self.mlp = ProjectionMLP(self.bert.configs['hidden_size'])
        self.discriminator = build_transformer_model(config_path, checkpoint_path, model=model_name, segment_vocab_size=0, dropout_rate=dropout_rate)
        self.electra_head = nn.Linear(self.bert.configs['hidden_size'], 2)
        self.sim = Similarity(temp=0.05)
    
    def forward(self, input_ids, mlm_inputs):
        # 和ESimCSE一致的计算逻辑
        attention_mask = input_ids.gt(0).long()
        hidden_state1, pooler = self.bert([input_ids])
        reps = get_pool_emb(hidden_state1, pooler, attention_mask, self.pool_method)
        if self.pool_method == 'cls':
            reps = self.mlp(reps)

        batch_size = input_ids.shape[0]//2
        embeddings_a = reps[:batch_size]
        embeddings_b = reps[batch_size:]
        scores = self.sim(embeddings_a.unsqueeze(1), embeddings_b.unsqueeze(0)) # [btz, btz]

        # Calculate loss for conditional ELECTRA
        with torch.no_grad():
            g_pred = generator([mlm_inputs])[1].argmax(-1)  # [btz, seq_len]
        g_pred[:, 0] = tokenizer._token_start_id
        e_labels = (g_pred != input_ids) * attention_mask
        e_inputs = g_pred * attention_mask
        # cls位置需要用句向量替换
        embeddings = self.discriminator.apply_embeddings([e_inputs])
        embeddings[0] = torch.cat([reps.unsqueeze(1), embeddings[0][:, 1:, :]], dim=1)
        outputs = self.discriminator.apply_main_layers(embeddings)
        mlm_outputs = self.discriminator.apply_final_layers(outputs)
        prediction_scores = self.electra_head(mlm_outputs)
        return scores, prediction_scores, e_labels
    
    def encode(self, token_ids):
        self.eval()
        with torch.no_grad():
            hidden_state, pooler = self.bert([token_ids])
            output = get_pool_emb(hidden_state, pooler, token_ids.gt(0).long(), self.pool_method)
        return output

class MyLoss(nn.Module):
    def forward(self, model_outputs, model_labels):
        scores, prediction_scores, e_labels = model_outputs
        labels, mlm_labels, attention_mask = model_labels
        # 这里不适用mlm_labels，mlm_labels主要是用于generator算loss，本方法generator是不参加训练的
        loss_simcse = F.cross_entropy(scores, labels)
        loss_electra = lambda_weight * F.cross_entropy(prediction_scores.view(-1, 2), e_labels.view(-1))
        return {'loss': loss_simcse+loss_electra, 'loss_simcse': loss_simcse, 'loss_electra': loss_electra}

def cal_metric(model_outputs, model_labels):
    scores, prediction_scores, e_labels = model_outputs
    labels, mlm_labels, attention_mask = model_labels
    rep = (e_labels == 1) * attention_mask
    fix = (e_labels == 0) * attention_mask
    prediction = prediction_scores.argmax(-1)
    result = {}
    result['electra_rep_acc'] = float((prediction*rep).sum()/rep.sum())
    result['electra_fix_acc'] = float(1.0 - (prediction*fix).sum()/fix.sum())
    result['electra_acc'] = float(((prediction == e_labels) * attention_mask).sum()/attention_mask.sum())
    return result

model = Model(pool_method=pooling).to(device)
model.compile(loss=MyLoss(), optimizer=optim.Adam(model.parameters(), 7e-6), metrics=cal_metric)

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


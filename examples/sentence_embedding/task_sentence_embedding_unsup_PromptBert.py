#! -*- coding: utf-8 -*-
# promptbert实现sentence embedding
# 官方项目：https://github.com/kongds/Prompt-BERT
# 参考项目：https://github.com/Macielyoung/sentence_representation_matching
# |     solution    |   ATEC  |  BQ  |  LCQMC  |  PAWSX  |  STS-B  |
# |    PromptBert   |  33.98  | 49.89|  73.18  |  13.30  |  73.42  |

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import ListDataset, sequence_padding, Callback
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
import numpy as np
import sys
import jieba
jieba.initialize()


# =============================基本参数=============================
model_type, task_name, dropout_rate = sys.argv[1:]  # 传入参数
# model_type, task_name, dropout_rate = 'BERT', 'ATEC', 0.3  # debug使用
print(model_type, task_name, dropout_rate)

assert model_type in {'BERT', 'RoBERTa', 'NEZHA', 'RoFormer', 'SimBERT'}
assert task_name in {'ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B'}
if model_type in {'BERT', 'RoBERTa', 'SimBERT'}:
    model_name = 'bert'
elif model_type in {'RoFormer'}:
    model_name = 'roformer'
elif model_type in {'NEZHA'}:
    model_name = 'nezha'

dropout_rate = float(dropout_rate)
batch_size = 32
template_len = 15

if task_name == 'PAWSX':
    maxlen = 128 + template_len
else:
    maxlen = 64 + template_len

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
    tokenizer = Tokenizer(dict_path, do_lower_case=True, pre_tokenize=lambda s: jieba.lcut(s, HMM=False), add_special_tokens='[X]')
else:
    tokenizer = Tokenizer(dict_path, do_lower_case=True, add_special_tokens='[X]')

replace_token = "[X]"
mask_token = "[MASK]"
prompt_templates = ['"{}" 的意思为[MASK]'.format(replace_token), '"{}"这句话的意思是[MASK]'.format(replace_token)]
tao = 0.05

token_dict = load_vocab(dict_path)
compound_tokens = [[len(token_dict)]]
token_dict['[X]'] = len(token_dict)

# 加载数据集
def load_data(filenames):
    D = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines(), desc='Load data'):
                cache = line.split('\t')
                text1, text2, label = cache[0][:maxlen-template_len], cache[1][:maxlen-template_len], cache[-1]
                for text in [text1, text2]:
                    sentence_pair = []
                    for template in prompt_templates:
                        sent_num = len(tokenizer.tokenize(text))
                        prompt_sent = template.replace(replace_token, text)
                        template_sent = template.replace(replace_token, replace_token * sent_num)
                        sentence_pair.extend([prompt_sent, template_sent])
                    D.append((sentence_pair, int(label)))
    return D

all_names = [f'{data_path}{task_name}/{task_name}.{f}.data' for f in ['train', 'valid', 'test']]
print(all_names)
train_texts = load_data(all_names)
valid_texts = list(zip(train_texts[::2], train_texts[1::2]))

if task_name != 'PAWSX':
    np.random.shuffle(train_texts)
    train_texts = train_texts[:10000]

# 加载训练数据集
def collate_fn(batch):
    batch_tensor = [[] for _ in range(4)]
    for prompt_data, _ in batch:
        for i, item in enumerate(prompt_data):
            batch_tensor[i].append(tokenizer.encode(item, maxlen=maxlen)[0])
    for i, item in enumerate(batch_tensor):
        batch_tensor[i] = torch.tensor(sequence_padding(item, maxlen), dtype=torch.long, device=device)
    labels = torch.arange(batch_tensor[0].size(0), device=device)
    return batch_tensor, labels

train_dataloader = DataLoader(ListDataset(data=train_texts), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 加载测试数据集
def collate_fn_test(batch):
    text1_ids, text2_ids, labels = [], [], []
    for text1, text2 in batch:
        label = text1[-1]
        text1, text2 = text1[0][0], text2[0][0]
        text1_ids.append(tokenizer.encode(text1, maxlen=maxlen)[0])
        text2_ids.append(tokenizer.encode(text2, maxlen=maxlen)[0])
        labels.append(label)
    text1_ids = torch.tensor(sequence_padding(text1_ids), dtype=torch.long, device=device)
    text2_ids = torch.tensor(sequence_padding(text2_ids), dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return [text1_ids, text2_ids], labels

valid_dataloader = DataLoader(ListDataset(data=valid_texts), batch_size=batch_size, collate_fn=collate_fn_test) 

# =============================定义模型=============================
class PromptBert(BaseModel):
    def __init__(self, scale=20.0):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model=model_name, 
                                            dropout_rate=dropout_rate, segment_vocab_size=0, compound_tokens=compound_tokens)
        self.scale = scale

    def forward(self, prompt0_input, template0_input, prompt1_input, template1_input):
        embeddings_a = self.get_sentence_embedding(prompt0_input, template0_input)
        embeddings_b = self.get_sentence_embedding(prompt1_input, template1_input)
        scores = self.cos_sim(embeddings_a, embeddings_b) * self.scale  # [btz, btz]
        return scores

    def get_sentence_embedding(self, prompt_input_ids, template_input_ids):
        prompt_mask_embedding = self.get_mask_embedding(prompt_input_ids)
        template_mask_embedding = self.get_mask_embedding(template_input_ids)
        # 在计算损失函数时为了消除Prompt模板影响，通过替换模板后的句子[MASK]获取的表征减去模板中[MASK]获取的表征来得到句子向量表征
        sentence_embedding = prompt_mask_embedding - template_mask_embedding
        return sentence_embedding

    def get_mask_embedding(self, input_ids):
        last_hidden_state = self.bert([input_ids])
        mask_index = (input_ids == tokenizer._token_mask_id).long()
        input_mask_expanded = mask_index.unsqueeze(-1).expand(last_hidden_state.size()).float()
        mask_embedding = torch.sum(last_hidden_state * input_mask_expanded, 1)
        return mask_embedding
    
    def predict(self, input_ids):
        self.eval()
        with torch.no_grad():
            mask_embedding = self.get_mask_embedding(input_ids)
        return mask_embedding
    
    @staticmethod
    def cos_sim(a, b):
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

model = PromptBert().to(device)
       
# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),
)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_sim = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_sim = self.evaluate(valid_dataloader)
        if val_sim > self.best_val_sim:
            self.best_val_sim = val_sim
            # model.save_weights('best_model.pt')
        print(f'val_sim: {val_sim:.5f}, best_val_sim: {self.best_val_sim:.5f}\n')
    
    @staticmethod
    def evaluate(data):
        embeddings1, embeddings2, labels = [], [], []
        for (text1_ids, text2_ids), label in data:
            embeddings1.append(model.predict(text1_ids))
            embeddings2.append(model.predict(text2_ids))
            labels.append(label)

        embeddings1 = torch.cat(embeddings1)
        embeddings2 = torch.cat(embeddings2)
        labels = torch.cat(labels)

        sims = F.cosine_similarity(embeddings1, embeddings2).cpu().numpy()
        labels = labels.cpu().numpy()
        return spearmanr(sims, labels)[0]

if __name__ == "__main__":
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=5, steps_per_epoch=None, callbacks=[evaluator])

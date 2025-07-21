#! -*- coding: utf-8 -*-
'''
promptbert实现sentence embedding
- 官方项目：https://github.com/kongds/Prompt-BERT
- 参考项目：https://github.com/Macielyoung/sentence_representation_matching

|     solution    |   ATEC  |  BQ  |  LCQMC  |  PAWSX  |  STS-B  |
|    PromptBert   |  33.98  | 49.89|  73.18  |  13.30  |  73.42  |

- 基本思路：
1. 用Prompt的方式生成句子表示
    1.1 模板1: "我想打电话"可转为prompt1:"我想打电话的意思为[MASK]"和templeate1:"[X][X][X][X][X]的意思为[MASK]"
    1.2 模板2: "我想打电话"可转为prompt2:"我想打电话这句话的意思是[MASK]"和templeate2:"[X][X][X][X][X]这句话的意思是[MASK]"
2. prompt1和templete1进入模型获取各自[MASK]位置的句向量sent1和temp1, 并使用sent1-temp1=作为句向量s1，以消除prompt模板的影响
3. prompt2和templete2进入模型获取各自[MASK]位置的句向量sent2和temp2, 并使用sent2-temp2作为句向量s2，以消除prompt模板的影响
4. 采用自监督的方式训练：同一个batch对应位置互为正样本（如s1和s2），其他为负样本
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import ListDataset, sequence_padding
from bert4torch.callbacks import Callback
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
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
template_len = 15
maxlen = template_len + (128 if task_name == 'PAWSX' else 64)

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
replace_token = "[X]"
mask_token = "[MASK]"
prompt_templates = ['"{}" 的意思为[MASK]'.format(replace_token), '"{}"这句话的意思是[MASK]'.format(replace_token)]
tao = 0.05
token_dict = load_vocab(dict_path)
token_dict[replace_token] = token_dict.pop('[unused1]')  # 替换一个token
tokenizer = Tokenizer(token_dict, do_lower_case=True, add_special_tokens='[X]',
                      pre_tokenize=(lambda s: jieba.lcut(s, HMM=False)) if model_type in ['RoFormer'] else None)

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
                                            dropout_rate=dropout_rate, segment_vocab_size=0)
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
        '''获取[MASK] token所在位置的embedding表示作为句向量'''
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
        cosine_scores, labels = [], []
        for (text1_ids, text2_ids), label in tqdm(data, desc='Evaluate'):
            embeddings1 = model.predict(text1_ids)
            embeddings2 = model.predict(text2_ids)
            cosine_score = F.cosine_similarity(embeddings1, embeddings2).cpu().numpy()
            cosine_scores.append(cosine_score)
            labels.append(label.cpu().numpy())

        labels = np.concatenate(labels)
        cosine_scores = np.concatenate(cosine_scores)
        return spearmanr(cosine_scores, labels)[0]

if __name__ == "__main__":
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=5, steps_per_epoch=None, callbacks=[evaluator])

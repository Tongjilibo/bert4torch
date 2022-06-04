#! -*- coding: utf-8 -*-
# prompt实现sentence embedding
# 参考链接（非官方）：https://gitee.com/moontheunderwater/bert4pytorch/blob/master/examples/textmatch/promptbert.py
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


learning_rate = 2.5e-5
num_train_epochs = 10
max_len = 120
batch_size = 12
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

replace_token = "[X]"
mask_token = "[MASK]"
prompt_templates = ['"{}" 的意思为[MASK]'.format(replace_token), '"{}"这句话的意思是[MASK]'.format(replace_token)]
tao = 0.05

token_dict = load_vocab(dict_path)
compound_tokens = [[len(token_dict)]]
token_dict['[X]'] = len(token_dict)

tokenizer = Tokenizer(token_dict, do_lower_case=True, add_special_tokens='[X]')

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()[:2000], desc='Load data'):
                sent = line.strip()[:max_len - 15]
                sentence_pair = []
                for template in prompt_templates:
                    sent_num = len(tokenizer.tokenize(sent))
                    prompt_sent = template.replace(replace_token, sent)
                    template_sent = template.replace(replace_token, replace_token * sent_num)
                    sentence_pair.extend([prompt_sent, template_sent])
                D.append(sentence_pair)
        return D

def collate_fn(batch):
    batch_tensor = [[] for _ in range(4)]
    for prompt_data in batch:
        for i, item in enumerate(prompt_data):
            batch_tensor[i].append(tokenizer.encode(item, maxlen=max_len)[0])

    for i, item in enumerate(batch_tensor):
        batch_tensor[i] = torch.tensor(sequence_padding(item, max_len), dtype=torch.long, device=device)
    
    labels = torch.arange(batch_tensor[0].size(0), device=device)
    return batch_tensor, labels

train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/pretrain/film/film.txt'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def load_valid_data(filename):
    D = []
    with open(filename, "r", encoding="utf-8-sig") as f:
        for line in f.readlines():
            cache = line.split('\t')
            text1, text2, label = cache[0][:max_len-15], cache[1][:max_len-15], cache[-1]
            text1 = prompt_templates[0].replace("[X]", text1)
            text2 = prompt_templates[0].replace("[X]", text2)
            D.append((text1, text2, int(label)))
    return D

def collate_fn_test(batch):
    text1_ids, text2_ids, labels = [], [], []
    for text1, text2, label in batch:
        text1_ids.append(tokenizer.encode(text1, maxlen=max_len)[0])
        text2_ids.append(tokenizer.encode(text2, maxlen=max_len)[0])
        labels.append(label)
    
    text1_ids = torch.tensor(sequence_padding(text1_ids), dtype=torch.long, device=device)
    text2_ids = torch.tensor(sequence_padding(text2_ids), dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    return [text1_ids, text2_ids], labels

valid_datset = load_valid_data('F:/Projects/data/corpus/sentence_embedding/STS-B/STS-B.test.data')
valid_dataloader = DataLoader(MyDataset(data=valid_datset), batch_size=batch_size, collate_fn=collate_fn_test) 

class Prompt(BaseModel):
    def __init__(self, scale=20.0):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0, compound_tokens=compound_tokens)
        self.scale = scale

    def forward(self, prompt0_input, template0_input, prompt1_input, template1_input):
        embeddings_a = self.get_sentence_embedding(prompt0_input, template0_input)
        embeddings_b = self.get_sentence_embedding(prompt1_input, template1_input)
        scores = self.cos_sim(embeddings_a, embeddings_b) * self.scale  # [btz, btz]
        return scores

    def get_sentence_embedding(self, prompt_input_ids, template_input_ids):
        prompt_mask_embedding = self.get_mask_embedding(prompt_input_ids)
        template_mask_embedding = self.get_mask_embedding(template_input_ids)
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

model = Prompt().to(device)
       
# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),  # 用足够小的学习率
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

        embeddings1 = torch.concat(embeddings1)
        embeddings2 = torch.concat(embeddings2)
        labels = torch.concat(labels)

        sims = F.cosine_similarity(embeddings1, embeddings2).cpu().numpy()
        labels = labels.cpu().numpy()
        return spearmanr(sims, labels)[0]

if __name__ == "__main__":
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=20, steps_per_epoch=100, callbacks=[evaluator])

#! -*- coding:utf-8 -*-
# W2NER: https://github.com/ljynlp/W2NER
# 数据集：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from bert4torch.snippets import sequence_padding, Callback, ListDataset, seed_everything
from bert4torch.optimizers import get_linear_schedule_with_warmup
from bert4torch.layers import LayerNorm
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict, deque
from sklearn.metrics import precision_recall_fscore_support

# 模型参数：训练
epochs = 20  # 训练轮数
steps_per_epoch = None  # 每轮步数
maxlen = 256  # 最大长度
batch_size = 8  # 根据gpu显存设置
learning_rate = 1e-3
bert_learning_rate = 5e-6
warm_factor = 0.1
weight_decay = 0
use_bert_last_4_layers = True
categories = {'LOC':2, 'PER':3, 'ORG':4}
label_num = len(categories) + 2

# 模型参数：网络结构
dist_emb_size = 20
type_emb_size = 20
lstm_hid_size = 512
conv_hid_size = 96
bert_hid_size = 768
biaffine_size = 512
ffnn_hid_size = 288
dilation = [1, 2, 3]
emb_dropout = 0.5
conv_dropout = 0.5
out_dropout = 0.33

# BERT base
config_path = 'F:/Projects/pretrain_ckpt/bert/[huggingface_torch_base]--bert-base-chinese/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[huggingface_torch_base]--bert-base-chinese/bert4torch_pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[huggingface_torch_base]--bert-base-chinese/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定seed
seed_everything(42)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 相对距离设置
dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9

# 用到的小函数
def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text

def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            for l in tqdm(f.split('\n\n'), desc='Load data'):
                if not l:
                    continue
                sentence, d = [], []
                for i, c in enumerate(l.split('\n')):
                    char, flag = c.split(' ')
                    sentence += char
                    if flag[0] == 'B':
                        d.append([i, i, flag[2:]])
                    elif flag[0] == 'I':
                        d[-1][1] = i
                if len(sentence) > maxlen - 2:
                    continue
                tokens = [tokenizer.tokenize(word)[1:-1] for word in sentence[:maxlen-2]]
                pieces = [piece for pieces in tokens for piece in pieces]
                tokens_ids = [tokenizer._token_start_id] + tokenizer.tokens_to_ids(pieces) + [tokenizer._token_end_id]
                assert len(tokens_ids) <= maxlen
                length = len(tokens)

                # piece和word的对应关系，中文两者一致，除了[CLS]和[SEP]
                _pieces2word = np.zeros((length, len(tokens_ids)), dtype=np.bool)
                e_start = 0
                for i, pieces in enumerate(tokens):
                    if len(pieces) == 0:
                        continue
                    pieces = list(range(e_start, e_start + len(pieces)))
                    _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                    e_start += len(pieces)

                # 相对距离
                _dist_inputs = np.zeros((length, length), dtype=np.int)
                for k in range(length):
                    _dist_inputs[k, :] += k
                    _dist_inputs[:, k] -= k

                for i in range(length):
                    for j in range(length):
                        if _dist_inputs[i, j] < 0:
                            _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                        else:
                            _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
                _dist_inputs[_dist_inputs == 0] = 19

                # golden标签
                _grid_labels = np.zeros((length, length), dtype=np.int)
                _grid_mask2d = np.ones((length, length), dtype=np.bool)

                for entity in d:
                    e_start, e_end, e_type = entity[0], entity[1]+1, entity[-1]
                    if e_end >= maxlen - 2:
                        continue
                    index = list(range(e_start, e_end))
                    for i in range(len(index)):
                        if i + 1 >= len(index):
                            break
                        _grid_labels[index[i], index[i + 1]] = 1
                    _grid_labels[index[-1], index[0]] = categories[e_type]
                _entity_text = set([convert_index_to_text(list(range(e[0], e[1]+1)), categories[e[-1]]) for e in d])
                D.append((tokens_ids, _pieces2word, _dist_inputs, _grid_labels, _grid_mask2d, _entity_text))
        return D


def collate_fn(data):
    tokens_ids, pieces2word, dist_inputs, grid_labels, grid_mask2d, _entity_text = map(list, zip(*data))

    sent_length = torch.tensor([i.shape[0] for i in pieces2word], dtype=torch.long, device=device)
    # max_wordlen: word长度，非token长度，max_tokenlen：token长度
    max_wordlen = torch.max(sent_length).item() 
    max_tokenlen = np.max([len(x) for x in tokens_ids])
    tokens_ids = torch.tensor(sequence_padding(tokens_ids), dtype=torch.long, device=device)
    batch_size = tokens_ids.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = torch.tensor(x, dtype=torch.long, device=device)
        return new_data

    dis_mat = torch.zeros((batch_size, max_wordlen, max_wordlen), dtype=torch.long, device=device)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_wordlen, max_wordlen), dtype=torch.long, device=device)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_wordlen, max_wordlen), dtype=torch.bool, device=device)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_wordlen, max_tokenlen), dtype=torch.bool, device=device)
    pieces2word = fill(pieces2word, sub_mat)

    return [tokens_ids, pieces2word, dist_inputs, sent_length, grid_mask2d], [grid_labels, grid_mask2d, _entity_text]

# 加载数据
train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/ner/china-people-daily-ner-corpus/example.train'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/ner/china-people-daily-ner-corpus/example.dev'), batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构
class ConvolutionLayer(nn.Module):
    '''卷积层
    '''
    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs


class Biaffine(nn.Module):
    '''仿射变换
    '''
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s

class MLP(nn.Module):
    '''MLP全连接
    '''
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x

class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)

        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        return o1 + o2

class Model(BaseModel):
    def __init__(self, use_bert_last_4_layers=False):
        super().__init__()
        self.use_bert_last_4_layers = use_bert_last_4_layers
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, # segment_vocab_size=0, 
                                            output_all_encoded_layers = True if use_bert_last_4_layers else False)
        lstm_input_size = self.bert.configs['hidden_size']

        self.dis_embs = nn.Embedding(20, dist_emb_size)
        self.reg_embs = nn.Embedding(3, type_emb_size)

        self.encoder = nn.LSTM(lstm_input_size, lstm_hid_size // 2, num_layers=1, batch_first=True,
                            bidirectional=True)

        conv_input_size = lstm_hid_size + dist_emb_size + type_emb_size

        self.convLayer = ConvolutionLayer(conv_input_size, conv_hid_size, dilation, conv_dropout)
        self.dropout = nn.Dropout(emb_dropout)
        self.predictor = CoPredictor(label_num, lstm_hid_size, biaffine_size,
                                     conv_hid_size * len(dilation), ffnn_hid_size, out_dropout)

        self.cln = LayerNorm(lstm_hid_size, conditional_size=lstm_hid_size)

    def forward(self, token_ids, pieces2word, dist_inputs, sent_length, grid_mask2d):
        bert_embs = self.bert([token_ids, torch.zeros_like(token_ids)])
        if self.use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[-4:], dim=-1).mean(-1)

        length = pieces2word.size(1)
        min_value = torch.min(bert_embs).item()

        # 最大池化
        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        # LSTM
        word_reps = self.dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = self.encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())

        # 条件LayerNorm
        cln = self.cln([word_reps.unsqueeze(2), word_reps])

        # concat
        dis_emb = self.dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()
        reg_emb = self.reg_embs(reg_inputs)
        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
        
        # 卷积层
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = self.convLayer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)

        # 输出层
        outputs = self.predictor(word_reps, word_reps, conv_outputs)
        return outputs

model = Model(use_bert_last_4_layers).to(device)

class Loss(nn.CrossEntropyLoss):
    def forward(self, outputs, labels):
        grid_labels, grid_mask2d, _ = labels
        grid_mask2d = grid_mask2d.clone()
        return super().forward(outputs[grid_mask2d], grid_labels[grid_mask2d])

bert_params = set(model.bert.parameters())
other_params = list(set(model.parameters()) - bert_params)
no_decay = ['bias', 'LayerNorm.weight']
params = [
    {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
        'lr': bert_learning_rate,
        'weight_decay': weight_decay},
    {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
        'lr': bert_learning_rate,
        'weight_decay': 0.0},
    {'params': other_params,
        'lr': learning_rate,
        'weight_decay': weight_decay},
]

optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
updates_total = (len(train_dataloader) if steps_per_epoch is None else steps_per_epoch) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_factor * updates_total, num_training_steps=updates_total)
model.compile(loss=Loss(), optimizer=optimizer, scheduler=scheduler, clip_grad_norm=5.0)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        f1, p, r, e_f1, e_p, e_r = self.evaluate(valid_dataloader)
        if e_f1 > self.best_val_f1:
            self.best_val_f1 = e_f1
            # model.save_weights('best_model.pt')
        print(f'[val-token  level] f1: {f1:.5f}, p: {p:.5f} r: {r:.5f}')
        print(f'[val-entity level] f1: {e_f1:.5f}, p: {e_p:.5f} r: {e_r:.5f} best_f1: {self.best_val_f1:.5f}\n')

    def evaluate(self, data_loader):
        def cal_f1(c, p, r):
            if r == 0 or p == 0:
                return 0, 0, 0
            r = c / r if r else 0
            p = c / p if p else 0
            if r and p:
                return 2 * p * r / (p + r), p, r
            return 0, p, r

        pred_result = []
        label_result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        for data_batch in tqdm(data_loader, desc='Evaluate'):
            (token_ids, pieces2word, dist_inputs, sent_length, grid_mask2d), (grid_labels, grid_mask2d, entity_text) = data_batch
            outputs = model.predict([token_ids, pieces2word, dist_inputs, sent_length, grid_mask2d])

            grid_mask2d = grid_mask2d.clone()

            outputs = torch.argmax(outputs, -1)
            ent_c, ent_p, ent_r, _ = self.decode(outputs.cpu().numpy(), entity_text, sent_length.cpu().numpy())

            total_ent_r += ent_r
            total_ent_p += ent_p
            total_ent_c += ent_c

            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(), pred_result.numpy(), average="macro")
        e_f1, e_p, e_r = cal_f1(total_ent_c, total_ent_p, total_ent_r)
        return f1, p, r, e_f1, e_p, e_r

    def decode(self, outputs, entities, length):
        class Node:
            def __init__(self):
                self.THW = []                # [(tail, type)]
                self.NNW = defaultdict(set)   # {(head,tail): {next_index}}

        ent_r, ent_p, ent_c = 0, 0, 0
        decode_entities = []
        q = deque()
        for instance, ent_set, l in zip(outputs, entities, length):
            predicts = []
            nodes = [Node() for _ in range(l)]
            count = 0
            for cur in reversed(range(l)):
                # if count >= 29:
                #     print(count)
                count += 1
                heads = []
                for pre in range(cur+1):
                    # THW
                    if instance[cur, pre] > 1: 
                        nodes[pre].THW.append((cur, instance[cur, pre]))
                        heads.append(pre)
                    # NNW
                    if pre < cur and instance[pre, cur] == 1:
                        # cur node
                        for head in heads:
                            nodes[pre].NNW[(head,cur)].add(cur)
                        # post nodes
                        for head,tail in nodes[cur].NNW.keys():
                            if tail >= cur and head <= pre:
                                nodes[pre].NNW[(head,tail)].add(cur)
                # entity
                for tail,type_id in nodes[cur].THW:
                    if cur == tail:
                        predicts.append(([cur], type_id))
                        continue
                    q.clear()
                    q.append([cur])
                    while len(q) > 0:
                        chains = q.pop()
                        for idx in nodes[chains[-1]].NNW[(cur,tail)]:
                            if idx == tail:
                                predicts.append((chains + [idx], type_id))
                            else:
                                q.append(chains + [idx])
            
            predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
            decode_entities.append([convert_text_to_index(x) for x in predicts])
            ent_r += len(ent_set)
            ent_p += len(predicts)
            ent_c += len(predicts.intersection(ent_set))
        return ent_c, ent_p, ent_r, decode_entities

if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[evaluator])
else: 
    model.load_weights('best_model.pt')

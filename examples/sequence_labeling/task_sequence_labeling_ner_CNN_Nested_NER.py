#! -*- coding:utf-8 -*-
# https://github.com/yhcc/CNN_Nested_NER


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from bert4torch.snippets import sequence_padding, Callback, ListDataset, seed_everything
from bert4torch.optimizers import get_linear_schedule_with_warmup
from bert4torch.layers import LayerNorm
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torch_scatter import scatter_max

# 模型参数：训练
epochs = 20  # 训练轮数
steps_per_epoch = None  # 每轮步数
maxlen = 256  # 最大长度
batch_size = 8  # 根据gpu显存设置
lr = 2e-5
warm_factor = 0.1
weight_decay = 1e-2
label2idx = {'LOC':0, 'PER':1, 'ORG':2}
non_ptm_lr_ratio = 100
biaffine_size = 400
n_head = 4
size_embed_dim = 25
ent_thres = 0.5
kernel_size = 3
cnn_dim = 200
logit_drop = 0
cnn_depth = 3

# BERT base
config_path = 'F:/Projects/pretrain_ckpt/bert/[huggingface_torch_base]--bert-base-chinese/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[huggingface_torch_base]--bert-base-chinese/bert4torch_pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[huggingface_torch_base]--bert-base-chinese/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定seed
seed_everything(42)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def get_new_ins(bpes, spans, indexes):
        bpes.append(tokenizer._token_end_id)
        cur_word_idx = indexes[-1]
        indexes.append(0)
        # int8范围-128~127
        matrix = np.zeros((cur_word_idx, cur_word_idx, len(label2idx)), dtype=np.int8)
        ent_target = []
        for _ner in spans:
            s, e, t = _ner
            matrix[s, e, t] = 1
            matrix[e, s, t] = 1
            ent_target.append((s, e, t))
        assert len(bpes)<=maxlen, len(bpes)
        return [bpes, indexes, matrix, ent_target]

    def load_data(self, filename):
        D = []
        word2bpes = {}
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            for l in tqdm(f.split('\n\n'), desc='Load data'):
                if not l:
                    continue
                _raw_words, _raw_ents = [], []
                for i, c in enumerate(l.split('\n')):
                    char, flag = c.split(' ')
                    _raw_words += char
                    if flag[0] == 'B':
                        _raw_ents.append([i, i, flag[2:]])
                    elif flag[0] == 'I':
                        _raw_ents[-1][1] = i
                if len(_raw_words) > maxlen - 2:
                    continue
                
                bpes = [tokenizer._token_start_id]
                indexes = [0]
                spans = []
                ins_lst = []
                _indexes = []
                _bpes = []

                for idx, word in enumerate(_raw_words, start=0):
                    if word in word2bpes:
                        __bpes = word2bpes[word]
                    else:
                        __bpes = tokenizer.encode(word)[0][1:-1]
                        word2bpes[word] = __bpes
                    _indexes.extend([idx]*len(__bpes))
                    _bpes.extend(__bpes)
                next_word_idx = indexes[-1]+1
                if len(bpes) + len(_bpes) <= maxlen:
                    bpes = bpes + _bpes
                    indexes += [i + next_word_idx for i in _indexes]
                    spans += [(s+next_word_idx-1, e+next_word_idx-1, label2idx.get(t), ) for s, e, t in _raw_ents]
                else:
                    new_ins = self.get_new_ins(bpes, spans, indexes)
                    ins_lst.append(new_ins)
                    indexes = [0] + [i + 1 for i in _indexes]
                    spans = [(s, e, label2idx.get(t), ) for s, e, t in _raw_ents]
                    bpes = [tokenizer._token_start_id] + _bpes

                D.append(self.get_new_ins(bpes, spans, indexes))
        return D


def collate_fn(data):
    tokens_ids, indexes, matrix, ent_target = map(list, zip(*data))
    tokens_ids = torch.tensor(sequence_padding(tokens_ids), dtype=torch.long, device=device)
    indexes = torch.tensor(sequence_padding(indexes), dtype=torch.long, device=device)
    seq_len = max([i.shape[0] for i in matrix])
    matrix_new = np.ones((len(tokens_ids), seq_len, seq_len, len(label2idx)), dtype=np.int8) * -100
    for i in range(len(tokens_ids)):
        matrix_new[i, :len(matrix[i][0]), :len(matrix[i][0]), :] = matrix[i]
    matrix = torch.tensor(matrix_new, dtype=torch.long, device=device)

    return [tokens_ids, indexes], [matrix, ent_target]

# 加载数据
train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/ner/china-people-daily-ner-corpus/example.train'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/ner/china-people-daily-ner-corpus/example.dev'), batch_size=batch_size, collate_fn=collate_fn) 

class MaskConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, groups=1):
        super(MaskConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False, groups=groups)

    def forward(self, x, mask):
        x = x.masked_fill(mask, 0)
        _x = self.conv2d(x)
        return _x


class MaskCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, depth=3):
        super(MaskCNN, self).__init__()

        layers = []
        for _ in range(depth):
            layers.extend([
                MaskConv2d(input_channels, input_channels, kernel_size=kernel_size, padding=kernel_size//2),
                LayerNorm((1, input_channels, 1, 1), dim_index=1),
                nn.GELU()])
        layers.append(MaskConv2d(input_channels, output_channels, kernel_size=3, padding=3//2))
        self.cnns = nn.ModuleList(layers)

    def forward(self, x, mask):
        _x = x  # 用作residual
        for layer in self.cnns:
            if isinstance(layer, LayerNorm):
                x = x + _x
                x = layer(x)
                _x = x
            elif not isinstance(layer, nn.GELU):
                x = layer(x, mask)
            else:
                x = layer(x)
        return _x


class MultiHeadBiaffine(nn.Module):
    def __init__(self, dim, out=None, n_head=4):
        super(MultiHeadBiaffine, self).__init__()
        assert dim%n_head==0
        in_head_dim = dim//n_head
        out = dim if out is None else out
        assert out%n_head == 0
        out_head_dim = out//n_head
        self.n_head = n_head
        self.W = nn.Parameter(nn.init.xavier_normal_(torch.randn(self.n_head, out_head_dim, in_head_dim, in_head_dim)))
        self.out_dim = out

    def forward(self, h, v):
        """
        :param h: bsz x max_len x dim
        :param v: bsz x max_len x dim
        :return: bsz x max_len x max_len x out_dim
        """
        bsz, max_len, dim = h.size()
        h = h.reshape(bsz, max_len, self.n_head, -1)
        v = v.reshape(bsz, max_len, self.n_head, -1)
        w = torch.einsum('blhx,hdxy,bkhy->bhdlk', h, self.W, v)
        w = w.reshape(bsz, self.out_dim, max_len, max_len)
        return w


class CNNNer(BaseModel):
    def __init__(self, num_ner_tag, cnn_dim=200, biaffine_size=200,
                 size_embed_dim=0, logit_drop=0, kernel_size=3, n_head=4, cnn_depth=3):
        super(CNNNer, self).__init__()
        self.pretrain_model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0)
        hidden_size = self.pretrain_model.configs['hidden_size']

        if size_embed_dim!=0:
            n_pos = 30
            self.size_embedding = torch.nn.Embedding(n_pos, size_embed_dim)
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos/2, -n_pos/2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos/2, n_pos/2-1) + n_pos/2
            self.register_buffer('span_size_ids', _span_size_ids.long())
            hsz = biaffine_size*2 + size_embed_dim + 2
        else:
            hsz = biaffine_size*2+2
        biaffine_input_size = hidden_size

        self.head_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(biaffine_input_size, biaffine_size),
            nn.LeakyReLU(),
        )
        self.tail_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(biaffine_input_size, biaffine_size),
            nn.LeakyReLU(),
        )

        self.dropout = nn.Dropout(0.4)
        if n_head>0:
            self.multi_head_biaffine = MultiHeadBiaffine(biaffine_size, cnn_dim, n_head=n_head)
        else:
            self.U = nn.Parameter(torch.randn(cnn_dim, biaffine_size, biaffine_size))
            torch.nn.init.xavier_normal_(self.U.data)
        self.W = torch.nn.Parameter(torch.empty(cnn_dim, hsz))
        torch.nn.init.xavier_normal_(self.W.data)
        if cnn_depth>0:
            self.cnn = MaskCNN(cnn_dim, cnn_dim, kernel_size=kernel_size, depth=cnn_depth)

        self.down_fc = nn.Linear(cnn_dim, num_ner_tag)
        self.logit_drop = logit_drop

    def forward(self, input_ids, indexes):
        last_hidden_states = self.pretrain_model([input_ids])
        state = scatter_max(last_hidden_states, index=indexes, dim=1)[0][:, 1:]  # bsz x word_len x hidden_size
        lengths, _ = indexes.max(dim=-1)

        head_state = self.head_mlp(state)
        tail_state = self.tail_mlp(state)
        if hasattr(self, 'U'):
            scores1 = torch.einsum('bxi, oij, byj -> boxy', head_state, self.U, tail_state)
        else:
            scores1 = self.multi_head_biaffine(head_state, tail_state)
        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat([self.dropout(head_state).unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                                 self.dropout(tail_state).unsqueeze(1).expand(-1, head_state.size(1), -1, -1)], dim=-1)

        if hasattr(self, 'size_embedding'):
            size_embedded = self.size_embedding(self.span_size_ids[:state.size(1), :state.size(1)])
            affined_cat = torch.cat([affined_cat, self.dropout(size_embedded).unsqueeze(0).expand(state.size(0), -1, -1, -1)], dim=-1)

        scores2 = torch.einsum('bmnh,kh->bkmn', affined_cat, self.W)  # bsz x dim x L x L
        scores = scores2 + scores1   # bsz x dim x L x L

        if hasattr(self, 'cnn'):            
            batch_size = lengths.shape[0]
            broad_cast_seq_len = torch.arange(int(lengths.max())).expand(batch_size, -1).to(lengths)
            mask = broad_cast_seq_len < lengths.unsqueeze(1)

            mask = mask[:, None] * mask.unsqueeze(-1)
            pad_mask = mask[:, None].eq(0)
            u_scores = scores.masked_fill(pad_mask, 0)
            if self.logit_drop != 0:
                u_scores = F.dropout(u_scores, p=self.logit_drop, training=self.training)
            # bsz, num_label, max_len, max_len = u_scores.size()
            u_scores = self.cnn(u_scores, pad_mask)
            scores = u_scores + scores

        scores = self.down_fc(scores.permute(0, 2, 3, 1))

        return scores

model = CNNNer(num_ner_tag=len(label2idx), cnn_dim=cnn_dim, biaffine_size=biaffine_size,
              size_embed_dim=size_embed_dim, logit_drop=logit_drop,
              kernel_size=kernel_size, n_head=n_head, cnn_depth=cnn_depth).to(device)

class Loss(object):
    def __call__(self, scores, y_true):
        matrix, _ = y_true
        assert scores.shape[-1] == matrix.shape[-1]
        flat_scores = scores.reshape(-1)
        flat_matrix = matrix.reshape(-1)
        mask = flat_matrix.ne(-100).float().view(scores.size(0), -1)
        flat_loss = F.binary_cross_entropy_with_logits(flat_scores, flat_matrix.float(), reduction='none')
        loss = ((flat_loss.view(scores.size(0), -1)*mask).sum(dim=-1)).mean()
        return loss

# optimizer
parameters = []
ln_params = []
non_ln_params = []
non_pretrain_params = []
non_pretrain_ln_params = []

for name, param in model.named_parameters():
    name = name.lower()
    if param.requires_grad is False:
        continue
    if 'pretrain_model' in name:
        if 'norm' in name or 'bias' in name:
            ln_params.append(param)
        else:
            non_ln_params.append(param)
    else:
        if 'norm' in name or 'bias' in name:
            non_pretrain_ln_params.append(param)
        else:
            non_pretrain_params.append(param)
optimizer = torch.optim.AdamW([{'params': non_ln_params, 'lr': lr, 'weight_decay': weight_decay},
                               {'params': ln_params, 'lr': lr, 'weight_decay': 0},
                               {'params': non_pretrain_ln_params, 'lr': lr*non_ptm_lr_ratio, 'weight_decay': 0},
                               {'params': non_pretrain_params, 'lr': lr*non_ptm_lr_ratio, 'weight_decay': weight_decay}])

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
            model.save_weights('best_model.pt')
        print(f'[val-token  level] f1: {f1:.5f}, p: {p:.5f} r: {r:.5f}')
        print(f'[val-entity level] f1: {e_f1:.5f}, p: {e_p:.5f} r: {e_r:.5f} best_f1: {self.best_val_f1:.5f}\n')

    def evaluate(self, data_loader, threshold=0.5):
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
            (tokens_ids, indexes), (matrix, ent_target) = data_batch
            scores = torch.sigmoid(model.predict([tokens_ids, indexes])).gt(threshold).long()
            scores = scores.masked_fill(matrix.eq(-100), 0)  # mask掉padding部分
            
            # token粒度
            mask = matrix.reshape(-1).ne(-100)
            label_result.append(matrix.reshape(-1).masked_select(mask).cpu())
            pred_result.append(scores.reshape(-1).masked_select(mask).cpu())

            # 实体粒度
            ent_c, ent_p, ent_r = self.decode(scores.cpu().numpy(), ent_target)
            total_ent_r += ent_r
            total_ent_p += ent_p
            total_ent_c += ent_c

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(), pred_result.numpy(), average="macro")
        e_f1, e_p, e_r = cal_f1(total_ent_c, total_ent_p, total_ent_r)
        return f1, p, r, e_f1, e_p, e_r

    def decode(self, outputs, ent_target):
        ent_c, ent_p, ent_r = 0, 0, 0
        for pred, label in zip(outputs, ent_target):
            ent_r += len(label)
            pred_tuple = []
            for item in range(pred.shape[-1]):
                if pred[:, :, item].sum() > 0:
                    _index = np.where(pred[:, :, item]>0)
                    tmp = [(i, j, item) if j >= i else (j, i, item) for i, j in zip(*_index)]
                    pred_tuple.extend(list(set(tmp)))
            ent_p += len(pred_tuple)
            ent_c += len(set(label).intersection(set(pred_tuple)))
            
        return ent_c, ent_p, ent_r

if __name__ == '__main__':
    if True:
        evaluator = Evaluator()
        model.fit(train_dataloader, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[evaluator])
    else: 
        model.load_weights('best_model.pt')
        evaluator = Evaluator()
        f1, p, r, e_f1, e_p, e_r = evaluator.evaluate(valid_dataloader)
        print(f'[val-token  level] f1: {f1:.5f}, p: {p:.5f} r: {r:.5f}')
        print(f'[val-entity level] f1: {e_f1:.5f}, p: {e_p:.5f} r: {e_r:.5f}\n')

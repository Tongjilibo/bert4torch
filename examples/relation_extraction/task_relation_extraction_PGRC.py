#! -*- coding:utf-8 -*-
# 三元组抽取任务，PGRC

import json
import numpy as np
from bert4torch.layers import LayerNorm
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from collections import Counter
import random


corres_threshold = 0.5
rel_threshold = 0.5
ensure_rel = False
ensure_corres = False
num_negs = 4
drop_prob = 0.3
emb_fusion = 'concat'
Label2IdxSub = {"B-H": 1, "I-H": 2, "O": 0}
Label2IdxObj = {"B-T": 1, "I-T": 2, "O": 0}


maxlen = 128
batch_size = 8
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载标签字典
predicate2id, id2predicate = {}, {}

with open('F:/Projects/data/corpus/relation_extraction/BD_Knowledge_Extraction/all_50_schemas', encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 解析样本
def get_spoes(text, spo_list):
    '''单独抽出来，这样读取数据时候，可以根据spoes来选择跳过
    '''
    def search(pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1

    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    # 整理三元组 {s: [(o, p)]}
    spoes = {}
    for s, p, o in spo_list:
        s = tokenizer.encode(s)[0][1:-1]
        p = predicate2id[p]
        o = tokenizer.encode(o)[0][1:-1]
        s_idx = search(s, token_ids)
        o_idx = search(o, token_ids)
        if s_idx != -1 and o_idx != -1:
            s = (s_idx, s_idx + len(s) - 1)
            o = (o_idx, o_idx + len(o) - 1, p)
            if s not in spoes:
                spoes[s] = []
            spoes[s].append(o)
    return token_ids, segment_ids, spoes


# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据
        单条格式：{'text': text, 'spo_list': [(s, p, o)]}
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in tqdm(f):
                l = json.loads(l)
                labels = [(spo['subject'], spo['predicate'], spo['object']) for spo in l['spo_list']]
                token_ids, segment_ids, spoes = get_spoes(l['text'], labels)
                if spoes:
                    # D.append({'text': l['text'], 'spo_list': labels, 'token_ids': token_ids, 
                    #           'segment_ids': segment_ids, 'spoes': spoes})
        
                    # construct tags of correspondence and relation
                    corres_tag = np.zeros((maxlen, maxlen))
                    rel_tag = len(predicate2id) * [0]
                    for subject, object_labels in spoes.items():
                        for object_label in object_labels:
                            # get sub and obj head
                            sub_head, obj_head, rel = subject[0], object_label[0], object_label[-1]
                            # construct relation tag
                            rel_tag[rel] = 1
                            if sub_head != -1 and obj_head != -1:
                                corres_tag[sub_head][obj_head] = 1

                    rel2ens = {}
                    # positive samples
                    for subject, object_labels in spoes.items():
                        for object_label in object_labels:
                            rel = object_label[-1]
                            object_ = (object_label[0], object_label[1])
                            rel2ens[rel] = rel2ens.get(rel, []) + [[subject, object_]]
                    
                    for rel, en_ll in rel2ens.items():
                        # init
                        tags_sub = maxlen * [Label2IdxSub['O']]
                        tags_obj = maxlen * [Label2IdxSub['O']]
                        for en in en_ll:
                            # get sub and obj head
                            sub_head, obj_head, sub_len, obj_len = en[0][0], en[1][0], en[0][-1] - en[0][0], en[1][-1] - en[1][0]
                            if sub_head != -1 and obj_head != -1:
                                if sub_head + sub_len <= maxlen:
                                    tags_sub[sub_head] = Label2IdxSub['B-H']
                                    tags_sub[sub_head + 1:sub_head + sub_len] = (sub_len - 1) * [Label2IdxSub['I-H']]
                                if obj_head + obj_len <= maxlen:
                                    tags_obj[obj_head] = Label2IdxObj['B-T']
                                    tags_obj[obj_head + 1:obj_head + obj_len] = (obj_len - 1) * [Label2IdxObj['I-T']]
                        seq_tag = [tags_sub, tags_obj]

                        # sanity check
                        D.append([token_ids, corres_tag, seq_tag, rel, rel_tag])

                    # relation judgement ablation
                    if not ensure_rel:
                        # negative samples
                        neg_rels = set(predicate2id.values()).difference(set(rel2ens.keys()))
                        neg_rels = random.sample(neg_rels, k=num_negs)
                        for neg_rel in neg_rels:
                            # init
                            seq_tag = maxlen * [Label2IdxSub['O']]
                            # sanity check
                            seq_tag = [seq_tag, seq_tag]
                            D.append([token_ids, corres_tag, seq_tag, neg_rel, rel_tag])
                # if len(D) > 1000:
                #     break
        return D

def collate_fn(data):
    token_ids, corres_tags, seq_tags, rel, rel_tags = map(list, zip(*data))
    token_ids = torch.tensor(sequence_padding(token_ids, length=maxlen), dtype=torch.long, device=device)
    corres_tags = torch.tensor(sequence_padding(corres_tags), dtype=torch.long, device=device)
    seq_tags = torch.tensor(sequence_padding(seq_tags), dtype=torch.long, device=device)
    rel = torch.tensor(rel, dtype=torch.long, device=device)
    rel_tags = torch.tensor(sequence_padding(rel_tags), dtype=torch.long, device=device)

    attention_mask = (token_ids != tokenizer._token_pad_id).long()
    return [token_ids, rel], [seq_tags, rel_tags, corres_tags, attention_mask]

train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/relation_extraction/BD_Knowledge_Extraction/train_data.json'), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataset = MyDataset('F:/Projects/data/corpus/relation_extraction/BD_Knowledge_Extraction/dev_data.json')
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn) 


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)
        return features_output


class SequenceLabelForSO(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(SequenceLabelForSO, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag_sub = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.hidden2tag_obj = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        """
        Args:
            input_features: (bs, seq_len, h)
        """
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        sub_output = self.hidden2tag_sub(features_tmp)
        obj_output = self.hidden2tag_obj(features_tmp)
        return sub_output, obj_output


class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.seq_tag_size = len(Label2IdxSub)
        self.rel_num = len(predicate2id)

        # pretrain model
        self.bert = build_transformer_model(config_path, checkpoint_path, segment_vocab_size=0)
        config = self.bert.configs
        # sequence tagging
        self.sequence_tagging_sub = MultiNonLinearClassifier(config.hidden_size * 2, self.seq_tag_size, drop_prob)
        self.sequence_tagging_obj = MultiNonLinearClassifier(config.hidden_size * 2, self.seq_tag_size, drop_prob)
        self.sequence_tagging_sum = SequenceLabelForSO(config.hidden_size, self.seq_tag_size, drop_prob)
        # global correspondence
        self.global_corres = MultiNonLinearClassifier(config.hidden_size * 2, 1, drop_prob)
        # relation judgement
        self.rel_judgement = MultiNonLinearClassifier(config.hidden_size, self.rel_num, drop_prob)
        self.rel_embedding = nn.Embedding(self.rel_num, config.hidden_size)

    @staticmethod
    def masked_avgpool(sent, mask):
        mask_ = mask.masked_fill(mask == 0, -1e9).float()
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)

    def forward(self, input_ids=None, potential_rels=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            potential_rels: (bs,), only in train stage.
        """
        # pre-train model
        sequence_output = self.bert([input_ids])  # sequence_output, pooled_output, (hidden_states), (attentions)
        bs, seq_len, h = sequence_output.size()
        attention_mask = (input_ids != tokenizer._token_pad_id).long()
        corres_mask, rel_pred = None, None

        if ensure_rel:
            # (bs, h)
            h_k_avg = self.masked_avgpool(sequence_output, attention_mask)
            # (bs, rel_num)
            rel_pred = self.rel_judgement(h_k_avg)

        # before fuse relation representation
        if ensure_corres:
            # for every position $i$ in sequence, should concate $j$ to predict.
            sub_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (bs, s, s, h)
            obj_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (bs, s, s, h)
            # batch x seq_len x seq_len x 2*hidden
            corres_pred = torch.cat([sub_extend, obj_extend], 3)
            # (bs, seq_len, seq_len)
            corres_pred = self.global_corres(corres_pred).squeeze(-1)
            mask_tmp1 = attention_mask.unsqueeze(-1)
            mask_tmp2 = attention_mask.unsqueeze(1)
            corres_mask = mask_tmp1 * mask_tmp2

        # relation predict and data construction in inference stage
        xi, pred_rels = None, None
        if ensure_rel:
            # (bs, rel_num)
            rel_pred_onehot = torch.where(torch.sigmoid(rel_pred) > rel_threshold,
                                          torch.ones(rel_pred.size(), device=rel_pred.device),
                                          torch.zeros(rel_pred.size(), device=rel_pred.device))

            # if potential relation is null
            for idx, sample in enumerate(rel_pred_onehot):
                if 1 not in sample:
                    # (rel_num,)
                    max_index = torch.argmax(rel_pred[idx])
                    sample[max_index] = 1
                    rel_pred_onehot[idx] = sample

            # 2*(sum(x_i),)
            bs_idxs, pred_rels = torch.nonzero(rel_pred_onehot, as_tuple=True)
            # get x_i
            xi_dict = Counter(bs_idxs.tolist())
            xi = [xi_dict[idx] for idx in range(bs)]

            pos_seq_output = []
            pos_potential_rel = []
            pos_attention_mask = []
            for bs_idx, rel_idx in zip(bs_idxs, pred_rels):
                # (seq_len, h)
                pos_seq_output.append(sequence_output[bs_idx])
                pos_attention_mask.append(attention_mask[bs_idx])
                pos_potential_rel.append(rel_idx)
            # (sum(x_i), seq_len, h)
            sequence_output = torch.stack(pos_seq_output, dim=0)
            # (sum(x_i), seq_len)
            attention_mask = torch.stack(pos_attention_mask, dim=0)
            # (sum(x_i),)
            potential_rels = torch.stack(pos_potential_rel, dim=0)
        # ablation of relation judgement
        elif not ensure_rel:
            # construct test data
            # sequence_output = sequence_output.repeat((1, self.rel_num, 1)).view(bs * self.rel_num, seq_len, h)
            # attention_mask = attention_mask.repeat((1, self.rel_num)).view(bs * self.rel_num, seq_len)
            # potential_rels = torch.arange(0, self.rel_num, device=input_ids.device).repeat(bs)
            pass

        # (bs/sum(x_i), h)
        rel_emb = self.rel_embedding(potential_rels)

        # relation embedding vector fusion
        rel_emb = rel_emb.unsqueeze(1).expand(-1, seq_len, h)
        if emb_fusion == 'concat':
            # (bs/sum(x_i), seq_len, 2*h)
            decode_input = torch.cat([sequence_output, rel_emb], dim=-1)
            # (bs/sum(x_i), seq_len, tag_size)
            output_sub = self.sequence_tagging_sub(decode_input)
            output_obj = self.sequence_tagging_obj(decode_input)
        elif emb_fusion == 'sum':
            # (bs/sum(x_i), seq_len, h)
            decode_input = sequence_output + rel_emb
            # (bs/sum(x_i), seq_len, tag_size)
            output_sub, output_obj = self.sequence_tagging_sum(decode_input)
        return output_sub, output_obj, corres_mask, pred_rels, xi, rel_pred
    
    def predict(self, inputs):

        self.eval()
        with torch.no_grad():
            output_sub, output_obj, corres_mask, pred_rels, xi, _ = self.forward(inputs)

        # (sum(x_i), seq_len)
        pred_seq_sub = torch.argmax(torch.softmax(output_sub, dim=-1), dim=-1)
        pred_seq_obj = torch.argmax(torch.softmax(output_obj, dim=-1), dim=-1)
        # (sum(x_i), 2, seq_len)
        pred_seqs = torch.cat([pred_seq_sub.unsqueeze(1), pred_seq_obj.unsqueeze(1)], dim=1)
        if ensure_corres:
            corres_pred = torch.sigmoid(corres_pred) * corres_mask
            # (bs, seq_len, seq_len)
            pred_corres_onehot = torch.where(corres_pred > corres_threshold,
                                                torch.ones(corres_pred.size(), device=corres_pred.device),
                                                torch.zeros(corres_pred.size(), device=corres_pred.device))
            return pred_seqs, pred_corres_onehot, xi, pred_rels
        return pred_seqs, xi, pred_rels


train_model = Model().to(device)

class Loss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, outputs, targets):
        output_sub, output_obj, corres_mask, pred_rels, xi, rel_pred = outputs
        seq_tags, rel_tags, corres_tags, attention_mask = targets

        bs = seq_tags.shape[0]
        # calculate loss
        attention_mask = attention_mask.view(-1)
        # sequence label loss
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_seq_sub = (loss_func(output_sub.view(-1, len(Label2IdxSub)), seq_tags[:, 0, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
        loss_seq_obj = (loss_func(output_obj.view(-1, len(Label2IdxSub)), seq_tags[:, 1, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
        loss_seq = (loss_seq_sub + loss_seq_obj) / 2
        # init
        loss_matrix, loss_rel = torch.tensor(0), torch.tensor(0)
        if ensure_corres:
            corres_pred = corres_pred.view(bs, -1)
            corres_mask = corres_mask.view(bs, -1)
            corres_tags = corres_tags.view(bs, -1)
            loss_func = nn.BCEWithLogitsLoss(reduction='none')
            loss_matrix = (loss_func(corres_pred, corres_tags.float()) * corres_mask).sum() / corres_mask.sum()

        if ensure_rel:
            loss_func = nn.BCEWithLogitsLoss(reduction='mean')
            loss_rel = loss_func(rel_pred, rel_tags.float())

        loss = loss_seq + loss_matrix + loss_rel
        return loss, loss_seq, loss_matrix, loss_rel

train_model.compile(loss=Loss(), optimizer=optim.Adam(train_model.parameters(), 1e-5))

def extract_spoes(text):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=device)

    # 抽取subject
    seq_output, subject_preds = train_model.predict_subject([token_ids, segment_ids])
    subject_preds[:, [0, -1]] *= 0  # 首cls, 尾sep置为0
    start = torch.where(subject_preds[0, :, 0] > 0.6)[0]
    end = torch.where(subject_preds[0, :, 1] > 0.5)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i.item(), j.item()))
    if subjects:
        spoes = []
        # token_ids = token_ids.repeat([len(subjects)]+[1]*(len(token_ids.shape)-1))
        # segment_ids = segment_ids.repeat([len(subjects)]+[1]*(len(token_ids.shape)-1))
        seq_output = seq_output.repeat([len(subjects)]+[1]*(len(seq_output.shape)-1))
        subjects = torch.tensor(subjects, dtype=torch.long, device=device)
        # 传入subject，抽取object和predicate
        object_preds = train_model.predict_object([seq_output, subjects])
        object_preds[:, [0, -1]] *= 0
        for subject, object_pred in zip(subjects, object_preds):
            start = torch.where(object_pred[:, :, 0] > 0.6)
            end = torch.where(object_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append(
                            ((mapping[subject[0]][0],
                              mapping[subject[1]][-1]), predicate1.item(),
                             (mapping[_start][0], mapping[_end][-1]))
                        )
                        break
        return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        R = set([SPO(spo) for spo in extract_spoes(d['text'])])
        T = set([SPO(spo) for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
                       ensure_ascii=False,
                       indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        # optimizer.apply_ema_weights()
        f1, precision, recall = evaluate(valid_dataset.data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            # train_model.save_weights('best_model.pt')
        # optimizer.reset_old_weights()
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


if __name__ == '__main__':
    # 训练
    if True:
        evaluator = Evaluator()
        train_model.fit(train_dataloader, steps_per_epoch=None, epochs=20, callbacks=[evaluator])
    # 预测并评估
    else:
        train_model.load_weights('best_model.pt')
        f1, precision, recall = evaluate(valid_dataset.data)


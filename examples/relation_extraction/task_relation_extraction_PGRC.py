#! -*- coding:utf-8 -*-
# 三元组抽取任务，PGRC: https://github.com/hy-struggle/PRGC

import json
import numpy as np
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
ensure_corres = True
num_negs = 4
drop_prob = 0.3
emb_fusion = 'concat'
Label2IdxSub = {"B-H": 1, "I-H": 2, "O": 0}
Label2IdxObj = {"B-T": 1, "I-T": 2, "O": 0}


maxlen = 128
batch_size = 32
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

    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    token_ids = tokenizer.tokens_to_ids(tokens)
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
    return token_ids, tokens, spoes


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
                token_ids, _, spoes = get_spoes(l['text'], labels)

                if spoes:
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
                # if len(D) > 1000:  # 快速测试使用
                #     break
        return D

class MyTestDataset(ListDataset):
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
                token_ids, tokens, spoes = get_spoes(l['text'], labels)

                if spoes:
                    spoes_new = []
                    for subject, object_labels in spoes.items():
                        for object_label in object_labels:
                            spoes_new.append((('H', subject[0], subject[1]), ('T', object_label[0], object_label[1]), object_label[-1]))
                    D.append([token_ids, spoes_new, tokens])
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

def collate_fn_test(data):
    token_ids, spoes, tokens = map(list, zip(*data))
    token_ids = torch.tensor(sequence_padding(token_ids, length=maxlen), dtype=torch.long, device=device)
    return token_ids, spoes, tokens

train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/relation_extraction/BD_Knowledge_Extraction/train_data.json'), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataset = MyTestDataset('F:/Projects/data/corpus/relation_extraction/BD_Knowledge_Extraction/dev_data.json')
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn_test) 


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

    def forward(self, input_ids=None, potential_rels=None, train_stage=True):
        """
        Args:
            input_ids: (batch_size, seq_len)
            potential_rels: (bs,), only in train stage.
        """
        # pre-train model
        sequence_output = self.bert([input_ids])  # sequence_output, pooled_output, (hidden_states), (attentions)
        bs, seq_len, h = sequence_output.size()
        attention_mask = (input_ids != tokenizer._token_pad_id).long()
        corres_mask, rel_pred, corres_pred = None, None, None

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
        if ensure_rel and (not train_stage):
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
        elif not ensure_rel and (not train_stage):
            # construct test data
            sequence_output = sequence_output.repeat((1, self.rel_num, 1)).view(bs * self.rel_num, seq_len, h)
            attention_mask = attention_mask.repeat((1, self.rel_num)).view(bs * self.rel_num, seq_len)
            potential_rels = torch.arange(0, self.rel_num, device=input_ids.device).repeat(bs)

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
        return output_sub, output_obj, corres_pred, corres_mask, pred_rels, xi, rel_pred
    
    def predict(self, inputs):

        self.eval()
        with torch.no_grad():
            output_sub, output_obj, corres_pred, corres_mask, pred_rels, xi, _ = self.forward(inputs, train_stage=False)

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
        output_sub, output_obj, corres_pred, corres_mask, pred_rels, xi, rel_pred = outputs
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
        return {'loss': loss, 'loss_seq': loss_seq, 'loss_matrix': loss_matrix, 'loss_rel': loss_rel}

train_model.compile(loss=Loss(), optimizer=optim.Adam(train_model.parameters(), 1e-5), clip_grad_norm=2.0)

def get_metrics(correct_num, predict_num, gold_num):
    p = correct_num / predict_num if predict_num > 0 else 0
    r = correct_num / gold_num if gold_num > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {
        'correct_num': correct_num,
        'predict_num': predict_num,
        'gold_num': gold_num,
        'precision': p,
        'recall': r,
        'f1': f1
    }

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    content = tag_name.split('-')
    tag_class = content[0]
    if len(content) == 1:
        return tag_class
    ht = content[-1]
    return tag_class, ht

def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position
    Args:
        seq: np.array[4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default1 = tags['O']
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default1 and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default1:
            res = get_chunk_type(tok, idx_to_tag)
            if len(res) == 1:
                continue
            tok_chunk_class, ht = get_chunk_type(tok, idx_to_tag)
            tok_chunk_type = ht
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

def tag_mapping_corres(predict_tags, pre_corres, pre_rels=None, label2idx_sub=None, label2idx_obj=None):
    """
    Args:
        predict_tags: np.array, (xi, 2, max_sen_len)
        pre_corres: (seq_len, seq_len)
        pre_rels: (xi,)
    """
    rel_num = predict_tags.shape[0]
    pre_triples = []
    for idx in range(rel_num):
        heads, tails = [], []
        pred_chunks_sub = get_chunks(predict_tags[idx][0], label2idx_sub)
        pred_chunks_obj = get_chunks(predict_tags[idx][1], label2idx_obj)
        pred_chunks = pred_chunks_sub + pred_chunks_obj
        for ch in pred_chunks:
            if ch[0] == 'H':
                heads.append(ch)
            elif ch[0] == 'T':
                tails.append(ch)
        retain_hts = [(h, t) for h in heads for t in tails if pre_corres[h[1]][t[1]] == 1]
        for h_t in retain_hts:
            if pre_rels is not None:
                triple = list(h_t) + [pre_rels[idx]]
            else:
                triple = list(h_t) + [idx]
            pre_triples.append(tuple(triple))
    return pre_triples

def span2str(triples, tokens):
    def _concat(token_list):
        result = ''
        for idx, t in enumerate(token_list):
            if idx == 0:
                result = t
            elif t.startswith('##'):
                result += t.lstrip('##')
            else:
                result += ' ' + t
        return result

    output = []
    for triple in triples:
        rel = triple[-1]
        sub_tokens = tokens[triple[0][1]:triple[0][-1]]
        obj_tokens = tokens[triple[1][1]:triple[1][-1]]
        sub = _concat(sub_tokens)
        obj = _concat(obj_tokens)
        output.append((sub, obj, rel))
    return output

def evaluate(data_iterator, mark='Val'):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    rel_num = len(predicate2id)

    predictions = []
    ground_truths = []
    correct_num, predict_num, gold_num = 0, 0, 0

    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        # to device
        input_ids, triples, input_tokens = batch
        bs, seq_len = input_ids.size()

        # inference
        pred_seqs, pre_corres, xi, pred_rels = train_model.predict(input_ids)

        # (sum(x_i), seq_len)
        pred_seqs = pred_seqs.detach().cpu().numpy()
        # (bs, seq_len, seq_len)
        pre_corres = pre_corres.detach().cpu().numpy()

        if ensure_rel:
            # (bs,)
            xi = np.array(xi)
            # (sum(s_i),)
            pred_rels = pred_rels.detach().cpu().numpy()
            # decode by per batch
            xi_index = np.cumsum(xi).tolist()
            # (bs+1,)
            xi_index.insert(0, 0)

        for idx in range(bs):
            if ensure_rel:
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[xi_index[idx]:xi_index[idx + 1]],
                                                 pre_corres=pre_corres[idx],
                                                 pre_rels=pred_rels[xi_index[idx]:xi_index[idx + 1]],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)
            else:
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[idx * rel_num:(idx + 1) * rel_num],
                                                 pre_corres=pre_corres[idx],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)

            gold_triples = span2str(triples[idx], input_tokens[idx])
            pre_triples = span2str(pre_triples, input_tokens[idx])
            ground_truths.append(list(set(gold_triples)))
            predictions.append(list(set(pre_triples)))
            # counter
            correct_num += len(set(pre_triples) & set(gold_triples))
            predict_num += len(set(pre_triples))
            gold_num += len(set(gold_triples))
    metrics = get_metrics(correct_num, predict_num, gold_num)
    return metrics['f1'], metrics['precision'], metrics['recall']

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        # optimizer.apply_ema_weights()
        f1, precision, recall = evaluate(valid_dataloader)
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


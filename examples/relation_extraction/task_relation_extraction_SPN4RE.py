#! -*- coding:utf-8 -*-
# 三元组抽取任务，SPN4RE方案: ttps://github.com/DianboWork/SPN4RE

import json
import numpy as np
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.layers import MultiHeadAttentionLayer, PositionWiseFeedForward
from bert4torch.snippets import sequence_padding, Callback, ListDataset, seed_everything
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import collections
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ==========================================================参数设置==========================================================
rel_loss_weight = 1
head_ent_loss_weight = 2
tail_ent_loss_weight = 2
num_generated_triples = 10
num_decoder_layers = 3
na_rel_coef = 1
matcher = "avg"
n_best_size = 100
max_span_length = 12
weight_decay = 1e-5
encoder_lr = 1e-5
decoder_lr = 2e-5

maxlen = 128
batch_size = 8
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed_everything(42)

# 加载标签字典
predicate2id, id2predicate = {}, {}

with open('F:/Projects/data/corpus/relation_extraction/BD_Knowledge_Extraction/all_50_schemas', encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)
num_classes = len(predicate2id)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# ==========================================================数据读取==========================================================
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
            assert token_ids[s_idx:s_idx + len(s)] == s
            assert token_ids[o_idx:o_idx + len(o)] == o
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
            for l in tqdm(f, desc='Loading data'):
                l = json.loads(l)
                labels = [(spo['subject'], spo['predicate'], spo['object']) for spo in l['spo_list']]
                token_ids, segment_ids, spoes = get_spoes(l['text'], labels)
                if spoes:
                    D.append({'text': l['text'], 'spo_list': labels, 'token_ids': token_ids, 
                              'segment_ids': segment_ids, 'spoes': spoes})
                if len(D) > 1000:
                    break
        return D

def collate_fn(batch):
    batch_token_ids, batch_segment_ids = [], []
    targets = []
    for d in batch:
        token_ids, segment_ids, spoes = d['token_ids'], d['segment_ids'], d['spoes']
        if spoes:
            target = {"relation": [], "head_start_index": [], "head_end_index": [], "tail_start_index": [], "tail_end_index": []}
            for (head_start_index, head_end_index), object_labels in spoes.items():
                for tail_start_index, tail_end_index, relation_id in object_labels:
                    target["relation"].append(relation_id)
                    target["head_start_index"].append(head_start_index)
                    target["head_end_index"].append(head_end_index)
                    target["tail_start_index"].append(tail_start_index)
                    target["tail_end_index"].append(tail_end_index)
            # 构建batch
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            targets.append(target)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    targets = [{k: torch.tensor(v, dtype=torch.long, device=device) for k, v in t.items()} for t in targets]
    return [batch_token_ids, batch_segment_ids], targets

train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/relation_extraction/BD_Knowledge_Extraction/train_data.json'), 
                   batch_size=batch_size, shuffle=False, collate_fn=collate_fn) 
valid_dataset = MyDataset('F:/Projects/data/corpus/relation_extraction/BD_Knowledge_Extraction/train_data.json')
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn) 


# ==========================================================小函数==========================================================
def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def generate_span(start_logits, end_logits, seq_lens):
    _Prediction = collections.namedtuple(
        "Prediction", ["start_index", "end_index", "start_prob", "end_prob"]
    )
    output = {}
    start_probs = start_logits.softmax(-1)
    end_probs = end_logits.softmax(-1)
    start_probs = start_probs.cpu().tolist()
    end_probs = end_probs.cpu().tolist()
    for (start_prob, end_prob, seq_len) in zip(start_probs, end_probs, seq_lens):
        output = {}
        for triple_id in range(num_generated_triples):
            predictions = []
            start_indexes = _get_best_indexes(start_prob[triple_id], n_best_size)
            end_indexes = _get_best_indexes(end_prob[triple_id], n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the sentence. We throw out all
                    # invalid predictions.
                    if start_index >= (seq_len-1): # [SEP]
                        continue
                    if end_index >= (seq_len-1):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_span_length:
                        continue
                    predictions.append(
                        _Prediction(
                            start_index=start_index,
                            end_index=end_index,
                            start_prob=start_prob[triple_id][start_index],
                            end_prob=end_prob[triple_id][end_index],
                        )
                    )
            output[triple_id] = predictions
    return output


def generate_relation(pred_rel_logits):
    rel_probs, pred_rels = torch.max(pred_rel_logits.softmax(-1), dim=2)
    rel_probs = rel_probs.cpu().tolist()
    pred_rels = pred_rels.cpu().tolist()
    output = {}
    _Prediction = collections.namedtuple(
        "Prediction", ["pred_rel", "rel_prob"]
    )
    for (rel_prob, pred_rel) in zip(rel_probs, pred_rels):
        output = {}
        for triple_id in range(num_generated_triples):
            output[triple_id] = _Prediction(
                            pred_rel=pred_rel[triple_id],
                            rel_prob=rel_prob[triple_id])
    return output


def generate_triple(output, seq_lens, num_classes):
    _Pred_Triple = collections.namedtuple(
        "Pred_Triple", ["pred_rel", "rel_prob", "head_start_index", "head_end_index", "head_start_prob", "head_end_prob", "tail_start_index", "tail_end_index", "tail_start_prob", "tail_end_prob"]
    )
    pred_head_ent_dict = generate_span(output["head_start_logits"], output["head_end_logits"], seq_lens)
    pred_tail_ent_dict = generate_span(output["tail_start_logits"], output["tail_end_logits"], seq_lens)
    pred_rel_dict = generate_relation(output['pred_rel_logits'])
    triples = []
    for triple_id in range(num_generated_triples):
        pred_rel = pred_rel_dict[triple_id]
        pred_head = pred_head_ent_dict[triple_id]
        pred_tail = pred_tail_ent_dict[triple_id]
        triple = generate_strategy(pred_rel, pred_head, pred_tail, num_classes, _Pred_Triple)
        if triple:
            triples.append(triple)
    # print(triples)
    return triples


def generate_strategy(pred_rel, pred_head, pred_tail, num_classes, _Pred_Triple):
    if pred_rel.pred_rel != num_classes:
        if pred_head and pred_tail:
            for ele in pred_head:
                if ele.start_index != 0:
                    break
            head = ele
            for ele in pred_tail:
                if ele.start_index != 0:
                    break
            tail = ele
            return _Pred_Triple(pred_rel=pred_rel.pred_rel, rel_prob=pred_rel.rel_prob, head_start_index=head.start_index, head_end_index=head.end_index, head_start_prob=head.start_prob, head_end_prob=head.end_prob, tail_start_index=tail.start_index, tail_end_index=tail.end_index, tail_start_prob=tail.start_prob, tail_end_prob=tail.end_prob)
        else:
            return
    else:
        return


# ==========================================================模型结构==========================================================
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, loss_weight, matcher):
        super().__init__()
        self.cost_relation = loss_weight["relation"]
        self.cost_head = loss_weight["head_entity"]
        self.cost_tail = loss_weight["tail_entity"]
        self.matcher = matcher

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_rel_logits": Tensor of dim [batch_size, num_generated_triples, num_classes] with the classification logits
                 "{head, tail}_{start, end}_logits": Tensor of dim [batch_size, num_generated_triples, seq_len] with the predicted index logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_generated_triples, num_gold_triples)
        """
        bsz, num_generated_triples = outputs["pred_rel_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        pred_rel = outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, num_classes]
        gold_rel = torch.cat([v["relation"] for v in targets])
        # after masking the pad token
        pred_head_start = outputs["head_start_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, seq_len]
        pred_head_end = outputs["head_end_logits"].flatten(0, 1).softmax(-1)
        pred_tail_start = outputs["tail_start_logits"].flatten(0, 1).softmax(-1)
        pred_tail_end = outputs["tail_end_logits"].flatten(0, 1).softmax(-1)

        gold_head_start = torch.cat([v["head_start_index"] for v in targets])
        gold_head_end = torch.cat([v["head_end_index"] for v in targets])
        gold_tail_start = torch.cat([v["tail_start_index"] for v in targets])
        gold_tail_end = torch.cat([v["tail_end_index"] for v in targets])
        if self.matcher == "avg":
            cost = - self.cost_relation * pred_rel[:, gold_rel] - self.cost_head * 1/2 * (pred_head_start[:, gold_head_start] + pred_head_end[:, gold_head_end]) - self.cost_tail * 1/2 * (pred_tail_start[:, gold_tail_start] + pred_tail_end[:, gold_tail_end])
        elif self.matcher == "min":
            cost = torch.cat([pred_head_start[:, gold_head_start].unsqueeze(1), pred_rel[:, gold_rel].unsqueeze(1), pred_head_end[:, gold_head_end].unsqueeze(1), pred_tail_start[:, gold_tail_start].unsqueeze(1), pred_tail_end[:, gold_tail_end].unsqueeze(1)], dim=1)
            cost = - torch.min(cost, dim=1)[0]
        else:
            raise ValueError("Wrong matcher")
        cost = cost.view(bsz, num_generated_triples, -1).cpu()
        num_gold_triples = [len(v["relation"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(num_gold_triples, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetDecoder(nn.Module):
    def __init__(self, config, num_generated_triples, num_layers, num_classes):
        super().__init__()
        self.num_generated_triples = num_generated_triples
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.get('layer_norm_eps', 1e-12))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.query_embed = nn.Embedding(num_generated_triples, config.hidden_size)
        self.decoder2class = nn.Linear(config.hidden_size, num_classes + 1)
        self.decoder2span = nn.Linear(config.hidden_size, 4)

        self.head_start_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_end_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_start_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_end_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_start_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_end_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_start_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_end_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_start_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.head_end_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_start_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_end_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)

        torch.nn.init.orthogonal_(self.head_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)

    def forward(self, encoder_hidden_states, encoder_attention_mask):
        bsz = encoder_hidden_states.size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        hidden_states = self.dropout(self.LayerNorm(hidden_states))

        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, encoder_hidden_states, encoder_extended_attention_mask)

        class_logits = self.decoder2class(hidden_states)
        head_start_logits = self.head_start_metric_3(torch.tanh(self.head_start_metric_1(hidden_states).unsqueeze(2) + \
                            self.head_start_metric_2(encoder_hidden_states).unsqueeze(1))).squeeze()
        head_end_logits = self.head_end_metric_3(torch.tanh(self.head_end_metric_1(hidden_states).unsqueeze(2) + \
                          self.head_end_metric_2(encoder_hidden_states).unsqueeze(1))).squeeze()
        tail_start_logits = self.tail_start_metric_3(torch.tanh(self.tail_start_metric_1(hidden_states).unsqueeze(2) + \
                            self.tail_start_metric_2(encoder_hidden_states).unsqueeze(1))).squeeze()
        tail_end_logits = self.tail_end_metric_3(torch.tanh(self.tail_end_metric_1(hidden_states).unsqueeze(2) + \
                          self.tail_end_metric_2(encoder_hidden_states).unsqueeze(1))).squeeze()

        return class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits

# 基于bert4torch实现，存在问题
# class DecoderLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.attention = MultiHeadAttentionLayer(**config)
#         self.crossattention = MultiHeadAttentionLayer(**config)
#         self.ffc = PositionWiseFeedForward(**config)

#     def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask):
#         attention_output = self.attention(hidden_states)
#         cross_attention_outputs = self.crossattention(attention_output, None, encoder_hidden_states, encoder_attention_mask)
#         layer_output = self.ffc(cross_attention_outputs)
#         return layer_output

# 基于transformers实现
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.is_decoder = False
        config.layer_norm_eps = 1e-12
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, encoder_hidden_states, encoder_extended_attention_mask):
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs[0]

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = build_transformer_model(config_path, checkpoint_path)
        self.encoder.embeddings.word_embeddings.weight.requires_grad = False
        self.encoder.embeddings.position_embeddings.weight.requires_grad = False
        self.encoder.embeddings.segment_embeddings.weight.requires_grad = False

        config = self.encoder.configs
        self.num_classes = num_classes
        self.decoder = SetDecoder(config, num_generated_triples, num_decoder_layers, num_classes)

    def forward(self, input_ids, segment_ids):
        last_hidden_state = self.encoder([input_ids, segment_ids])
        attention_mask = (input_ids != tokenizer._token_pad_id).long()
        class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = self.decoder(encoder_hidden_states=last_hidden_state, encoder_attention_mask=attention_mask)
        # head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = span_logits.split(1, dim=-1)
        head_start_logits = head_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        head_end_logits = head_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        tail_start_logits = tail_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        tail_end_logits = tail_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)# [bsz, num_generated_triples, seq_len]
        outputs = {'pred_rel_logits': class_logits, 'head_start_logits': head_start_logits, 'head_end_logits': head_end_logits, 'tail_start_logits': tail_start_logits, 'tail_end_logits': tail_end_logits}
        return outputs

    def gen_triples(self, input_ids, segment_ids):
        with torch.no_grad():
            outputs = self.forward(input_ids, segment_ids)
            # print(outputs)
            seq_lens = (input_ids != tokenizer._token_pad_id).long().sum(dim=-1).cpu().numpy()
            pred_triple = generate_triple(outputs, seq_lens, self.num_classes)
            # print(pred_triple)
        return pred_triple

model = Model().to(device)

class SetCriterion(nn.Module):
    """ Loss的计算
    """
    def __init__(self, num_classes, loss_weight, na_coef, losses, matcher):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.matcher = HungarianMatcher(loss_weight, matcher)
        self.losses = losses
        rel_weight = torch.ones(self.num_classes + 1, device=device)
        rel_weight[-1] = na_coef
        self.register_buffer('rel_weight', rel_weight)

    def forward(self, outputs, targets):
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss == "entity" and self.empty_targets(targets):
                pass
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices))
        losses = sum(losses[k] * self.loss_weight[k] for k in losses.keys() if k in self.loss_weight)
        return losses

    def relation_loss(self, outputs, targets, indices):
        src_logits = outputs['pred_rel_logits'] # [bsz, num_generated_triples, num_rel+1]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["relation"][i] for t, (_, i) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=self.rel_weight)
        losses = {'relation': loss}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        pred_rel_logits = outputs['pred_rel_logits']
        device = pred_rel_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_rel_logits.argmax(-1) != pred_rel_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices,  **kwargs):
        loss_map = {'relation': self.relation_loss, 'cardinality': self.loss_cardinality, 'entity': self.entity_loss}
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def entity_loss(self, outputs, targets, indices):
        """Compute the losses related to the position of head entity or tail entity
        """
        idx = self._get_src_permutation_idx(indices)
        selected_pred_head_start = outputs["head_start_logits"][idx]
        selected_pred_head_end = outputs["head_end_logits"][idx]
        selected_pred_tail_start = outputs["tail_start_logits"][idx]
        selected_pred_tail_end = outputs["tail_end_logits"][idx]

        target_head_start = torch.cat([t["head_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_head_end = torch.cat([t["head_end_index"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_start = torch.cat([t["tail_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_end = torch.cat([t["tail_end_index"][i] for t, (_, i) in zip(targets, indices)])


        head_start_loss = F.cross_entropy(selected_pred_head_start, target_head_start)
        head_end_loss = F.cross_entropy(selected_pred_head_end, target_head_end)
        tail_start_loss = F.cross_entropy(selected_pred_tail_start, target_tail_start)
        tail_end_loss = F.cross_entropy(selected_pred_tail_end, target_tail_end)
        losses = {'head_entity': 1/2*(head_start_loss + head_end_loss), "tail_entity": 1/2*(tail_start_loss + tail_end_loss)}
        # print(losses)
        return losses

    @staticmethod
    def empty_targets(targets):
        flag = True
        for target in targets:
            if len(target["relation"]) != 0:
                flag = False
                break
        return flag

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
component = ['encoder', 'decoder']
grouped_params = [
    {
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and component[0] in n],
        'weight_decay': weight_decay,
        'lr': encoder_lr
    },
    {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and component[0] in n],
        'weight_decay': 0.0,
        'lr': encoder_lr
    },
    {
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and component[1] in n],
        'weight_decay': weight_decay,
        'lr': decoder_lr
    },
    {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and component[1] in n],
        'weight_decay': 0.0,
        'lr': decoder_lr
    }
]

model.compile(loss=SetCriterion(num_classes, loss_weight={"relation": rel_loss_weight, "head_entity": head_ent_loss_weight, "tail_entity": tail_ent_loss_weight}, 
                                na_coef=na_rel_coef, losses=["entity", "relation"], matcher=matcher), 
              optimizer=optim.Adam(grouped_params))

def extract_spoes(text, threshold=0):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=device)

    # 抽取subject
    preds = model.gen_triples(token_ids, segment_ids)
    spoes = set()
    for pred in preds:
        if (pred.head_start_prob > threshold) and \
           (pred.head_end_prob > threshold) and \
           (pred.tail_start_prob > threshold) and \
           (pred.tail_end_prob > threshold) and \
           (pred.rel_prob > threshold):
            spoes.add((
                    text[mapping[pred.head_start_index][0]:mapping[pred.head_end_index][-1] + 1], id2predicate[pred.pred_rel],
                    text[mapping[pred.tail_start_index][0]:mapping[pred.tail_end_index][-1] + 1]
                ))
    return spoes


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
        model.fit(train_dataloader, steps_per_epoch=None, epochs=20, callbacks=[evaluator])
    # 预测并评估
    else:
        train_model.load_weights('best_model.pt')
        f1, precision, recall = evaluate(valid_dataset.data)


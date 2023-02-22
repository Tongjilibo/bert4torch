#! -*- coding:utf-8 -*-
# 三元组抽取任务，tplinker, cat方式实体部分收敛较快，关系部分收敛较慢
# 官方链接：https://github.com/131250208/TPlinker-joint-extraction
# 数据集：http://ai.baidu.com/broad/download?dataset=sked

import json
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback, ListDataset
from bert4torch.layers import TplinkerHandshakingKernel
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

maxlen = 64
batch_size = 64
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

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据
        单条格式：{'text': text, 'spo_list': [(s, p, o)]}
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                D.append({'text': l['text'],
                          'spo_list': [(spo['subject'], spo['predicate'], spo['object']) for spo in l['spo_list']]})
        return D

def trans_ij2k(seq_len, i, j):
    '''把第i行，第j列转化成上三角flat后的序号
    '''
    if (i > seq_len - 1) or (j > seq_len - 1) or (i > j):
        return 0
    return int(0.5*(2*seq_len-i+1)*i+(j-i))
map_ij2k = {(i, j): trans_ij2k(maxlen, i, j) for i in range(maxlen) for j in range(maxlen) if j >= i}
map_k2ij = {v: k for k, v in map_ij2k.items()}

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

def collate_fn(batch):
    pair_len = maxlen * (maxlen+1)//2
    # batch_entity_labels: [btz, pair_len]
    # batch_head_labels: [btz, rel_size, pair_len]
    # batch_tail_labels: [btz, rel_size, pair_len]
    batch_entity_labels = torch.zeros((len(batch), pair_len), dtype=torch.long, device=device)
    batch_head_labels = torch.zeros((len(batch), len(predicate2id), pair_len), dtype=torch.long, device=device)
    batch_tail_labels = torch.zeros((len(batch), len(predicate2id), pair_len), dtype=torch.long, device=device)

    batch_token_ids = []
    for i, d in enumerate(batch):
        token_ids = tokenizer.encode(d['text'])[0][1:-1][:maxlen]  # 这里要限制取前max_len个
        batch_token_ids.append(token_ids)
        # 整理三元组 {s: [(o, p)]}
        for s, p, o in d['spo_list']:
            s = tokenizer.encode(s)[0][1:-1]
            p = predicate2id[p]
            o = tokenizer.encode(o)[0][1:-1]
            sh = search(s, token_ids)  # 这里超过长度就会找不到
            oh = search(o, token_ids)
            if sh != -1 and oh != -1:
                st, ot = sh+len(s)-1, oh+len(o)-1
                batch_entity_labels[i, map_ij2k[sh, st]] = 1
                batch_entity_labels[i, map_ij2k[oh, ot]] = 1
                if sh <= oh:
                    batch_head_labels[i, p, map_ij2k[sh, oh]] = 1
                else:
                    batch_head_labels[i, p, map_ij2k[oh, sh]] = 2
                if st <= ot:
                    batch_tail_labels[i, p, map_ij2k[st, ot]] = 1
                else:
                    batch_tail_labels[i, p, map_ij2k[ot, st]] = 2

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, length=maxlen), dtype=torch.long, device=device)
    return [batch_token_ids], [batch_entity_labels, batch_head_labels, batch_tail_labels]
    
train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/relation_extraction/BD_Knowledge_Extraction/train_data.json'), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_dataset = MyDataset('F:/Projects/data/corpus/relation_extraction/BD_Knowledge_Extraction/dev_data.json')
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path, checkpoint_path, segment_vocab_size=0)
        self.combine_fc = nn.Linear(768*2, 768)
        self.ent_fc = nn.Linear(768, 2)
        self.head_rel_fc = nn.Linear(768, len(predicate2id)*3)
        self.tail_rel_fc = nn.Linear(768, len(predicate2id)*3)
        self.handshaking_kernel = TplinkerHandshakingKernel(768, shaking_type='cat')

    def forward(self, *inputs):
        last_hidden_state = self.bert(inputs)  # [btz, seq_len, hdsz]
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)  # [btz, pair_len, hdsz]
        ent_shaking_outputs = self.ent_fc(shaking_hiddens)  # [btz, pair_len, 2]

        btz, pair_len = shaking_hiddens.shape[:2]
        head_rel_shaking_outputs = self.head_rel_fc(shaking_hiddens).reshape(btz, -1, pair_len, 3)  #[btz, predicate_num, pair_len, 3]
        tail_rel_shaking_outputs = self.tail_rel_fc(shaking_hiddens).reshape(btz, -1, pair_len, 3)

        return ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs

model = Model().to(device)

class MyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
    def forward(self, y_preds, y_trues):
        loss_list = []
        for y_pred, y_true in zip(y_preds, y_trues):
            loss = super().forward(y_pred.view(-1, y_pred.size()[-1]), y_true.view(-1))
            loss_list.append(loss)
        
        z = (2 * len(predicate2id) + 1)
        total_steps = 6000 # 前期实体识别的权重高一些，建议也可以设置为model.total_steps
        w_ent = max(1 / z + 1 - model.global_step / total_steps, 1 / z)
        w_rel = min((len(predicate2id) / z) * model.global_step / total_steps, (len(predicate2id) / z))
        loss = w_ent*loss_list[0] + w_rel*loss_list[1] + w_rel*loss_list[2]

        return {'loss': loss, 'entity_loss': loss_list[0], 'head_loss': loss_list[1], 'tail_loss': loss_list[2]}

model.compile(loss=MyLoss(), optimizer=optim.Adam(model.parameters(), 1e-4))

def extract_spoes(text):
    """抽取输入text所包含的三元组
    """
    def get_spots_fr_shaking_tag(shaking_tag):
        '''解析关系
        '''
        spots = []
        for shaking_inds in shaking_tag.nonzero():
            rel_id = shaking_inds[0].item()
            tag_id = shaking_tag[rel_id][shaking_inds[1]].item()
            matrix_inds = map_k2ij[shaking_inds[1].item()]
            # 保证前面是subject，后面是object
            if tag_id == 1:
                spot = (rel_id, matrix_inds[0], matrix_inds[1])
            elif tag_id == 2:
                spot = (rel_id, matrix_inds[1], matrix_inds[0])
            spots.append(spot)
        return spots

    tokens = tokenizer.tokenize(text)[1:-1]
    mapping = tokenizer.rematch(text, tokens)
    token_ids = tokenizer.encode(text)[0][1:-1]
    token_ids_ts = torch.tensor(sequence_padding([token_ids], length=maxlen), dtype=torch.long, device=device)
    outputs = model.predict([token_ids_ts])
    outputs = [o[0].argmax(dim=-1) for o in outputs]
    # 抽取entity
    ent_matrix_spots = set()
    ent_text = set()
    for shaking_ind in outputs[0].nonzero():
        shaking_ind_ = shaking_ind[0].item()
        # tag_id = outputs[0][shaking_ind_]
        matrix_inds = map_k2ij[shaking_ind_]
        spot = (matrix_inds[0], matrix_inds[1])
        if (spot[0] < len(mapping)) and (spot[1] < len(mapping)):  # 实体起始在mapping范围内
            ent_matrix_spots.add(spot)
            ent_text.add(text[mapping[spot[0]][0]:mapping[spot[1]][-1] + 1])

    # 识别对应的predicate
    head_rel_matrix_spots = get_spots_fr_shaking_tag(outputs[1])
    tail_rel_matrix_spots = get_spots_fr_shaking_tag(outputs[2])

    spoes = []
    for rel_h, sh, oh in head_rel_matrix_spots:
        for rel_t, st, ot in tail_rel_matrix_spots:
            # 如果关系相同，且(sh, st)和(oh, ot)都在entity_maxtrix_spots中
            if (rel_h == rel_t) and ((sh, st) in ent_matrix_spots) and ((oh, ot) in ent_matrix_spots):
                spoes.append((text[mapping[sh][0]:mapping[st][-1] + 1], id2predicate[rel_h], text[mapping[oh][0]:mapping[ot][-1] + 1]))
    return spoes, token_ids, ent_text


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (tuple(tokenizer.tokenize(spo[0])), spo[1], tuple(tokenizer.tokenize(spo[2])))

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 0, 1e-10, 1e-10
    E1, E2 = 0, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        spoes, token_ids, ent_text_pred = extract_spoes(d['text'])
        # spo_list是用来根据maxlen删减的
        spo_list = []
        for s, p, o in d['spo_list']:
            s_ = tokenizer.encode(s)[0][1:-1]
            o_ = tokenizer.encode(o)[0][1:-1]
            sh = search(s_, token_ids)  # 这里超过长度就会找不到
            oh = search(o_, token_ids)
            if sh != -1 and oh != -1:
                spo_list.append((s, p, o))

        # 计算三元组的f1值
        R = set([SPO(spo) for spo in spoes])
        T = set([SPO(spo) for spo in spo_list])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

        # 计算实体的指标
        ent_text_truth = set([spo[0] for spo in spo_list] + [spo[-1] for spo in spo_list])
        E1 += len(ent_text_pred & ent_text_truth)
        E2 += len(ent_text_truth)
        E_acc = E1 / E2

        # 计算entity_matrix, head_matrix，tail_matrix的accuracy
        pbar.update()
        pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f, ent_acc: %.5f' % (f1, precision, recall, E_acc))
        s = json.dumps({'text': d['text'], 'spo_list': list(T), 'spo_list_pred': list(R),
                        'new': list(R - T), 'lack': list(T - R)}, ensure_ascii=False, indent=4)
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
        f1, precision, recall = evaluate(valid_dataset.data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            # model.save_weights('best_model.pt')
        print('f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' % (f1, precision, recall, self.best_val_f1))


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, steps_per_epoch=None, epochs=20, callbacks=[evaluator])
else:
    model.load_weights('best_model.pt')

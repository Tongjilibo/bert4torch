#! -*- coding: utf-8 -*-
# 追一科技2019年NL2SQL挑战赛的一个Baseline（个人作品，非官方发布，基于Bert）
# 比赛地址：https://tianchi.aliyun.com/competition/entrance/231716/introduction
# 科学空间：https://kexue.fm/archives/6771
# 苏神结果是58%左右，我复现出来58.39%

# 思路：[CLS] question [SEP] [CLS] col1 [SEP] [CLS] col2 [SEP]
# 整句的[CLS]用来做conds连接符判断: {0:"", 1:"and", 2:"or"}
# col的[CLS]用来预测该列是否被select+agg聚合判断: {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM", 6:"不被select"}
''' 单条样本示例
{
    "table_id": "a1b2c3d4", # 相应表格的id
    "question": "世茂茂悦府新盘容积率大于1，请问它的套均面积是多少？", # 自然语言问句
    "sql":{ # 真实SQL
        "sel": [7], # SQL选择的列 
        "agg": [0], # 选择的列相应的聚合函数, '0'代表无
        "cond_conn_op": 0, # 条件之间的关系
        "conds": [
            [1, 2, "世茂茂悦府"], # 条件列, 条件类型, 条件值，col_1 == "世茂茂悦府"
            [6, 0, "1"]
        ]
    }
}
'''

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, Callback
from bert4torch.optimizers import get_linear_schedule_with_warmup
import json
import codecs
import numpy as np
from tqdm import tqdm
import jieba
import editdistance
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim
import re

batch_size = 16
maxlen = 160
num_agg = 7 # agg_sql_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM", 6:"不被select"}
num_op = 5 # {0:">", 1:"<", 2:"==", 3:"!=", 4:"不被select"}
num_cond_conn_op = 3 # conn_sql_dict = {0:"", 1:"and", 2:"or"}
learning_rate = 2.5e-5
epochs = 15
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'


def read_data(data_file, table_file):
    data, tables = [], {}
    with open(data_file, 'r', encoding='utf-8') as f:
        for l in f:
            data.append(json.loads(l))
    with open(table_file, 'r', encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            d = {}
            d['headers'] = l['header']
            d['header2id'] = {j: i for i, j in enumerate(d['headers'])}
            d['content'] = {}
            d['all_values'] = set()
            rows = np.array(l['rows'])
            for i, h in enumerate(d['headers']):
                d['content'][h] = set(rows[:, i])
                d['all_values'].update(d['content'][h])
            d['all_values'] = set([i for i in d['all_values'] if hasattr(i, '__len__')])
            tables[l['id']] = d
    return data, tables

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R
tokenizer = OurTokenizer(token_dict)


def most_similar(s, slist):
    """从词表中找最相近的词（当无法全匹配的时候）
    """
    if len(slist) == 0:
        return s
    scores = [editdistance.eval(s, t) for t in slist]
    return slist[np.argmin(scores)]


def most_similar_2(w, s):
    """从句子s中找与w最相近的片段，
    借助分词工具和ngram的方式尽量精确地确定边界。
    """
    sw = jieba.lcut(s)
    sl = list(sw)
    sl.extend([''.join(i) for i in zip(sw, sw[1:])])
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:])])
    return most_similar(w, sl)

class MyDataset(Dataset):
    def __init__(self, data, tables):
        self.data = data
        self.tables = tables
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]
        # [CLS] question [SEP] [CLS] col1 [SEP] [CLS] col2 [SEP]
        x1 = tokenizer.encode(d['question'])[0]
        xm = [0] + [1] * len(d['question']) + [0]
        h = []
        for j in self.tables[d['table_id']]['headers']:
            _x1 = tokenizer.encode(j)[0]
            h.append(len(x1))
            x1.extend(_x1)
        if len(x1) > maxlen:
            return
        hm = [1] * len(h)  # 列的mask

        # 列是否被选择
        sel = []
        for j in range(len(h)):
            if j in d['sql']['sel']:
                j = d['sql']['sel'].index(j)
                sel.append(d['sql']['agg'][j])
            else:
                sel.append(num_agg - 1) # 不被select则被标记为num_agg-1
        conn = [d['sql']['cond_conn_op']]
        csel = np.zeros(len(d['question']) + 2, dtype='int32') # 这里的0既表示padding，又表示第一列，padding部分训练时会被mask
        cop = np.zeros(len(d['question']) + 2, dtype='int32') + num_op - 1 # 不被select则被标记为num_op-1
        for j in d['sql']['conds']:
            if j[2] not in d['question']:
                j[2] = most_similar_2(j[2], d['question'])
            if j[2] not in d['question']:
                continue
            k = d['question'].index(j[2])
            csel[k + 1: k + 1 + len(j[2])] = j[0]
            cop[k + 1: k + 1 + len(j[2])] = j[1]

        # x1: bert的输入 [101, 123, 121, 122, 123, 2399, 122, 118, 126, 3299, 5168, 6369, 2832, 6598, ...]
        # xm: bert输入mask [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...]
        # h: 列名[CLS]所在位置   [56, 60, 74, 89, 104, 114, 123, 132]
        # hm: 列名mask          [1, 1, 1, 1, 1, 1, 1, 1]
        # sel: 被select查找的列  [4, 6, 6, 6, 6, 6, 6, 6], 6表示列未被select，4表示COUNT
        # conn: 连接类型 [1], 1表示and
        # csel: 条件中的列                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # cop: 条件中的运算符（同时也是值的标记） [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        return x1, xm, h, hm, sel, conn, csel, cop

def collate_fn(batch):
    x1, xm, h, hm, sel, conn, csel, cop = zip(*[i for i in batch if i])
    x1 = torch.tensor(sequence_padding(x1), dtype=torch.long, device=device)
    xm = torch.tensor(sequence_padding(xm, length=x1.shape[1]), dtype=torch.long, device=device)
    h = torch.tensor(sequence_padding(h), dtype=torch.long, device=device)
    hm = torch.tensor(sequence_padding(hm), dtype=torch.long, device=device)
    sel = torch.tensor(sequence_padding(sel), dtype=torch.long, device=device)
    conn = torch.tensor(sequence_padding(conn), dtype=torch.long, device=device)
    csel = torch.tensor(sequence_padding(csel, length=x1.shape[1]), dtype=torch.long, device=device)
    cop = torch.tensor(sequence_padding(cop, length=x1.shape[1]), dtype=torch.long, device=device)
    return [x1, h, hm], [sel, conn, csel, cop, xm, hm]

datadir = 'F:/Projects/data/corpus/other/ZhuiyiTechnology_NL2SQL'
train_dataloader = DataLoader(MyDataset(*read_data(f'{datadir}/train/train.json', f'{datadir}/train/train.tables.json')), 
                              batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
valid_data, valid_table = read_data(f'{datadir}/val/val.json', f'{datadir}/val/val.tables.json')
test_data, test_table = read_data(f'{datadir}/test/test.json', f'{datadir}/test/test.tables.json')

class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0)
        hidden_size = self.bert.configs['hidden_size']
        self.conn = nn.Linear(hidden_size, num_cond_conn_op)
        self.agg = nn.Linear(hidden_size, num_agg)
        self.op = nn.Linear(hidden_size, num_op)
        self.dense1 = nn.Linear(hidden_size, 256)
        self.dense2 = nn.Linear(hidden_size, 256)
        self.dense3 = nn.Linear(256, 1)

    def forward(self, x1_in, h, hm):
        x = self.bert([x1_in])

        # cls判断条件连接符 {0:"", 1:"and", 2:"or"}
        x4conn = x[:, 0]  # [cls位]
        pconn = self.conn(x4conn)  # [btz, num_cond_conn_op]

        # 列的cls位用来判断列名的agg和是否被select {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM", 6:"不被select"}
        x4h = torch.gather(x, dim=1, index=h.unsqueeze(-1).expand(-1, -1, 768))  # [btz, col_len, hdsz]
        psel = self.agg(x4h)  # [btz, col_len, num_agg]

        # 序列标注conds的值和运算符
        pcop = self.op(x)  # [btz, seq_len, num_op]
        x = x.unsqueeze(2)  # [btz, seq_len, 1, hdsz]
        x4h = x4h.unsqueeze(1)  # [btz, 1, col_len, hdsz]

        pcsel_1 = self.dense1(x)  # [btz, seq_len, 1, 256]
        pcsel_2 = self.dense2(x4h)  # [btz, 1, col_len, 256]
        pcsel = pcsel_1 + pcsel_2
        pcsel = torch.tanh(pcsel)
        pcsel = self.dense3(pcsel)  # [btz, seq_len, col_len, 1]
        pcsel = pcsel[..., 0] - (1 - hm[:, None]) * 1e10  # [btz, seq_len, col_len]
        return pconn, psel, pcop, pcsel

model = Model().to(device)

class MyLoss(nn.Module):
    def forward(self, outputs, labels):
        pconn, psel, pcop, pcsel = outputs
        sel_in, conn_in, csel_in, cop_in, xm, hm = labels
        cm = torch.not_equal(cop_in, num_op - 1)

        batch_size = psel.shape[0]
        psel_loss = F.cross_entropy(psel.view(-1, num_agg), sel_in.view(-1), reduction='none').reshape(batch_size, -1)
        psel_loss = torch.sum(psel_loss * hm) / torch.sum(hm)
        pconn_loss = F.cross_entropy(pconn, conn_in.view(-1))
        pcop_loss = F.cross_entropy(pcop.view(-1, num_op), cop_in.view(-1), reduction='none').reshape(batch_size, -1)
        pcop_loss = torch.sum(pcop_loss * xm) / torch.sum(xm)
        pcsel_loss = F.cross_entropy(pcsel.view(-1, pcsel.shape[-1]), csel_in.view(-1), reduction='none').reshape(batch_size, -1)
        pcsel_loss = torch.sum(pcsel_loss * xm * cm) / torch.sum(xm * cm)
        loss = psel_loss + pconn_loss + pcop_loss + pcsel_loss
        return {'loss': loss, 'psel_loss': psel_loss, 'pconn_loss': pconn_loss, 'pcop_loss': pcop_loss, 'pcsel_loss': pcsel_loss}

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, len(train_dataloader), len(train_dataloader)*epochs)

model.compile(
    loss=MyLoss(),
    optimizer=optimizer,
    scheduler=scheduler
)

def nl2sql(question, table):
    """输入question和headers，转SQL
    """
    x1 = tokenizer.encode(question)[0]
    h = []
    for i in table['headers']:
        _x1 = tokenizer.encode(i)[0]
        h.append(len(x1))
        x1.extend(_x1)
    hm = [1] * len(h)
    pconn, psel, pcop, pcsel = model.predict([
        torch.tensor([x1], dtype=torch.long, device=device),
        torch.tensor([h], dtype=torch.long, device=device),
        torch.tensor([hm], dtype=torch.long, device=device)
    ])
    pconn, psel, pcop, pcsel = pconn.cpu().numpy(), psel.cpu().numpy(), pcop.cpu().numpy(), pcsel.cpu().numpy()
    R = {'agg': [], 'sel': []}
    for i, j in enumerate(psel[0].argmax(1)):
        if j != num_agg - 1: # num_agg-1类是不被select的意思
            R['sel'].append(i)
            R['agg'].append(int(j))
    conds = []
    v_op = -1
    for i, j in enumerate(pcop[0, :len(question)+1].argmax(1)):
        # 这里结合标注和分类来预测条件
        if j != num_op - 1:
            if v_op != j:
                if v_op != -1:
                    v_end = v_start + len(v_str)
                    csel = pcsel[0][v_start: v_end].mean(0).argmax()
                    conds.append((csel, v_op, v_str))
                v_start = i
                v_op = j
                v_str = question[i - 1]
            else:
                v_str += question[i - 1]
        elif v_op != -1:
            v_end = v_start + len(v_str)
            csel = pcsel[0][v_start: v_end].mean(0).argmax()
            conds.append((csel, v_op, v_str))
            v_op = -1
    R['conds'] = set()
    for i, j, k in conds:
        if re.findall('[^\d\.]', k):
            j = 2 # 非数字只能用等号
        if j == 2:
            if k not in table['all_values']:
                # 等号的值必须在table出现过，否则找一个最相近的
                k = most_similar(k, list(table['all_values']))
            h = table['headers'][i]
            # 然后检查值对应的列是否正确，如果不正确，直接修正列名
            if k not in table['content'][h]:
                for r, v in table['content'].items():
                    if k in v:
                        i = table['header2id'][r]
                        break
        R['conds'].add((int(i), int(j), str(k)))
    R['conds'] = list(R['conds'])
    if len(R['conds']) <= 1: # 条件数少于等于1时，条件连接符直接为0
        R['cond_conn_op'] = 0
    else:
        R['cond_conn_op'] = 1 + int(pconn[0, 1:].argmax()) # 不能是0
    return R


def is_equal(R1, R2):
    """判断两个SQL字典是否全匹配
    """
    return (R1['cond_conn_op'] == R2['cond_conn_op']) &\
    (set(zip(R1['sel'], R1['agg'])) == set(zip(R2['sel'], R2['agg']))) &\
    (set([tuple(i) for i in R1['conds']]) == set([tuple(i) for i in R2['conds']]))


class Evaluate(Callback):
    def __init__(self):
        self.accs = []
        self.best = 0.
        self.passed = 0
        self.stage = 0

    def on_epoch_end(self, global_step, epoch, logs=None):
        acc = self.evaluate(valid_data, valid_table)
        self.accs.append(acc)
        if acc > self.best:
            self.best = acc
            # model.save_weights('best_model.weights')
        print('acc: %.5f, best acc: %.5f\n' % (acc, self.best))
    
    def evaluate(self, data, tables):
        right = 0.
        pbar = tqdm()
        F = open('evaluate_pred.json', 'w', encoding='utf-8')
        for i, d in enumerate(data):
            question = d['question']
            table = tables[d['table_id']]
            R = nl2sql(question, table)
            right += float(is_equal(R, d['sql']))
            pbar.update(1)
            pbar.set_description('< acc: %.5f >' % (right / (i + 1)))
            d['sql_pred'] = R
            try:
                s = json.dumps(d, ensure_ascii=False, indent=4)
            except:
                continue
            F.write(s + '\n')
        F.close()
        pbar.close()
        return right / len(data)

    def test(self, data, tables, outfile='result.json'):
        pbar = tqdm()
        F = open(outfile, 'w')
        for i, d in enumerate(data):
            question = d['question']
            table = tables[d['table_id']]
            R = nl2sql(question, table)
            pbar.update(1)
            s = json.dumps(R, ensure_ascii=False)
            F.write(s.encode('utf-8') + '\n')
        F.close()
        pbar.close()


if __name__ == '__main__':
    evaluator = Evaluate()
    model.fit(
        train_dataloader,
        steps_per_epoch=None,
        epochs=epochs,
        callbacks=[evaluator]
    )
else:
    model.load_weights('best_model.weights')
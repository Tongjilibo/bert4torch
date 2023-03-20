#! -*- coding: utf-8 -*-
# 用 语言模型+棋谱 的方式监督训练一个下中国象棋模型
# 介绍：https://kexue.fm/archives/7877
# 数据：https://github.com/bojone/gpt_cchess
# 模型训练可以在python2/python3进行。但是cchess模块只支持python3，
# 因此如果需要交互式体验模型棋力，那么需要在python3下进行。
# 权重转换脚本见：https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_roberta_chess.py

import json
import numpy as np
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from bert4torch.snippets import sequence_padding, ListDataset, Callback
from cchess import *

# 基本信息
maxlen = 512
steps_per_epoch = None
epochs = 10000
batch_size = 16

# bert配置
config_path = 'F:/Projects/pretrain_ckpt/roberta/[hit_torch_base]--chinese-roberta-wwm-ext-base/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/roberta/[hit_torch_base]--chinese-roberta-wwm-ext-base/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/roberta/[hit_torch_base]--chinese-roberta-wwm-ext-base/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """读取全局棋谱
        返回：[(棋谱, 结果)]，其中结果等于2为红方赢棋，1为和棋，
        0为黑方赢棋，-1则为无明确标注胜负。
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                if not l['fen']:
                    result = int(l['items'].get(u'棋局结果', -1))
                    D.append((l['iccs'], result))
        return D


# 建立分词器
chars = [u'[PAD]'] + list(u'0123456789abcdefghi')
token_dict = dict(zip(chars, range(len(chars))))
tokenizer = Tokenizer(token_dict)
tokenizer._token_unk_id = 0
bert_token_dict = load_vocab(dict_path)
keep_tokens = [bert_token_dict[c] for c in chars]

count = 0
def get_count():
    if count < 20000:
        n = 8
    elif count < 40000:
        n = 4
    elif count < 80000:
        n = 2
    else:
        n = 1
    return n

def collate_fn(batch):
    """数据生成器
    """
    batch_token_ids, batch_segment_ids = [], []
    for text, _ in batch:
        token_ids, segment_ids = tokenizer.encode(' '.join(text), maxlen=maxlen // get_count() + 1)
        batch_token_ids.append([0] + token_ids[1:-1])
        batch_segment_ids.append([0] + segment_ids[1:-1])
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    global count
    count += 1
    return [batch_token_ids, batch_segment_ids], batch_token_ids

# 加载数据集
train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/seq2seq/qipu/qipu.json'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

# 由于字典中0不代表padding位，为避免attention_mask计算错误，这里token_pad_ids=-100
model = build_transformer_model(config_path, checkpoint_path, application='lm', with_mlm=True,
                                keep_tokens=keep_tokens, token_pad_ids=-100, add_trainer=True).to(device)

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, outputs, target):
        _, mlm_scores = outputs
        mlm_scores = mlm_scores[:, :-1, :].reshape(-1, mlm_scores.shape[-1])
        target = target[:, 1:].flatten()
        return super().forward(mlm_scores, target)

model.compile(loss=CrossEntropyLoss(ignore_index=0), optimizer=optim.Adam(model.parameters(), 1e-5))

class ChessPlayer(object):
    """交互式下棋程序
    """
    def move_to_chinese(self, move):
        """将单步走法转为中文描述
        """
        if not isinstance(move, Move):
            move = Move(self.board, move[0], move[1])
        return move.to_chinese()

    def move_to_iccs(self, move):
        """将单步走法转为iccs表示
        """
        if not isinstance(move, Move):
            move = Move(self.board, move[0], move[1])
        return move.to_iccs()

    def print_board(self):
        """打印当前棋盘
        直观起见，红方用红色表示，黑方用绿色表示。
        """
        for l in self.board.dump_board():
            for c in u'兵炮车马相仕帅':
                l = l.replace(c, u'\033[1;31;40m%s\033[0m' % c)
            for c in u'卒砲砗碼象士将':
                l = l.replace(c, u'\033[1;32;40m%s\033[0m' % c)
            print(l)

    def movable_steps(self):
        """给出当前局面所有候选走法
        """
        return [self.move_to_iccs(m) for m in self.board.create_moves()]

    def human_input(self):
        """人类行棋
        """
        while True:
            try:
                iccs = input(u'请输入iccs棋着: ')
                print(iccs)
                move = self.board.move_iccs(iccs)
                if move is not None:
                    return iccs, move
            except KeyboardInterrupt:
                return None
            except:
                pass

    def record(self, iccs):
        """将局面往前推进一步
        """
        self.history += iccs
        self.board.next_turn()
        self.print_board()
        self.current = (self.current + 1) % 2

    def new_game(self, current=0):
        """开新局
        """
        self.board = ChessBoard()
        self.board.from_fen(FULL_INIT_FEN)
        self.print_board()
        self.history = ''
        self.current = current
        if self.current == 0:  # 人类先手
            iccs, move = self.human_input()
            self.record(iccs)
        while True:
            # 机器走棋
            moves = self.movable_steps()
            iccses = [' '.join(self.history + m) for m in moves]
            token_ids = [[0] + tokenizer.encode(ic)[0][1:-1] for ic in iccses]
            token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
            segment_ids = torch.zeros_like(token_ids)
            preds = model.predict([token_ids, segment_ids])[-1][:, -5:-1]
            preds = nn.Softmax(dim=-1)(preds)
            preds = torch.take_along_dim(preds, token_ids[:, -4:, None], dim=2)
            preds = torch.log(preds + 1e-8)[:, :, 0].sum(dim=1)
            iccs = moves[preds.argmax()]
            move = self.board.move_iccs(iccs)
            self.record(iccs)
            if self.board.is_win():
                print(u'机器赢了')
                break
            # 人类走棋
            iccs, move = self.human_input()
            self.record(iccs)
            if self.board.is_win():
                print(u'人类赢了')
                break


chessplayer = ChessPlayer()


class Evaluator(Callback):
    """评估与保存
    """
    def on_epoch_end(self, global_step, epoch, logs=None):
        # 保存模型
        # model.save_weights('./best_model_chess.pt')
        pass


if __name__ == '__main__':
    choice = 'eval'

    if choice == 'train':
        evaluator = Evaluator()
        model.fit(train_dataloader, steps_per_epoch=steps_per_epoch, epochs=20, callbacks=[evaluator])
    else:
        model.load_weights('./best_model_chess.pt')
        chessplayer.new_game(0)  # 启动新棋局，0为人类先手，1为机器先手

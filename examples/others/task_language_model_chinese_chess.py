#! -*- coding: utf-8 -*-
# 用 语言模型+棋谱 的方式监督训练一个下中国象棋模型
# 介绍：https://kexue.fm/archives/7877
# 数据：https://github.com/bojone/gpt_cchess
# 模型训练可以在python2/python3进行。但是cchess模块只支持python3，
# 因此如果需要交互式体验模型棋力，那么需要在python3下进行。

import json
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from bert4torch.snippets import sequence_padding, ListDataset, take_along_dim
from bert4torch.callbacks import Callback
from cchess import *

# 基本信息
maxlen = 512
steps_per_epoch = None
epochs = 10000
batch_size = 16

# bert配置
config_path = 'E:/pretrain_ckpt/roberta/hfl@chinese-roberta-wwm-ext-base/config.json'
checkpoint_path = 'E:/pretrain_ckpt/roberta/hfl@chinese-roberta-wwm-ext-base/pytorch_model.bin'
dict_path = 'E:/pretrain_ckpt/roberta/hfl@chinese-roberta-wwm-ext-base/vocab.txt'
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
train_dataloader = DataLoader(MyDataset('F:/data/corpus/seq2seq/qipu/qipu.json'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

# 由于字典中0不代表padding位，为避免attention_mask计算错误，这里pad_token_id=-100
model = build_transformer_model(config_path, checkpoint_path, application='lm', with_mlm=True,
                                keep_tokens=keep_tokens, pad_token_id=-100, add_trainer=True).to(device)

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
            preds = take_along_dim(preds, token_ids[:, -4:, None], dim=2)
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


def convert(tf_path, torch_path):
    # 用 语言模型+棋谱 的方式监督训练一个下中国象棋模型
    # 介绍：https://kexue.fm/archives/7877
    # 只是转换苏神已经train好的模型，注意不是预训练模型
    import numpy as np
    import h5py
    import torch
    # 这里用的keras==2.3.1
    from keras.engine import saving


    torch_state_dict = {}
    # 1表示transpose, 0表示不变
    key_map = {
        'Embedding-Token/embeddings:0': ['embeddings.word_embeddings.weight', 0],
        'Embedding-Segment/embeddings:0': ['embeddings.segment_embeddings.weight', 0],
        'Embedding-Position/embeddings:0': ['embeddings.position_embeddings.weight', 0],
        'Embedding-Norm/gamma:0': ['embeddings.layerNorm.weight', 0],
        'Embedding-Norm/beta:0': ['embeddings.layerNorm.bias', 0],
        'MLM-Dense/kernel:0': ['mlmDense.weight', 1],
        'MLM-Dense/bias:0': ['mlmDense.bias', 0],
        'MLM-Norm/gamma:0': ['mlmLayerNorm.weight', 0],
        'MLM-Norm/beta:0': ['mlmLayerNorm.bias', 0],
        'MLM-Bias/bias:0': ['mlmBias', 0],
        }

    for i in range(12):
        key_map.update({
        f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+1}/kernel:0': [f'encoderLayer.{i}.multiHeadAttention.q.weight', 1],
        f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+1}/bias:0': [f'encoderLayer.{i}.multiHeadAttention.q.bias', 0],
        f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+2}/kernel:0': [f'encoderLayer.{i}.multiHeadAttention.k.weight', 1],
        f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+2}/bias:0': [f'encoderLayer.{i}.multiHeadAttention.k.bias', 0],
        f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+3}/kernel:0': [f'encoderLayer.{i}.multiHeadAttention.v.weight', 1],
        f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+3}/bias:0': [f'encoderLayer.{i}.multiHeadAttention.v.bias', 0],
        f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+4}/kernel:0': [f'encoderLayer.{i}.multiHeadAttention.o.weight', 1],
        f'Transformer-{i}-MultiHeadSelfAttention/dense_{i*6+4}/bias:0': [f'encoderLayer.{i}.multiHeadAttention.o.bias', 0],
        f'Transformer-{i}-MultiHeadSelfAttention-Norm/gamma:0': [f'encoderLayer.{i}.attnLayerNorm.weight', 0],
        f'Transformer-{i}-MultiHeadSelfAttention-Norm/beta:0': [f'encoderLayer.{i}.attnLayerNorm.bias', 0],
        f'Transformer-{i}-FeedForward/dense_{i*6+5}/kernel:0': [f'encoderLayer.{i}.feedForward.intermediateDense.weight', 1],
        f'Transformer-{i}-FeedForward/dense_{i*6+5}/bias:0': [f'encoderLayer.{i}.feedForward.intermediateDense.bias', 0],
        f'Transformer-{i}-FeedForward/dense_{i*6+6}/kernel:0': [f'encoderLayer.{i}.feedForward.outputDense.weight', 1],
        f'Transformer-{i}-FeedForward/dense_{i*6+6}/bias:0': [f'encoderLayer.{i}.feedForward.outputDense.bias', 0],
        f'Transformer-{i}-FeedForward-Norm/gamma:0': [f'encoderLayer.{i}.ffnLayerNorm.weight', 0],
        f'Transformer-{i}-FeedForward-Norm/beta:0': [f'encoderLayer.{i}.ffnLayerNorm.bias', 0],
        })

    consume_keys = set()
    with h5py.File(tf_path, mode='r') as f:
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        layer_names = saving.load_attributes_from_hdf5_group(f, 'layer_names')
        
        weight_value_tuples = []
        for k, name in enumerate(layer_names):
            g = f[name]
            weight_names = saving.load_attributes_from_hdf5_group(g, 'weight_names')
            weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]

            for i, weight_name in enumerate(weight_names):
                new_key = key_map[weight_name][0]
                if key_map[weight_name][1] == 1:  # transpose
                    torch_state_dict[new_key] = torch.from_numpy(weight_values[i]).T
                else:
                    torch_state_dict[new_key] = torch.from_numpy(weight_values[i])
                assert new_key not in consume_keys, 'duplicate keys'
                consume_keys.add(new_key)

        if hasattr(f, 'close'):
            f.close()
        elif hasattr(f.file, 'close'):
            f.file.close()


    torch_state_dict['mlmDecoder.weight'] = torch_state_dict['embeddings.word_embeddings.weight']
    torch_state_dict['mlmDecoder.bias'] = torch_state_dict['mlmBias']

    # for k, v in torch_state_dict.items():
    #     print(k, v.shape)
    torch.save(torch_state_dict, torch_path)


if __name__ == '__main__':
    choice = 'eval'

    if choice == 'train':
        evaluator = Evaluator()
        model.fit(train_dataloader, steps_per_epoch=steps_per_epoch, epochs=20, callbacks=[evaluator])
    else:
        # convert('E:/Github/bert4keras/examples/best_model_chess.weights', 'E:/Github/bert4torch/examples/others/best_model_chess.pt')
        model.load_weights('./best_model_chess.pt')
        chessplayer.new_game(0)  # 启动新棋局，0为人类先手，1为机器先手

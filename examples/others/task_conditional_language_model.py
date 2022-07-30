#! -*- coding: utf-8 -*-
# bert做conditional language model任务
# 按类随机生成文本，这个demo的类别是情感极性（正／负）
# 请参考：https://kexue.fm/archives/7124

from pydantic import NoneStrBytes
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.snippets import sequence_padding, text_segmentate, Callback, AutoRegressiveDecoder, ListDataset
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn


# 模型配置
maxlen = 128
batch_size = 16
num_classes = 2
epochs = 20

# bert配置
config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filenames):
        """加载数据，并尽量划分为不超过maxlen的句子
        """
        D = []
        seps, strips = u'\n。！？!?；;，, ', u'；;，, '
        for filename in filenames:
            with open(filename, encoding='utf-8') as f:
                for l in f:
                    text, label = l.strip().split('\t')
                    for t in text_segmentate(text, maxlen - 2, seps, strips):
                        D.append((t, int(label)))
                    # if len(D) >= 100:
                    #     break
        return D

def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for text, label in batch:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append(label)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids, batch_labels], batch_token_ids

# 加载数据集
train_dataloader = DataLoader(MyDataset([
    'F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.train.data',
    'F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.valid.data',
    'F:/Projects/data/corpus/sentence_classification/sentiment/sentiment.test.data']), 
    batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        c = nn.Embedding(num_classes, 128)
        self.bert = build_transformer_model(config_path,
                                            checkpoint_path,
                                            with_mlm=True,
                                            application='lm',
                                            keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
                                            layer_norm_cond=c,
                                            ignore_invalid_weights=True)  # 忽略未初始化的权重

    def forward(self, inputs):
        _, seq_output = self.bert(inputs)  # [btz, seq_len, vocab_size]
        return seq_output

model = Model().to(device)

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, input, target):
        input = input[:, :-1, :].reshape(-1, input.shape[-1])
        target = target[:, 1:].flatten()
        return super().forward(input, target)

model.compile(loss=CrossEntropyLoss(ignore_index=0), optimizer=optim.Adam(model.parameters(), 1e-5))


class RandomSentiment(AutoRegressiveDecoder):
    """根据情感标签（0:负，1:正）随机生成一批句子
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids = output_ids
        segment_ids = torch.zeros_like(token_ids, device=device)
        label = inputs[0]
        return model.predict([token_ids, segment_ids, label])[:, -1, :]

    def generate(self, label, n=1, topp=0.95):
        results = self.random_sample([[label]], n, topp=topp)  # 基于随机采样
        return [tokenizer.decode(ids.cpu().numpy()) for ids in results]


random_sentiment = RandomSentiment(
    start_id=tokenizer._token_start_id,
    end_id=tokenizer._token_end_id,
    maxlen=maxlen,
    device=device
)


def just_show():
    print(u'正面采样:')
    print(random_sentiment.generate(1, 5, 0.95), '\n')
    print(u'负面采样:')
    print(random_sentiment.generate(0, 5, 0.95), '\n')


class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, steps, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            # model.save_weights('./best_model.pt')
        # 演示效果
        just_show()


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(train_dataloader, epochs=epochs, steps_per_epoch=None, callbacks=[evaluator])
else:

    model.load_weights('./best_model.pt')

"""
正面采样:
[
    u'外观时尚、漂亮、性价比高。',
    u'外观漂亮，配置均衡，比较满意，性价比高，外观漂亮，性能较高。',
    u'我是在大学的时候看到这本书的，所以一直在买。书中的作者是林静蕾，她用自己的口吻写出了一个孩子成长中的心路历程，让我看到了她们成长中的不同之处，以及她们成长过程中的不同境界。让我很欣赏！',
    u'我想这是一本能够告诉读者什么是坏的，而不是教你怎样说话，告诉我什么是错。这里我推荐了《我要讲故事》，这本书是我很喜欢的一本书，我认为它的理由很多，但是，我相信我。如果你从中得到一些改进，或者你已经有了一个明智的决定。',
    u'我们一家五口住的是标间，大床房，大床的床很舒服；而我们在携程网上订了两套大床房，这个酒店的价格还是比较合理的；但是房间的隔音效果不太理想，有点响的声音；酒店门口的地铁在施工中，不方便；但是酒店的门口的出租车不知道是哪个车的，打车不是很方便；酒店外面的停'
]

负面采样:
[
    u'不知道是不是因为电池不太好，不是我不喜欢。',
    u'看了评论才买的. 结果发现不是那么便宜, 价格也不便宜.',
    u'1、外壳不容易沾手印，不容易洗洗2、屏幕有点旧， 不能下载铃声',
    u'我是7月6日订购了《杜拉拉升职记》并已通过银行付款，为什么订单下了两周多至今还未到货？是收货时间太快了，可能就这么过去了吧？',
    u'这本书我是在网上先看了一遍，后来我再看了一遍。感觉作者的文笔实在太烂了，特别是在写他的博客时特别别扭，写得很不专业，特别是他写股票时那个情绪调节的小男孩，简直就是自作聪明的样子，简直就是自作聪明的一种表现！'
]
"""

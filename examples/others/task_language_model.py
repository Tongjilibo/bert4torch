#! -*- coding: utf-8 -*-
# bert做language model任务，小说生成

import glob, re
from tqdm import tqdm
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.snippets import sequence_padding, AutoRegressiveDecoder, Callback, ListDataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim

maxlen = 256
batch_size = 8
epochs = 10000

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
        novels = []

        for txt in glob.glob(filenames):
            txt = open(txt, encoding='utf-8').read()
            txt = txt.replace('\r', '').replace('\n', '')
            txt = txt.replace(u'整理制作，并提供下载', '')
            txt = re.sub(u'www.*?com', '', txt)
            txt = txt.replace(u'\u3000', ' ')
            sents = []
            for t in txt.split('  '):
                for s in re.findall(u'.*?。', t):
                    if len(s) <= maxlen - 2:
                        sents.append(s)
            novels.append(sents)
        data = []
        pbar = tqdm(desc=u'构建语料中', total=sum(len(n) for n in novels))

        for novel in novels:
            s = u''
            for i in range(len(novel)):
                for j in range(len(novel) - i):
                    if len(s) + len(novel[i + j]) > maxlen - 2:
                        data.append(s)
                        s = u''
                        break
                    else:
                        s += novel[i + j]
                pbar.update(1)
                if i + j >= len(novel):
                    break
            if s:
                data.append(s)

        pbar.close()
        return data

def collate_fn(batch):
    batch_token_ids, batch_segment_ids = [], []
    for text in batch:
        token_ids, segment_ids = tokenizer.encode(text)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], batch_token_ids

# 加载数据集
train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/pretrain/金庸小说/*.txt'), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

# 建模
model = build_transformer_model(
    config_path,
    checkpoint_path,
    with_mlm=True,
    application='lm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    add_trainer=True
).to(device)
summary(model, input_data=[next(iter(train_dataloader))[0]])

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, outputs, target):
        _, mlm_scores = outputs
        mlm_scores = mlm_scores[:, :-1, :].reshape(-1, mlm_scores.shape[-1])
        target = target[:, 1:].flatten()
        return super().forward(mlm_scores, target)

model.compile(loss=CrossEntropyLoss(ignore_index=0), optimizer=optim.Adam(model.parameters(), 1e-5))

# 随机采样
class StoryCompletion(AutoRegressiveDecoder):
    """基于随机采样的故事续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids = inputs[0]
        token_ids = torch.cat([token_ids, output_ids], 1)
        segment_ids = torch.zeros_like(token_ids, device=device)
        _, mlm_scores = model.predict([token_ids, segment_ids])
        return mlm_scores[:, -1, :]

    def generate(self, text, n=1, topp=0.95):
        token_ids, _ = tokenizer.encode(text)
        results = self.random_sample([token_ids[:-1]], n, topp=topp)  # 基于随机采样
        return [text + tokenizer.decode(ids.cpu().numpy()) for ids in results]

story_completion = StoryCompletion(start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen, device=device)

def just_show():
    s1 = u'当晚两人在一家小客店中宿歇。张无忌躺在炕上，越想越是担心，走到赵敏窗外，但听她呼吸调匀，正自香梦沉酣。'
    s2 = u'虚竹飞身跃上松树的枝干，只见段延庆的钢杖深深嵌在树枝之中，全凭一股内力粘劲，挂住了下面四人，内力之深厚，实是非同小可。虚竹伸左手抓住钢杖，提将上来。'
    s3 = u'杨过居住在侠客岛，是令狐冲的弟子，武器是金蛇剑。'
    for s in [s1, s2, s3]:
        t = story_completion.generate(s)
        print(u'输入: %s' % s)
        print(u'结果: %s\n' % ('\n'.join(t)))


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

    model.load_weights('./best_model.weights')
"""
效果：

输入: 当晚两人在一家小客店中宿歇。张无忌躺在炕上，越想越是担心，走到赵敏窗外，但听她呼吸调匀，正自香梦沉酣。
结果: 当晚两人在一家小客店中宿歇。张无忌躺在炕上，越想越是担心，走到赵敏窗外，但听她呼吸调匀，正自香梦沉酣。次日清晨，张无忌便和赵敏去买了一匹高头大马，自己骑了随伴。那马甚有神骏，三十六斤重的身躯之中，竟无一头白马。他心中怦怦乱跳，暗想：若能将赵敏引出迷城，我决不致再和她相会，但若和赵姑娘相遇，我一生一世决计再难相见。何况我是她的私生女儿，这般亲热，岂不是好？我如何能和她相见？今后我要教训教训她才好？我教教她，教训她，要她心里快快活活的。他心如刀割，当即回到客店，将张无忌的所在说了。

输入: 虚竹飞身跃上松树的枝干，只见段延庆的钢杖深深嵌在树枝之中，全凭一股内力粘劲，挂住了下面四人，内力之深厚，实是非同小可。虚竹伸左手抓住钢杖，提将上来。
结果: 虚竹飞身跃上松树的枝干，只见段延庆的钢杖深深嵌在树枝之中，全凭一股内力粘劲，挂住了下面四人，内力之深厚，实是非同小可。虚竹伸左手抓住钢杖，提将上来。那矮子见他如此功力，大吃一惊，叫道：什么人？是谁？你干什么？我师父是谁？你们是谁？是谁？你们是谁？我师父是谁？你这矮子，便是段延庆。你们不知道我师父便是，是不是？快快说来。那矮子道：我师父便是延庆太子，他的徒弟也是段延庆。他老人家在唐朝做镇南王，你们便将他改名为延庆太子，叫做延庆太子！这名头倒怪，你们大伙儿听见了，也不知道他老人家是死是活。

输入: 杨过居住在侠客岛，是令狐冲的弟子，武器是金蛇剑。
结果: 杨过居住在侠客岛，是令狐冲的弟子，武器是金蛇剑。这时见他手中所握，竟是一柄特制的短剑，心中大喜，叫道：：原来是金蛇郎君的剑！原来你便是金蛇郎君的弟子，这一下可要叫我失望了。那人哈哈一笑，说道：好啊！好啊，好啊！我的金蛇剑是我的，不过我是你的。这人道：我姓杨名过，名字叫过。你是我儿子，是我女儿，是不是？你这么大的年纪，怎地自称金刀驸马？我这就给你取个名字，叫作过儿。
"""

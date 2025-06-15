#! -*- coding: utf-8 -*-
# guwenbert做Seq2Seq任务，采用UNILM方案，由于type_token_ids的权重为[1, hdsz]的全0向量，因此在指定use_segment_embedding=False
# 即传入segment_ids但是仅仅用于生成unilm的mask，并不经过segment_embedding层
# 介绍链接：https://kexue.fm/archives/6933

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.snippets import sequence_padding, text_segmentate, ListDataset
from bert4torch.generation import AutoRegressiveDecoder
from bert4torch.callbacks import Callback
from bert4torch.losses import CausalLMLoss
import torch
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import json
from tqdm import tqdm

# 基本参数
maxlen = 256
batch_size = 16
epochs = 10000

# bert配置
config_path = 'E:/data/pretrain_ckpt/ethanyt/guwenbert-base/bert4torch_config.json'
checkpoint_path = 'E:/data/pretrain_ckpt/ethanyt/guwenbert-base/pytorch_model.bin'
dict_path = 'E:/data/pretrain_ckpt/ethanyt/guwenbert-base/vocab.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class MyDataset(ListDataset):
    @staticmethod
    def load_data(file_paths):
        result = []
        for file_path in file_paths:
            data = open(file_path, encoding='utf-8').readlines()
            for line in tqdm(data, desc=file_path.split('\\')[-1]):
                result.append(json.loads(line))
        return result

def collate_fn(batch):
    """单条样本格式：[CLS]篇章[SEP]答案[SEP]问题[SEP]
    """
    batch_token_ids, batch_segment_ids = [], []
    for line in batch:
        token_ids, segment_ids = tokenizer.encode(line['content'], line['title'], maxlen=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    padd_mask = (batch_token_ids != tokenizer._token_pad_id).long()
    return [batch_token_ids, batch_segment_ids], [batch_token_ids, batch_segment_ids * padd_mask]

train_dataloader = DataLoader(MyDataset(glob.glob('F:/data/corpus/sentence_classification/THUCNews/*.jsonl')), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

model = build_transformer_model(
    config_path,
    checkpoint_path,
    with_mlm=True,
    application='unilm',
    add_trainer=True,
    use_segment_embedding=False,
).to(device)
# summary(model, input_data=[next(iter(train_dataloader))[0]])

model.compile(loss=CausalLMLoss(offset=True, logits_index=1, ignore_index=0), optimizer=optim.Adam(model.parameters(), 1e-5))

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = torch.cat([token_ids, output_ids], 1)
        segment_ids = torch.cat([segment_ids, torch.ones_like(output_ids, device=device)], 1)
        _, y_pred = model.predict([token_ids, segment_ids])
        return y_pred[:, -1, :]

    def generate(self, text, top_k=1, top_p=0.95):
        max_c_len = maxlen - self.max_new_tokens
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids], top_k=top_k)[0]  # 基于beam search
        return tokenizer.decode(output_ids.cpu().numpy())


autotitle = AutoTitle(bos_token_id=None, eos_token_id=tokenizer._token_end_id, max_new_tokens=32, device=device)


def just_show():
    s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及 时 就医 。'
    s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午 ，华 住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'
    for s in [s1, s2]:
        print(u'生成标题:', autotitle.generate(s))

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
    just_show()
    evaluator = Evaluator()

    model.fit(
        train_dataloader,
        steps_per_epoch=None,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    model.load_weights('./best_model.pt')

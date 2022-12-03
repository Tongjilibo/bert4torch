#! -*- coding: utf-8 -*-
# guwenbert做Seq2Seq任务，采用UNILM方案，由于type_token_ids的权重为[1, hdsz]的全0向量，因此在指定use_segment_embedding=False
# 即传入segment_ids但是仅仅用于生成unilm的mask，并不经过segment_embedding层
# 介绍链接：https://kexue.fm/archives/6933

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.snippets import sequence_padding, text_segmentate
from bert4torch.snippets import AutoRegressiveDecoder, Callback, ListDataset
import torch
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob

# 基本参数
maxlen = 256
batch_size = 16
epochs = 10000

# bert配置
config_path = 'F:/Projects/pretrain_ckpt/robert/[guwen_hf_torch_base]--ethanyt-guwenbert-base/config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/robert/[guwen_hf_torch_base]--ethanyt-guwenbert-base/bert4torch_pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/robert/[guwen_hf_torch_base]--ethanyt-guwenbert-base/vocab.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def collate_fn(batch):
    """单条样本格式：[CLS]篇章[SEP]答案[SEP]问题[SEP]
    """
    batch_token_ids, batch_segment_ids = [], []
    for txt in batch:
        text = open(txt, encoding='utf-8').read()
        text = text.split('\n')
        if len(text) > 1:
            title = text[0]
            content = '\n'.join(text[1:])
            token_ids, segment_ids = tokenizer.encode(content, title, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, value=tokenizer._token_pad_id), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids, value=tokenizer._token_pad_id), dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], [batch_token_ids, batch_segment_ids]

train_dataloader = DataLoader(ListDataset(glob.glob('F:/Projects/data/corpus/sentence_classification/THUCNews/*/*.txt')), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

model = build_transformer_model(
    config_path,
    checkpoint_path,
    with_mlm=True,
    application='unilm',
    trainer=True,
    token_pad_ids=tokenizer._token_pad_id, 
    use_segment_embedding=False,
    custom_position_ids='start_at_padding'
).to(device)
# summary(model.module, input_data=[next(iter(train_dataloader))[0]])

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, outputs, target):
        '''
        y_pred: [btz, seq_len, vocab_size]
        targets: y_true, y_segment
        unilm式样，需要手动把非seq2seq部分mask掉
        '''
        _, y_pred = outputs
        y_true, y_unilm_mask = target
        y_true = y_true[:, 1:]# 目标token_ids
        y_unilm_mask = y_unilm_mask[:, 1:]  # y_mask
        y_pred = y_pred[:, :-1, :]  # 预测序列，错开一位
        
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_padd_mask = (y_true != tokenizer._token_pad_id).long()
        y_true = (y_true*y_padd_mask*y_unilm_mask).flatten()
        return super().forward(y_pred, y_true)
model.compile(loss=CrossEntropyLoss(ignore_index=0), optimizer=optim.Adam(model.parameters(), 1e-5))

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

    def generate(self, text, topk=1, topp=0.95):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids], topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids.cpu().numpy())


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=32, device=device)


def just_show():
    s1 = u'白日依山尽'
    s2 = u'水光潋滟晴方好'
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

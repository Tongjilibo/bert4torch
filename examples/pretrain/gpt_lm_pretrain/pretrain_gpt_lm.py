#! -*- coding: utf-8 -*-
# 中文GPT模型预训练

from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding, ListDataset
from bert4torch.callbacks import Callback
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.generation import AutoRegressiveDecoder
from bert4torch.losses import CausalLMLoss
import glob
import json
from tqdm import tqdm

# 基本参数
maxlen = 256
batch_size = 8
epochs = 10000

# 模型配置
root_path = 'E:/pretrain_ckpt/gpt/thu-coai@CDial-GPT_LCCC-base/'
config_path = root_path + 'bert4torch_config.json'
checkpoint_path = root_path + 'pytorch_model.bin'
dict_path = root_path + 'bert4torch_vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

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
    return [batch_token_ids, batch_segment_ids], batch_token_ids

train_dataloader = DataLoader(MyDataset(glob.glob('F:/data/corpus/sentence_classification/THUCNews/*.jsonl')), 
                   batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 

encoder = build_transformer_model(config_path, checkpoint_path, add_trainer=True).to(device)  # 建立模型，加载权重

encoder.compile(loss=CausalLMLoss(offset=True, ignore_index=0), optimizer=optim.Adam(encoder.parameters(), 1e-5))

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = torch.cat([token_ids, output_ids], 1)
        segment_ids = torch.cat([segment_ids, torch.ones_like(output_ids, device=device)], 1)
        y_pred = encoder.predict([token_ids, segment_ids])
        return y_pred[:, -1, :]

    def generate(self, text, top_k=1, top_p=0.95):
        max_c_len = maxlen - self.max_new_tokens
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids], top_k=top_k)[0]  # 基于beam search
        return tokenizer.decode(output_ids.cpu().numpy())


autotitle = AutoTitle(bos_token_id=None, eos_token_id=tokenizer._token_end_id, max_new_tokens=32, device=device)


def just_show():
    s1 = u'别爱我没结果'
    s2 = u'你这样会失去我的'
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

    encoder.fit(
        train_dataloader,
        steps_per_epoch=None,
        epochs=epochs,
        callbacks=[]
    )

else:
    encoder.load_weights('./best_model.pt')

#! -*- coding: utf-8 -*-
# 加载前注意修改config文件
# 基于PromptCLUE_base_v1-5的微调
# huggingface: https://huggingface.co/ClueAI/PromptCLUE-base-v1-5
# 数据样例: https://github.com/CLUEbenchmark/pCLUE
# ps: 因bert4torch 0.2.7的seq2seq代码有bug, 所以该脚本finetune仅在>=0.2.7.post2可正常运行
# ps: 这里的evaluator没做具体metrics, 需要的可以参考https://github.com/CLUEbenchmark/pCLUE/blob/main/evaluate_pclue.py

import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import SpTokenizer
from bert4torch.snippets import AutoRegressiveDecoder, sequence_padding, Callback, ListDataset

# 配置
pretrain_model = './pt_clueAI-PromptCLUE-base-v1-5/'
config_path = pretrain_model + 'bert4torch_config.json'
checkpoint_path = pretrain_model + 'pytorch_model.bin'
spm_path = pretrain_model + 'spiece.model'

token_pad_ids = -100
max_i_len = 512
max_t_len = 128
batch_size = 32
epochs = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 分词器
tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>', keep_accents=True)

# model
model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='mt5.1.1',
    segment_vocab_size=0,
    logit_scale=False,
    token_pad_ids=token_pad_ids,  # 为了不和decoder_start_ids=0冲突
    add_trainer=True
).to(device)


# loss
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, outputs, y_true):
        _, _, y_pred = outputs
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        return super().forward(y_pred, y_true)


# collate_fn
def collate_fn(batch):
    """
    格式为:

    """
    batch_input_ids, batch_target_ids = [], []

    for input_text, target_text in batch:
        token_ids, _ = tokenizer.encode(input_text, maxlen=max_i_len)
        batch_input_ids.append(token_ids)

        token_ids, _ = tokenizer.encode(target_text, maxlen=max_t_len)
        batch_target_ids.append([0] + token_ids)

    batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, value=token_pad_ids),
                                   dtype=torch.long,
                                   device=device)

    batch_target_ids = torch.tensor(sequence_padding(batch_target_ids, value=token_pad_ids),
                                    dtype=torch.long,
                                    device=device)

    return [[batch_input_ids], [batch_target_ids[:, :-1]]], batch_target_ids[:, 1:].flatten()


class MyDataset(ListDataset):
    """
    e.g.
    {"input": "问题：：啊哈哈哈？\n答案： ", "target": "是", "answer_choices":["是", "否"], "type": "classify"}

    """

    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                input_text, target_text = l['input'], l['target']
                D.append((input_text, target_text))

        return D


# compile
model.compile(loss=CrossEntropyLoss(ignore_index=token_pad_ids),
              # mixed_precision=True, 开amp loss直接nan
              optimizer=optim.Adam(model.parameters(), lr=2e-15))


# 解码器
class AutoTitle(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        return model.decoder.predict([output_ids] + inputs)[-1][:, -1, :]

    def generate(self, text, topk=1):
        token_ids, _ = tokenizer.encode(text, maxlen=max_i_len)
        token_ids = torch.tensor([token_ids], device=device)
        encoder_output = model.encoder.predict([token_ids])
        # 基于beam search
        output_ids = self.beam_search(encoder_output, topk=topk)
        return tokenizer.decode([int(i) for i in output_ids.cpu().numpy()])


autogen = AutoTitle(start_id=0, end_id=tokenizer._token_end_id, maxlen=max_t_len, device=device)


def just_show():
    input_text = """
    假设”就在这个时候,王琪瑶已经坐在了蒋丽丽的床边“我们可以推断“蒋丽丽没有床”?选项：是的,不是,或也许？\n答案：
    """
    print(input_text + autogen.generate(text=input_text))


class Evaluator(Callback):
    def on_epoch_end(self, global_step, epoch, logs=None):
        model.save_weights('./best_promptclue_base.pt')
        just_show()


if __name__ == '__main__':
    evaluator = Evaluator()

    train_dataloader = DataLoader(MyDataset('./pCLUE_test_mine.json'),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)

    model.fit(train_dataloader=train_dataloader,
              steps_per_epoch=None,
              epochs=epochs,
              callbacks=[evaluator])

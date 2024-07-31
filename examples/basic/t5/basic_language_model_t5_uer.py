#! -*- coding: utf-8 -*-
# 调用预训练的t5-chinese模型直接做预测,使用的BertTokenizer
# t5使用的是t5.1.0的结构

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, load_vocab
from bert4torch.generation import AutoRegressiveDecoder

config_path = 'E:/data/pretrain_ckpt/t5/uer@t5-small-chinese-cluecorpussmall/bert4torch_config.json'
checkpoint_path = 'E:/data/pretrain_ckpt/t5/uer@t5-small-chinese-cluecorpussmall/pytorch_model.bin'
dict_path = 'E:/data/pretrain_ckpt/t5/uer@t5-small-chinese-cluecorpussmall/vocab.txt'

# config_path = 'E:/data/pretrain_ckpt/t5/uer@t5-base-chinese-cluecorpussmall/bert4torch_config.json'
# checkpoint_path = 'E:/data/pretrain_ckpt/t5/uer@t5-base-chinese-cluecorpussmall/pytorch_model.bin'
# dict_path = 'E:/data/pretrain_ckpt/t5/uer@t5-base-chinese-cluecorpussmall/vocab.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载并精简词表，建立分词器
token_dict = load_vocab(
    dict_path=dict_path,
    simplified=False,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

model = build_transformer_model(config_path, checkpoint_path).to(device)

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids = inputs[0] 
        res = model.predict([[token_ids], [output_ids]])
        return res[-1][:, -1, :] if isinstance(res, list) else res[:, -1, :]  # 保留最后一位

    def generate(self, text, top_k=1, top_p=0.95):
        token_ids, _ = tokenizer.encode(text, maxlen=256)
        output_ids = self.beam_search([token_ids], top_k=top_k)[0]  # 基于beam search
        return tokenizer.decode(output_ids.cpu().numpy())

autotitle = AutoTitle(bos_token_id=tokenizer._token_start_id, eos_token_id=1, max_new_tokens=32, device=device)  # 这里end_id可以设置为tokenizer._token_end_id这样结果更短

if __name__ == '__main__':
    print(autotitle.generate('中国的首都是extra0京'))


#! -*- coding: utf-8 -*-
# 基本测试：中文GPT模型，base版本，CDial-GPT版
# 项目链接：https://github.com/thu-coai/CDial-GPT
# 参考项目：https://github.com/bojone/CDial-GPT-tf

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.generation import AutoRegressiveDecoder

root_path = 'E:/pretrain_ckpt/gpt/thu-coai@CDial-GPT_LCCC-base'
# root_path = 'E:/pretrain_ckpt/gpt/thu-coai@CDial-GPT_LCCC-large'

config_path = root_path + '/bert4torch_config.json'
checkpoint_path = root_path + '/pytorch_model.bin'
dict_path = root_path + '/bert4torch_vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
speakers = [tokenizer.token_to_id('[speaker1]'), tokenizer.token_to_id('[speaker2]')]

# config中设置shared_segment_embeddings=True，segment embedding用word embedding的权重生成
encoder = build_transformer_model(config_path, checkpoint_path).to(device)  # 建立模型，加载权重

class ChatBot(AutoRegressiveDecoder):
    """基于随机采样的闲聊回复
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        curr_segment_ids = torch.zeros_like(output_ids) + token_ids[0, -1]
        token_ids = torch.cat([token_ids, output_ids], 1)
        segment_ids = torch.cat([segment_ids, curr_segment_ids], 1)
        logits = encoder.predict([token_ids, segment_ids])
        return logits[:, -1, :]

    def response(self, texts, n=1, topk=5):
        token_ids = [tokenizer._token_start_id, speakers[0]]
        segment_ids = [tokenizer._token_start_id, speakers[0]]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:-1] + [speakers[(i + 1) % 2]]
            token_ids.extend(ids)
            segment_ids.extend([speakers[i % 2]] * len(ids))
            segment_ids[-1] = speakers[(i + 1) % 2]
        results = self.random_sample([token_ids, segment_ids], n=n, topk=topk)  # 基于随机采样
        return tokenizer.decode(results[0].cpu().numpy())


chatbot  = ChatBot(bos_token_id=None, eos_token_id=tokenizer._token_end_id, max_new_tokens=32, device=device)

print(chatbot.response([u'别爱我没结果', u'你这样会失去我的', u'失去了又能怎样']))
"""
回复是随机的，例如：你还有我 | 那就不要爱我 | 你是不是傻 | 等等。
"""
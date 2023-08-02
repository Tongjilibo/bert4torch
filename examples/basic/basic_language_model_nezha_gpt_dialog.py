#! -*- coding: utf-8 -*-
# NEZHA模型做闲聊任务，这里只提供了测试脚本
# 源项目：https://github.com/bojone/nezha_gpt_dialog
# 权重转换脚本见：https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_nezha_gpt_dialog.py

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.generation import AutoRegressiveDecoder
import torch

# nezha配置
config_path = 'E:/pretrain_ckpt/nezha/[sushen_tf_base]--nezha_gpt_dialog/config.json'
checkpoint_path = 'E:/pretrain_ckpt/nezha/[sushen_tf_base]--nezha_gpt_dialog/pytorch_model.bin'
dict_path = 'E:/pretrain_ckpt/nezha/[sushen_tf_base]--nezha_gpt_dialog/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 建立并加载模型
model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='nezha',
    application='lm',
)


class ChatBot(AutoRegressiveDecoder):
    """基于随机采样对话机器人
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = torch.concat([token_ids, output_ids], 1)
        curr_segment_ids = torch.ones_like(output_ids) - segment_ids[0, -1]
        segment_ids = torch.concat([segment_ids, curr_segment_ids], 1)
        return model.predict([token_ids, segment_ids])[-1][:, -1]

    def response(self, texts, topk=5):
        token_ids, segment_ids = [tokenizer._token_start_id], [0]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:]
            token_ids.extend(ids)
            segment_ids.extend([i % 2] * len(ids))
        results = self.random_sample([token_ids, segment_ids], n=1, topk=topk)
        return tokenizer.decode(results[0].cpu().numpy())


chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)
print(chatbot.response([u'别爱我没结果', u'你这样会失去我的', u'失去了又能怎样']))
"""
回复是随机的，例如：那你还爱我吗 | 不知道 | 爱情是不是不能因为一点小事就否定了 | 我会一直爱你，你一个人会很辛苦 | 等等。
"""
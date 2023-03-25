#! -*- coding: utf-8 -*-
# 基本测试：chatglm的对话测试
# 转换脚本：https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_chatglm.py
# fp16半精度下显存占用14G

import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer
from bert4torch.snippets import AutoRegressiveDecoder

dir_path = "F:/Projects/pretrain_ckpt/chatglm/6B"
config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = dir_path + '/bert4torch_pytorch_model.bin'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=True)
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='glm').half().to(device)  # 建立模型，加载权重

class Chat(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = torch.cat([inputs[0], output_ids], 1)
        logits = model.predict([token_ids])
        return logits[:, -1, :]

    def generate(self, text, n=1, topp=0.95):
        token_ids = tokenizer.encode(text)
        results = self.random_sample([token_ids], n, topp=topp)  # 基于随机采样
        return [text + tokenizer.decode(ids.cpu().numpy()) for ids in results]


article_completion = Chat(
    start_id=None,
    end_id=150005,  # eos标记
    maxlen=256,
    minlen=20,
    device=device
)

for text in [u'你好']:
    print(article_completion.generate(text))
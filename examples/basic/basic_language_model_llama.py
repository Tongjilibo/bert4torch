#! -*- coding: utf-8 -*-
# 基本测试：llama的7b模型的测试, 调试中

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import SpTokenizer
from bert4torch.snippets import AutoRegressiveDecoder

config_path = 'F:/Projects/pretrain_ckpt/llama/7B/bert4torch_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/llama/7B/bert4torch_pytorch_model.bin'
spm_path = 'F:/Projects/pretrain_ckpt/llama/tokenizer.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = SpTokenizer(spm_path, token_start='<s>', token_end=None, keep_accents=True)

model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='llama').to(device)  # 建立模型，加载权重

class ArticleCompletion(AutoRegressiveDecoder):
    """基于随机采样的文章续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = torch.cat([inputs[0], output_ids], 1)
        logits = model.predict([token_ids])
        return logits[:, -1, :]

    def generate(self, text, n=1, topp=0.95):
        token_ids, _ = tokenizer.encode(text)
        results = self.random_sample([token_ids], n, topp=topp)  # 基于随机采样
        return [text + tokenizer.decode(ids.cpu().numpy()) for ids in results]


article_completion = ArticleCompletion(
    start_id=None,
    end_id=2,
    maxlen=100,
    minlen=50,
    device=device
)

for text in [u'I believe the meaning of life is']:
    print(article_completion.generate(text))
#! -*- coding: utf-8 -*-
# 基本测试：belle-7b模型的基本测试
# belle模型：https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import SpTokenizer
from bert4torch.snippets import AutoRegressiveDecoder, SeqGeneration

config_path = 'F:/Projects/pretrain_ckpt/belle_llama_7b/bert4torch_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/belle_llama_7b/bert4torch_pytorch_model.bin'
spm_path = 'F:/Projects/pretrain_ckpt/belle_llama_7b/tokenizer.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = SpTokenizer(spm_path, token_start='<s>', token_end=None, keep_accents=True)

model = build_transformer_model(config_path=config_path,
                                checkpoint_path=checkpoint_path,
                                model='llama').half().to(device)  # 建立模型，加载权重


# 第一种方式
class ArticleCompletion(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids = torch.cat([inputs[0], output_ids], 1)
        logits = model.predict([token_ids])
        return logits[:, -1, :]

    def generate(self, text, n=1, topp=0.95, add_input=True):
        token_ids, _ = tokenizer.encode(text)
        results = self.random_sample([token_ids], n, topp=topp)  # 基于随机采样
        text = text if add_input else ''
        return [text + tokenizer.decode(ids.cpu().numpy()) for ids in results]


article_completion = ArticleCompletion(
    start_id=None,
    end_id=2,  # </s>标记
    maxlen=256,
    minlen=20,
    device=device
)

# 第二种方式
article_completion = SeqGeneration(model, tokenizer, start_id=None, end_id=2, mode='random_sample',
                                   maxlen=256, default_rtype='logits', use_states=True)

# 必须指定human + assistant的prompt
text = "Human: 写一首中文歌曲，赞美大自然 \n\nAssistant: "
print(article_completion.generate(text, add_input=True))

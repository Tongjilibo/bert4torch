#! -*- coding: utf-8 -*-
# 基本测试：llama的7b模型的测试, fp32精度的单卡占用约27g，fp16的显存占用约14g
# 使用前需要安装最新版本的bert4torch并进行权重转换 https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_llama_facebook.py

# 0. Install lastest bert4torch: `pip install git+https://www.github.com/Tongjilibo/bert4torch.git` or git clone
# 1. Download weights：[Github](https://github.com/facebookresearch/llama) | [huggingface](https://huggingface.co/decapoda-research/llama-7b-hf) | [torrent](https://pan.baidu.com/s/1yBaYZK5LHIbJyCCbtFLW3A?pwd=phhd)，本人实现是基于第三种
# 2. Convert weights：https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_llama_facebook.py
# 3. Inference script：https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_llama.py
# 4. VRAM request in single gpu：fp32 27G, fp16 14g

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import SpTokenizer
from bert4torch.snippets import AutoRegressiveDecoder, SeqGeneration

config_path = 'F:/Projects/pretrain_ckpt/llama/7B/bert4torch_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/llama/7B/bert4torch_pytorch_model.bin'
spm_path = 'F:/Projects/pretrain_ckpt/llama/tokenizer.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = SpTokenizer(spm_path, token_start='<s>', token_end=None, keep_accents=True)

model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='llama').half().to(device)  # 建立模型，加载权重

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

for text in [u'I believe the meaning of life is ']:
    print(article_completion.generate(text, add_input=True))
#! -*- coding: utf-8 -*-
# 基本测试：uer的gpt2 chinese的效果测试
# 项目链接：https://huggingface.co/uer/gpt2-chinese-cluecorpussmall

ckpt_dir = 'E:/data/pretrain_ckpt/uer/gpt2-chinese-cluecorpussmall/'
texts = ['这是很久之前的事情了', '话说当年']

# ===============transformers======================
# from transformers import BertTokenizer, GPT2LMHeadModel, TextGeneration
# tokenizer = BertTokenizer.from_pretrained(ckpt_dir)
# model = GPT2LMHeadModel.from_pretrained(ckpt_dir)
# text_generator = TextGeneration(model, tokenizer)   
# output = text_generator(texts, max_length=100, do_sample=True)
# print('====transformers结果====')
# print(output)

# ===============bert4torch======================
import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.generation import AutoRegressiveDecoder, SeqGeneration
import os

config_path = ckpt_dir + 'bert4torch_config.json'
checkpoint_path = ckpt_dir + 'pytorch_model.bin'
dict_path = ckpt_dir + 'vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
top_k = 50
eos_token_id = 50256  # 50256:open-end generation, 511:表示句号
mode = 'random_sample'

tokenizer = Tokenizer(dict_path, token_start=None, token_end=None, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path).to(device)

print('==============自定义单条样本================')
class ArticleCompletion(AutoRegressiveDecoder):
    """基于随机采样的文章续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits', use_states=False)
    def predict(self, inputs, output_ids, states):
        token_ids = torch.cat([inputs[0], output_ids], 1)
        logits = model.predict([token_ids])
        return logits[:, -1, :]

    def generate(self, text, n=1, top_p=0.7, top_k=50, add_input=True):
        token_ids, _ = tokenizer.encode(text)
        results = self.random_sample([token_ids], n=n, top_p=top_p, top_k=top_k)  # 基于随机采样
        add_input = text if add_input else ''
        return [text + tokenizer.decode(ids.cpu().numpy()) for ids in results]
article_completion = ArticleCompletion(bos_token_id=None, eos_token_id=eos_token_id, max_length=100, device=device)
for text in texts:
    print(article_completion.generate(text, n=1, top_k=top_k))


print('==============默认单条无cache================')
article_completion = SeqGeneration(model, tokenizer, bos_token_id=None, eos_token_id=eos_token_id, mode=mode,
                                   max_length=100, default_rtype='logits', use_states=False)
for text in texts:
    print(text + article_completion.generate(text, n=1, top_k=top_k))


print('==============默认单条cache================')
article_completion = SeqGeneration(model, tokenizer, bos_token_id=None, eos_token_id=eos_token_id, mode=mode,
                                   max_length=100, default_rtype='logits', use_states=True)
for text in texts:
    print(text + article_completion.generate(text, n=1, top_k=top_k))


print('==============默认batch 无cache================')
article_completion = SeqGeneration(model, tokenizer, bos_token_id=None, eos_token_id=eos_token_id, mode=mode,
                                   max_length=100, default_rtype='logits', use_states=False)
results = article_completion.generate(texts, top_k=top_k)
for text, result in zip(texts, results):
    print(text + result)


print('==============默认batch cache================')
article_completion = SeqGeneration(model, tokenizer, bos_token_id=None, eos_token_id=eos_token_id, mode=mode,
                                   max_length=100, default_rtype='logits', use_states=True)
results = article_completion.generate(texts, top_k=top_k)
for text, result in zip(texts, results):
    print(text + result)


print('==============默认stream================')
article_completion = SeqGeneration(model, tokenizer, bos_token_id=None, eos_token_id=eos_token_id, mode=mode,
                                   max_length=100, default_rtype='logits', use_states=True)
text = texts[0]
for output in article_completion.stream_generate(text, top_k=top_k):
    os.system('clear')
    print(text+output, flush=True)


print('==============直接调用.generate()================')
generate_configs = {'tokenizer': tokenizer, 'bos_token_id': None, 'eos_token_id': eos_token_id, 'mode':mode,
                    'max_length':100, 'default_rtype':'logits', 'use_states':False, 'n':1, 'top_k':top_k, 'include_input':True}
for text in texts:
    print(model.generate(text, **generate_configs))
print(model.generate(texts, **generate_configs))  # 批量生成
for output in model.stream_generate(text, **generate_configs):
    os.system('clear')
    print(output, flush=True)

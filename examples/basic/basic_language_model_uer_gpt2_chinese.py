#! -*- coding: utf-8 -*-
# 基本测试：uer的gpt2 chinese的效果测试
# 项目链接：https://huggingface.co/uer/gpt2-chinese-cluecorpussmall
# 权重需转换后方可加载，转换脚本见convert_script文件夹

# ===============transformers======================
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("F:/Projects/pretrain_ckpt/gpt2/[uer_gpt2_torch_base]--gpt2-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("F:/Projects/pretrain_ckpt/gpt2/[uer_gpt2_torch_base]--gpt2-chinese-cluecorpussmall")
text_generator = TextGenerationPipeline(model, tokenizer)   
output = text_generator("这是很久之前的事情了", max_length=100, do_sample=True)
print('====transformers结果====')
print(output)

# ===============bert4torch======================
import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from bert4torch.snippets import AutoRegressiveDecoder, SeqGeneration

config_path = 'F:/Projects/pretrain_ckpt/gpt2/[uer_gpt2_torch_base]--gpt2-chinese-cluecorpussmall/bert4torch_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/gpt2/[uer_gpt2_torch_base]--gpt2-chinese-cluecorpussmall/bert4torch_pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/gpt2/[uer_gpt2_torch_base]--gpt2-chinese-cluecorpussmall/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer(dict_path, token_start=None, token_end=None, do_lower_case=True)  # 建立分词器

model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='gpt2', segment_vocab_size=0).to(device)  # 建立模型，加载权重

# 第一种方式
class ArticleCompletion(AutoRegressiveDecoder):
    """基于随机采样的文章续写
    """
    @AutoRegressiveDecoder.wraps(default_rtype='logits', use_states=False)
    def predict(self, inputs, output_ids, states):
        token_ids = torch.cat([inputs[0], output_ids], 1)
        logits = model.predict([token_ids])
        return logits[:, -1, :]

    def generate(self, text, n=1, topp=0.7, add_input=True):
        token_ids, _ = tokenizer.encode(text)
        results = self.random_sample([token_ids], n, topp=topp)  # 基于随机采样
        add_input = text if add_input else ''
        return [text + tokenizer.decode(ids.cpu().numpy()) for ids in results]

article_completion = ArticleCompletion(
    start_id=None,
    end_id=50256,
    maxlen=100,
    device=device
)

# 第二种方式
article_completion = SeqGeneration(model, tokenizer, start_id=None, end_id=50256, mode='random_sample',
                                   maxlen=100, default_rtype='logits', use_states=True)

print('====bert4torch结果====')
for text in [u'这是很久之前的事情了']:
    print(article_completion.generate(text, add_input=True))
    
"""
部分结果：
>>> article_completion.generate(u'这是很久之前的事情了')

"""

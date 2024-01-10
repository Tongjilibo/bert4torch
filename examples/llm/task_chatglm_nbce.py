#! -*- coding: utf-8 -*-
# Naive Bayes-based Context Extension (NBCE)
# 使用朴素贝叶斯增加LLM的Context处理长度
# 链接：https://kexue.fm/archives/9617

import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer
import re
import json
import os

choice = 'default'  # default, int4, int8, v1.1.0
if choice == 'default':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B"
    config_path = dir_path + '/bert4torch_config.json'
    checkpoint_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('.bin')]
elif choice == 'v1.1.0':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-v1_1_0"
    config_path = dir_path + '/bert4torch_config.json'
    checkpoint_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('.bin')]
elif choice == 'int4':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-int4"
    config_path = dir_path + '/bert4torch_config.json'
    checkpoint_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('.bin')]
elif choice == 'int8':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-int8"
    config_path = dir_path + '/bert4torch_config.json'
    checkpoint_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('.bin')]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(dir_path.replace('/', '\\'), trust_remote_code=True)
tokenizer.padding_side = 'left'

# 加载chatglm-6b模型
# 建立模型，加载权重
if choice in {'default', 'v1.1.0'}:
    model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path).half()
    model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8).to(device)
else:
    # 在config中已经写入了量化的配置参数
    model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path).to(device)
model.eval()

# 加载示例Context
contexts = json.load(open('E:/Github/bert4torch/examples/datasets/nbce_contexts.json', encoding='utf-8'))

# 示例问题集（一次性问多个问题，NBCE自行根据Context逐一输出答案）
question = """请仔细阅读材料，回答下面问题：
- 创新药新巨头吉利德公司有多少个员工？
- 领英计划裁员多少人？
"""

# 拼接context和question
contexts = [''] + contexts  # 添加空Context（无Context预测）
batch = ['''
要求： 基于已知内容，请用中文以要求的格式简短直接地回答用户的问题。

已知内容： %s

问题:  %s
''' % (context, question) for context in contexts]

print('Context长度分布：', [len(text) for text in batch])
print('Context总长度：', sum([len(text) for text in batch]))
# print(batch)

@torch.inference_mode()
def generate(max_tokens):
    """Naive Bayes-based Context Extension 演示代码
    """
    eop_list= []

    inputs = tokenizer(batch, padding='longest', return_tensors='pt', return_attention_mask=True, skip_special_tokens=True).to(device)
    input_ids = past_token_ids = inputs.input_ids

    res = ''
    n = input_ids.shape[0]
    past_key_values = None
    for i in range(max_tokens):

        # 模型输出
        #print(f'第{i+1}token开始输出')
        logits, model_kwargs = model(input_ids,
                        past_key_values=past_key_values,
                        past_token_ids=past_token_ids,
                        use_states=True
                       )
        past_key_values = model_kwargs['past_key_values']
        torch.cuda.empty_cache()

        # ===== 核心代码开始 =====
        beta = 0.25

        logits = logits[:, -1]
        logits -= torch.max(logits,dim=1).values.reshape(logits.shape[0],-1)
        probas = torch.nn.functional.softmax(logits.float(), dim=-1)

        logits = probas.log()
        k = (probas * logits).sum(dim=-1)[1:].argmax() + 1
        logits_max = logits[k]
        logits_uncond = logits[0]
        logits = (1 + beta) * logits_max - beta * logits_uncond
        # ===== 核心代码结束 =====

        # 构建分布，采样
        # tau = 0.01  # tau = 1是标准的随机采样，tau->0则是贪心搜索

        probas = torch.nn.functional.softmax(logits[None], dim=-1)
        next_tokens = torch.topk(probas,1).indices

        s = tokenizer.convert_ids_to_tokens(next_tokens)
        res += s[0]
        if s[0] == '<eop>':
            if len(eop_list)==3:
                break
            else:
                eop_list.append('<eop>')
        else:
            eop_list = []
        # prepare for next iteration
        input_ids = next_tokens.tile(n, 1)
        past_token_ids = torch.cat([past_token_ids, input_ids], dim=1)

    print('==================question===================')
    print(question)

    print('===================answer====================')
    print(re.sub('<n>+', '\n', re.sub('▁|<eop>|<sop>','',res)))
    #['据公开报道,截至2021年6月,吉利德公司有约16,000名员工。', '领英计划裁员716人。', 'Pharmasset被吉利德以110亿美元收购。']


if __name__ == '__main__':
    generate(1000)
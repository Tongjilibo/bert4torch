#! -*- coding: utf-8 -*-
'''
# 调用transformer_xl模型，该模型流行度较低，未找到中文预训练模型
# last_hidden_state目前是debug到transformer包中查看，经比对和本框架一致
# 用的是transformer中的英文预训练模型来验证正确性

- [权重链接](https://huggingface.co/transfo-xl-wt103)
- 该项目是英文的：只用于bert4torch中transformer_xl的调试模型结构，并未实际用于finetune
'''

from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

pretrained_model = "/data/pretrain_ckpt/transfo-xl/transfo-xl-wt103"

try:
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
except:
    inputs = {'input_ids': torch.tensor([[14049,     2,   617,  3225,    23, 16072]])}


# ----------------------bert4torch配置----------------------

model = build_transformer_model(pretrained_model)
print('bert4torch last_hidden_state: \n', model.predict([inputs['input_ids']]))
# tensor([[[ 0.1027,  0.0604, -0.2585,  ...,  0.3137, -0.2679,  0.1036],
#          [ 0.3482, -0.0458, -0.4582,  ...,  0.0242, -0.0721,  0.2311],
#          [ 0.3426, -0.1353, -0.4145,  ...,  0.1123,  0.1374,  0.1313],
#          [ 0.0038, -0.0978, -0.5570,  ...,  0.0487, -0.1891, -0.0608],
#          [-0.2155, -0.1388, -0.5549,  ..., -0.1458,  0.0774,  0.0419],
#          [ 0.0967, -0.1781, -0.4328,  ..., -0.1831, -0.0808,  0.0890]]])


# ----------------------transformers包----------------------
model = AutoModelForCausalLM.from_pretrained(pretrained_model)
model.eval()
with torch.no_grad():
    # 这里只能断点进去看
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.losses
print('transforms loss: ', loss)
# tensor([[ 3.8697,  8.4586, 11.3868,  5.4984, 10.7303]])

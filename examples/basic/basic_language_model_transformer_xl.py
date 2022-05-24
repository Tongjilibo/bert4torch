#! -*- coding: utf-8 -*-
# 调用transformer_xl模型，该模型流行度较低，未找到中文预训练模型
# last_hidden_state目前是debug到transformer包中查看，经比对和本框架一致
# 用的是transformer中的英文预训练模型来验证正确性
# 转换脚本: convert_script/convert_transformer_xl.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

pretrained_model = "F:/Projects/pretrain_ckpt/transformer_xl/[english_hugging_face_torch]--transfo-xl-wt103"

# ----------------------transformers包----------------------
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelForCausalLM.from_pretrained(pretrained_model)
model.eval()
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
with torch.no_grad():
    # 这里只能断点进去看
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.losses
print('transforms loss: ', loss)

# ----------------------bert4torch配置----------------------
from bert4torch.models import build_transformer_model
config_path = f'{pretrained_model}/bert4torch_config.json'
checkpoint_path = f'{pretrained_model}/bert4torch_pytorch_model.bin'

model = build_transformer_model(
    config_path,
    checkpoint_path=checkpoint_path,
    model='transformer_xl',
    segment_vocab_size=0,
)

print('bert4torch last_hidden_state: ', model.predict([inputs['input_ids']]))
# tensor([[[ 0.1027,  0.0604, -0.2585,  ...,  0.3137, -0.2679,  0.1036],
#          [ 0.3482, -0.0458, -0.4582,  ...,  0.0242, -0.0721,  0.2311],
#          [ 0.3426, -0.1353, -0.4145,  ...,  0.1123,  0.1374,  0.1313],
#          [ 0.0038, -0.0978, -0.5570,  ...,  0.0487, -0.1891, -0.0608],
#          [-0.2155, -0.1388, -0.5549,  ..., -0.1458,  0.0774,  0.0419],
#          [ 0.0967, -0.1781, -0.4328,  ..., -0.1831, -0.0808,  0.0890]]])
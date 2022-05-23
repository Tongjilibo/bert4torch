#! -*- coding: utf-8 -*-
# 调用transformer_xl模型，该模型流行度较低，未找到中文预训练模型
# 用的是transformer中的英文预训练模型来验证正确性

from transformers import AutoTokenizer, AutoModelForCausalLM

# transformers包
pretrained_model = "F:/Projects/pretrain_ckpt/transformer_xl/[english_hugging_face_torch]--transfo-xl-wt103"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

model = AutoModelForCausalLM.from_pretrained(pretrained_model)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.losses
print('transforms loss: ', loss)

# bert4torch配置
from bert4torch.models import build_transformer_model
config_path = f'{pretrained_model}/bert4torch_config.json'
checkpoint_path = f'{pretrained_model}/pytorch_model.bin'

model = build_transformer_model(
    config_path,
    checkpoint_path=None,
    model='transformer_xl',
    segment_vocab_size=0,
)

print(model.predict([inputs['input_ids']]))

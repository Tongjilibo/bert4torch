'''基础测试：modernbert的mlm预测'''
#! -*- coding: utf-8 -*-
from transformers import AutoTokenizer
from bert4torch.models import build_transformer_model
import torch

model_dir = "E:/data/pretrain_ckpt/ModernBERT/answerdotai@ModernBERT-base"
# model_dir = "E:/data/pretrain_ckpt/ModernBERT/answerdotai@ModernBERT-large"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
text = "The capital of France is [MASK]."

# ==========================bert4torch调用=========================
# 建立分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir, with_mlm=True).to(device)  # 建立模型，加载权重

inputs = tokenizer(text, return_tensors="pt").to(device)

# 需要传入参数with_mlm
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

    masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
    predicted_token_id = outputs[-1][0, masked_index].argmax(axis=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    print("Predicted token:", predicted_token)
    # Predicted token:  Paris

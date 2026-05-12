'''基础测试：modernbert的mlm预测'''
#! -*- coding: utf-8 -*-
from bert4torch.models import AutoTokenizer, build_transformer_model
import torch

model_dir = "/data/pretrain_ckpt/answerdotai/ModernBERT-base"
# model_dir = "/data/pretrain_ckpt/answerdotai/ModernBERT-large"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
text = "The capital of France is [MASK]."

# ==========================bert4torch调用=========================
tokenizer = AutoTokenizer.from_pretrained(model_dir)
inputs = tokenizer(text, return_tensors="pt").to(device)

model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir, with_mlm=True).to(device)

# 需要传入参数with_mlm
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

    masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
    predicted_token_id = outputs[-1][0, masked_index].argmax(axis=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    print("Predicted token:", predicted_token)
    # Predicted token:  Paris

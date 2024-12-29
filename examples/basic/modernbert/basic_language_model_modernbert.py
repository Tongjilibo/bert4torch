#! -*- coding: utf-8 -*-
# 基础测试：mlm预测
from transformers import AutoTokenizer
from bert4torch.models import build_transformer_model
import torch

# 加载模型，请更换成自己的路径, 以下两个权重是一样的，一个是tf用转换命令转的，一个是hf上的bert_base_chinese
# root_model_path = "E:/data/pretrain_ckpt/bert/google@chinese_L-12_H-768_A-12"
# vocab_path = root_model_path + "/vocab.txt"
# config_path = root_model_path + "/bert4torch_config.json"
# checkpoint_path = root_model_path + '/pytorch_model.bin'

model_dir = "E:\data\pretrain_ckpt\ModernBERT\\answerdotai@ModernBERT-base"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
text = "The capital of France is [MASK]."

# ==========================bert4torch调用=========================
# 建立分词器
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir, with_mlm='softmax').to(device)  # 建立模型，加载权重

inputs = tokenizer(text, return_tensors="pt")

# 需要传入参数with_mlm
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

    masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
    predicted_token_id = outputs.logits[0, masked_index].argmax(axis=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    print("Predicted token:", predicted_token)
    # Predicted token:  Paris

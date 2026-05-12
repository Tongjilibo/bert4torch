#! -*- coding: utf-8 -*-
# 基础测试：英文mlm预测
# 权重下载链接：https://huggingface.co/roberta-base

from bert4torch import build_transformer_model, AutoTokenizer
import torch
from torch.nn.functional import softmax

# 加载模型，请更换成自己的路径
root_model_path = "/data/pretrain_ckpt/FacebookAI/roberta-base"

# 建立分词器
tokenizer = AutoTokenizer.from_pretrained(root_model_path)
input_text = "The goal of life is <mask>."


# ==========================bert4torch调用==========================
model = build_transformer_model(root_model_path, with_mlm='softmax')

token_ids = tokenizer.encode(input_text)
segments_ids = [0] * len(token_ids)

tokens_ids_tensor = torch.tensor([token_ids])
segment_ids_tensor = torch.tensor([segments_ids])

# 需要传入参数with_mlm
model.eval()
with torch.no_grad():
    _, probas = model([tokens_ids_tensor, segment_ids_tensor])
    result = torch.argmax(probas[0, -3:-2], dim=-1).numpy()
    print('====bert4torch output====')
    print(tokenizer.decode(result))


# ==========================transformer调用==========================
from transformers import AutoTokenizer, RobertaForMaskedLM
tokenizer = AutoTokenizer.from_pretrained(root_model_path)
model = RobertaForMaskedLM.from_pretrained(root_model_path)
input_ids = torch.tensor(tokenizer.encode(input_text, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=input_ids)
loss, prediction_scores = outputs[:2]
logit_prob = softmax(prediction_scores[0, -3],dim=-1).data.tolist()
predicted_index = torch.argmax(prediction_scores[0, -3]).item()
predicted_token = tokenizer.decode([predicted_index])
print('====transformers output====')
print(predicted_token, logit_prob[predicted_index])
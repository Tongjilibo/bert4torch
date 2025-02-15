#! -*- coding: utf-8 -*-
# 基础测试：mlm预测
# 链接: https://huggingface.co/bert-base-multilingual-cased


from bert4torch.models import build_transformer_model
from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.nn.functional import softmax

# bert-base-multilingual-cased
# bert_base_uncased
# bert_base_cased
root_model_path = "E:/data/pretrain_ckpt/google-bert/bert-base-multilingual-cased"
config_path = root_model_path + "/bert4torch_config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'
text = "Hello I'm a [MASK] model."

# ==========================transformer调用==========================
tokenizer = BertTokenizer.from_pretrained(root_model_path)
model = BertForMaskedLM.from_pretrained(root_model_path)
encoded_input = tokenizer(text, return_tensors='pt')
mask_pos = encoded_input['input_ids'][0].tolist().index(103)
outputs = model(**encoded_input)
prediction_scores = outputs[0]
logit_prob = softmax(prediction_scores[0, mask_pos],dim=-1).data.tolist()
predicted_index = torch.argmax(prediction_scores[0, mask_pos]).item()
predicted_token = tokenizer.decode([predicted_index])
print('====transformers output====')
print(predicted_token, logit_prob[predicted_index])


# ==========================bert4torch调用==========================
# 建立分词器
model = build_transformer_model(config_path, checkpoint_path, with_mlm='softmax')

# 需要传入参数with_mlm
model.eval()
with torch.no_grad():
    _, probas = model([encoded_input['input_ids'], encoded_input['token_type_ids']])
    result = torch.argmax(probas[0, mask_pos], dim=-1).numpy()
    print('====bert4torch output====')
    print(tokenizer.decode([result]))

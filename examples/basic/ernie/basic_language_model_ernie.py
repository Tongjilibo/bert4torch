#! -*- coding: utf-8 -*-
# 基础测试：ERNIE模型测试

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
import torch
from torch.nn.functional import softmax

# 加载模型，请更换成自己的路径
# root_model_path = "E:/data/pretrain_ckpt/ernie/baidu@ernie-1-base-zh"
root_model_path = "E:/data/pretrain_ckpt/ernie/baidu@ernie-3-base-zh"
input_text = "科学[MASK][MASK]是第一生产力"

# ==========================bert4torch调用=========================
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/bert4torch_config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'

# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path, with_mlm='softmax')  # 建立模型，加载权重

token_ids, segments_ids = tokenizer.encode(input_text)
print('====bert4torch output====')
print(''.join(tokenizer.ids_to_tokens(token_ids)))

tokens_ids_tensor = torch.tensor([token_ids])
segment_ids_tensor = torch.tensor([segments_ids])

# 需要传入参数
model.eval()
with torch.no_grad():
    logits, probas = model([tokens_ids_tensor, segment_ids_tensor])
    result = torch.argmax(probas[0, 3:5], dim=-1).numpy()
    print(tokenizer.decode(result))



# ==========================transformer调用==========================
from transformers import BertTokenizer, ErnieForMaskedLM, ErnieModel

tokenizer = BertTokenizer.from_pretrained(root_model_path)
model = ErnieForMaskedLM.from_pretrained(root_model_path)

encoded_input = tokenizer(input_text, return_tensors='pt')
outputs = model(**encoded_input)

prediction_scores = outputs['logits']
predicted_index = torch.argmax(prediction_scores[0, 3:5], dim=-1).numpy()
print('====transformers output====')
print(tokenizer.decode(predicted_index).replace(' ', ''))

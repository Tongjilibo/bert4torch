#! -*- coding: utf-8 -*-
# 基础测试：mlm测试roformer、roformer_v2模型

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
import torch

choice = 'roformer_v2'  # roformer roformer_v2
if choice == 'roformer':
    args_model_path = "F:/Projects/pretrain_ckpt/roformer/[sushen_torch_base]--roformer_v1_base/"
    args_model = 'roformer'
else:
    args_model_path = "F:/Projects/pretrain_ckpt/roformer/[sushen_torch_base]--roformer_v2_char_base/"
    args_model = 'roformer_v2'
    
# 加载模型，请更换成自己的路径
root_model_path = args_model_path
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'


# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path, model=args_model, with_mlm='softmax')  # 建立模型，加载权重

token_ids, segments_ids = tokenizer.encode("今天M很好，我M去公园玩。")
token_ids[3] = token_ids[8] = tokenizer._token_mask_id
print(''.join(tokenizer.ids_to_tokens(token_ids)))

tokens_ids_tensor = torch.tensor([token_ids])
segment_ids_tensor = torch.tensor([segments_ids])

# 需要传入参数with_mlm
model.eval()
with torch.no_grad():
    _, logits = model([tokens_ids_tensor, segment_ids_tensor])

pred_str = 'Predict: '
for i, logit in enumerate(logits[0]):
    if token_ids[i] == tokenizer._token_mask_id:
        pred_str += tokenizer.id_to_token(torch.argmax(logit, dim=-1).item())
    else:
        pred_str += tokenizer.id_to_token(token_ids[i])
print(pred_str)

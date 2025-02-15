#! -*- coding: utf-8 -*-
# 基础测试：mlm测试roformer、roformer_v2模型

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
import torch

args_model_path = "E:/data/pretrain_ckpt/junnyu/roformer_chinese_base/"
# args_model_path = "E:/data/pretrain_ckpt/junnyu/roformer_v2_chinese_char_base/"
    
# 加载模型，请更换成自己的路径
root_model_path = args_model_path
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/bert4torch_config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path, with_mlm='softmax').to(device)

token_ids, segments_ids = tokenizer.encode("今天[MASK]很好，我[MASK]去公园玩。")
print(''.join(tokenizer.ids_to_tokens(token_ids)))

tokens_ids_tensor = torch.tensor([token_ids], device=device)
segment_ids_tensor = torch.tensor([segments_ids], device=device)

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

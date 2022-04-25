#! -*- coding: utf-8 -*-
# 基础测试：GAU_alpha的mlm预测，测试效果感觉没有很好

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
import torch

# 加载模型，请更换成自己的路径
config_path = 'F:/Projects/pretrain_ckpt/gau/[sushen-torch]--chinese_GAU-alpha-char_L-24_H-768/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/gau/[sushen-torch]--chinese_GAU-alpha-char_L-24_H-768/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/gau/[sushen-torch]--chinese_GAU-alpha-char_L-24_H-768/vocab.txt'


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path, model='gau_alpha', with_mlm='linear')  # 建立模型，加载权重

token_ids, segments_ids = tokenizer.encode("科学技术是第一生产力")
token_ids[3] = token_ids[4] = tokenizer._token_mask_id
print(''.join(tokenizer.ids_to_tokens(token_ids)))

tokens_ids_tensor = torch.tensor([token_ids])
segment_ids_tensor = torch.tensor([segments_ids])

# 需要传入参数with_mlm
model.eval()
with torch.no_grad():
    _, probas = model([tokens_ids_tensor, segment_ids_tensor])
    result = torch.argmax(probas[0, 3:5], dim=-1).numpy()
    print(tokenizer.decode(result))

#! -*- coding: utf-8 -*-
# 基础测试：GAU_alpha的mlm预测，和bert4keras版本比对一致
# 测试中长文本效果明显高于短文本效果
# 博客：https://kexue.fm/archives/9052
# 权重转换脚本：./convert_script/convert_GAU_alpha.py

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
import torch

# 加载模型，请更换成自己的路径
config_path = 'F:/Projects/pretrain_ckpt/gau/[sushen-torch]--chinese_GAU-alpha-char_L-24_H-768/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/gau/[sushen-torch]--chinese_GAU-alpha-char_L-24_H-768/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/gau/[sushen-torch]--chinese_GAU-alpha-char_L-24_H-768/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path, model='gau_alpha', with_mlm='softmax').to(device)  # 建立模型，加载权重

token_ids, segments_ids = tokenizer.encode("近期正是上市公司财报密集披露的时间，但有多家龙头公司的业绩令投资者失望")
token_ids[5] = token_ids[6] = tokenizer._token_mask_id
print(''.join(tokenizer.ids_to_tokens(token_ids)))

tokens_ids_tensor = torch.tensor([token_ids], device=device)
segment_ids_tensor = torch.tensor([segments_ids], device=device)

# 需要传入参数with_mlm
model.eval()
with torch.no_grad():
    _, probas = model([tokens_ids_tensor, segment_ids_tensor])
    result = torch.argmax(probas[0, 5:7], dim=-1).cpu().numpy()
    print(tokenizer.decode(result))

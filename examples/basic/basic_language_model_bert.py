#! -*- coding: utf-8 -*-
# 基础测试：mlm预测

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
import torch

# 加载模型，请更换成自己的路径
root_model_path = "F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12"
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/bert_config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'


# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path, with_mlm='softmax')  # 建立模型，加载权重

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

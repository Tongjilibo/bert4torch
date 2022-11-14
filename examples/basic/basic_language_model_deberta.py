#! -*- coding: utf-8 -*-
# 基础测试：mlm预测

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
import torch

# 加载模型，请更换成自己的路径
root_model_path = "F:/Projects/pretrain_ckpt/deberta/[IDEA-CCNL-torch]--Erlangshen-DeBERTa-v2-320M-Chinese"
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/config.json"
checkpoint_path = root_model_path + '/bert4torch_pytorch_model.bin'


# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path, model='deberta_v2', with_mlm='softmax')  # 建立模型，加载权重

token_ids, segments_ids = tokenizer.encode("生活的真谛是爱。")
token_ids[7] = tokenizer._token_mask_id
print(''.join(tokenizer.ids_to_tokens(token_ids)))

tokens_ids_tensor = torch.tensor([token_ids])
segment_ids_tensor = torch.tensor([segments_ids])

# 需要传入参数with_mlm
model.eval()
with torch.no_grad():
    _, probas = model([tokens_ids_tensor, segment_ids_tensor])
    result = torch.argmax(probas[0, 7:8], dim=-1).numpy()
    print(tokenizer.decode(result))

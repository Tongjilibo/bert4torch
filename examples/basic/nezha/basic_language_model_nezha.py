#! -*- coding: utf-8 -*-
# 基础测试：mlm预测

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
import torch


root_model_path = "/data/pretrain_ckpt/nezha/huawei_noah@nezha-cn-base"
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/bert4torch_config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
inputtext = "今天[MASK]情很好"

# ==========================bert4torch调用=========================
# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path, with_mlm='softmax').to(device)  # 建立模型，加载权重

token_ids, segments_ids = tokenizer.encode(inputtext)
maskpos = token_ids.index(103)
tokens_ids_tensor = torch.tensor([token_ids]).to(device)
segment_ids_tensor = torch.tensor([segments_ids]).to(device)

# 需要传入参数with_mlm
model.eval()
with torch.no_grad():
    _, probas = model([tokens_ids_tensor, segment_ids_tensor])
    result = torch.argmax(probas[0, [maskpos]], dim=-1).cpu().numpy()
    print('====bert4torch output====')
    print(tokenizer.decode(result))

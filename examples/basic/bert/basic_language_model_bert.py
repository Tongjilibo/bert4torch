#! -*- coding: utf-8 -*-
# 基础测试：mlm预测

from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
import torch

# 加载模型，请更换成自己的路径, 以下两个权重是一样的，一个是tf用转换命令转的，一个是hf上的bert_base_chinese
# root_model_path = "E:/pretrain_ckpt/bert/google@chinese_L-12_H-768_A-12"
# vocab_path = root_model_path + "/vocab.txt"
# config_path = root_model_path + "/bert4torch_config.json"
# checkpoint_path = root_model_path + '/pytorch_model.bin'

root_model_path = "E:/pretrain_ckpt/bert/google@bert-base-chinese"
vocab_path = root_model_path + "/vocab.txt"
config_path = root_model_path + "/bert4torch_config.json"
checkpoint_path = root_model_path + '/pytorch_model.bin'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path, with_mlm='softmax').to(device)  # 建立模型，加载权重

token_ids, segments_ids = tokenizer.encode("科学[MASK][MASK]是第一生产力")
print(''.join(tokenizer.ids_to_tokens(token_ids)))

tokens_ids_tensor = torch.tensor([token_ids]).to(device)
segment_ids_tensor = torch.tensor([segments_ids]).to(device)

# 需要传入参数with_mlm
model.eval()
with torch.no_grad():
    _, probas = model([tokens_ids_tensor, segment_ids_tensor])
    result = torch.argmax(probas[0, 3:5], dim=-1).cpu().numpy()
    print(tokenizer.decode(result))

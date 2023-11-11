# 基础测试：中文版chinese-roberta-wwm-ext-base的测试

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer


# 加载模型，
base_path = 'E:/pretrain_ckpt/roberta/hfl@chinese-roberta-wwm-ext-base/'
dict_path = base_path + '/vocab.txt'
config_path = base_path + '/config.json'
checkpoint_path = base_path + '/pytorch_model.bin'

# 分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 模型
model = build_transformer_model(config_path, checkpoint_path, with_mlm='softmax')

token_ids, segments_ids = tokenizer.encode("科学[MASK][MASK]是第一生产力")
print(''.join(tokenizer.ids_to_tokens(token_ids)))

tokens_ids_tensor = torch.tensor([token_ids])
segment_ids_tensor = torch.tensor([segments_ids])

# 需要传入参数with_mlm
model.eval()
with torch.no_grad():
    _, probas = model([tokens_ids_tensor, segment_ids_tensor])
    result = torch.argmax(probas[0, 3:5], dim=-1).numpy()
    print(tokenizer.decode(result))

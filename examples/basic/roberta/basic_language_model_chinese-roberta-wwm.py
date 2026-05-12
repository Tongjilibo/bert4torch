# 基础测试：中文版chinese-roberta-wwm-ext的测试

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import BertTokenizer


# 加载模型，
model_dir = '/data/pretrain_ckpt/hfl/chinese-roberta-wwm-ext'

# 分词器
tokenizer = BertTokenizer(model_dir + '/vocab.txt', do_lower_case=True)

# 模型
model = build_transformer_model(model_dir, with_mlm='softmax')

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

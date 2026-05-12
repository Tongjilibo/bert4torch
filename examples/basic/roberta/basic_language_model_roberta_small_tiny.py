# 基础测试：苏神 or UER  roberta-small/tiny mlm预测

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import BertTokenizer


# 加载模型
model_dir = '/data/pretrain_ckpt/Tongjilibo/chinese_roberta_L-4_H-312_A-12'
# model_dir = '/data/pretrain_ckpt/Tongjilibo/chinese_roberta_L-6_H-384_A-12'

# 分词器
tokenizer = BertTokenizer(model_dir + '/vocab.txt', do_lower_case=True)

# 模型
model = build_transformer_model(model_dir, with_mlm='softmax')


if __name__ == '__main__':
    text = '中国的首都是[MASK]京'

    token_ids, segment_ids = tokenizer.encode(text)
    token_ids = torch.tensor([token_ids])
    segment_ids = torch.tensor([segment_ids])

    _, probas = model.predict([token_ids, segment_ids])
    print(tokenizer.decode(torch.argmax(probas[0, 7:8], dim=-1).numpy()))

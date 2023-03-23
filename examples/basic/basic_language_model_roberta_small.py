# 基础测试：苏神 or UER  roberta-small/tiny mlm预测
# 使用的时候需要with_pool=False, 否则会有warnings, CLS的输出直接按last_hidden_state[:, 0]取得

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer


# 加载模型，
base_path = './pt_chinese_roberta_L-6_H-512'
dict_path = base_path + '/vocab.txt'
config_path = base_path + '/config.json'
checkpoint_path = base_path + '/pytorch_model.bin'

# 分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 模型
model = build_transformer_model(config_path,
                                checkpoint_path,
                                with_mlm='softmax',
                                segment_vocab_size=0)


if __name__ == '__main__':
    text = '中国的首都是[MASK]京'

    token_ids, _ = tokenizer.encode(text)
    token_ids = torch.tensor([token_ids])

    _, probas = model.predict([token_ids])
    print(tokenizer.decode(torch.argmax(probas[0, 7:8], dim=-1).numpy()))

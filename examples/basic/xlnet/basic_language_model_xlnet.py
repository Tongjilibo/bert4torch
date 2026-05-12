from transformers import XLNetTokenizer, XLNetModel
from bert4torch.models import build_transformer_model
import torch

pretrained_model = "/data/pretrain_ckpt/hfl/chinese-xlnet-base"

try:
    tokenizer = XLNetTokenizer.from_pretrained(pretrained_model)
    inputs = tokenizer(["你好啊，我叫张三", "天气不错啊"], padding=True, return_tensors="pt")
except:
    inputs = {
        'input_ids': torch.tensor([[   19,  1100,   453, 12864,    17,   378,  1821,   480,    86,     4, 3],
                                    [    5,     5,     5,     5,    19, 10022,    63,  4856, 12864,     4, 3]]), 
        'token_type_ids': torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                                        [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 2]]), 
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])
    }

# ----------------------bert4torch----------------------
model = build_transformer_model(
    pretrained_model,
    # with_lm=True
    pad_token_id=tokenizer.pad_token_id,
)
print('--------bert4torch last_hidden_state--------\n', model.predict([inputs['input_ids'], inputs['token_type_ids']]))


# ----------------------transformers----------------------
model = XLNetModel.from_pretrained(pretrained_model)
outputs = model(**inputs)
print('--------transformers last_hidden_state--------\n', outputs.last_hidden_state)

# tensor([[[ 1.6439,  2.5081, -0.8168,  ...,  2.9858, -1.7187,  2.2608],
#         [ 1.4438,  0.0143, -2.3434,  ..., -0.4675, -1.3934,  0.0586],
#         [-2.4990,  0.1179, -4.5759,  ..., -1.3046, -1.1044, -0.8428],
#         ...,
#         [-0.9851,  3.5510, -2.7199,  ..., -1.2289,  1.3538,  3.7172],
#         [-0.3565,  3.4840, -1.1415,  ...,  0.7239, -0.8993,  2.6563],
#         [-1.2611,  0.4786,  0.2938,  ..., -0.1406, -0.8540, -0.0551]],
#     [[ 4.6301,  1.6720,  2.4374,  ...,  0.7585, -2.0101, -2.6759],
#         [ 3.3043,  1.9915,  3.5986,  ...,  2.4108, -5.0444, -2.9610],
#         [ 2.4178,  4.7303,  1.0546,  ...,  6.0337, -7.9682, -3.1286],
#         ...,
#         [ 0.8774,  0.1549, -2.8244,  ..., -0.8508, -0.7616,  0.6366],
#         [-1.0617, -2.1594, -2.6308,  ...,  1.1179, -2.0446, -0.3432],
#         [-0.9595, -0.1471,  0.0589,  ...,  0.5352, -0.8691, -0.0425]]],
#     grad_fn=<CloneBackward0>)
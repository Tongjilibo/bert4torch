# 转换huggingface上IDEA-CCNL/Erlangshen-DeBERTa-v2权重
# 权重链接：
# https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese/tree/main
# https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese/tree/main
# https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese/tree/main
import torch

rep_str = {
    'query_proj': 'query',
    'key_proj': 'key',
    'value_proj': 'value'
}

# root_path = 'F:/Projects/pretrain_ckpt/deberta/[IDEA-CCNL-torch]--Erlangshen-DeBERTa-v2-97M-Chinese'
root_path = 'F:/Projects/pretrain_ckpt/deberta/[IDEA-CCNL-torch]--Erlangshen-DeBERTa-v2-320M-Chinese'
# root_path = 'F:/Projects/pretrain_ckpt/deberta/[IDEA-CCNL-torch]--Erlangshen-DeBERTa-v2-710M-Chinese'

ckpt_file = f'{root_path}/pytorch_model.bin'
state_dict = torch.load(ckpt_file)
state_dict_new = {}
for k, v in state_dict.items():
    for old_str, new_str in rep_str.items():
        if old_str in k:
            k = k.replace(old_str, new_str)
            state_dict_new[k] = v
    else:
        state_dict_new[k] = v
torch.save(state_dict_new, f'{root_path}/bert4torch_pytorch_model.bin')
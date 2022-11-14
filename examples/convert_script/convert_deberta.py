import torch

rep_str = {
    'query_proj': 'query',
    'key_proj': 'key',
    'value_proj': 'value'
}


ckpt_file = 'F:/Projects/pretrain_ckpt/deberta/[IDEA-CCNL-torch]--Erlangshen-DeBERTa-v2-320M-Chinese/pytorch_model.bin'
state_dict = torch.load(ckpt_file)
state_dict_new = {}
for k, v in state_dict.items():
    for old_str, new_str in rep_str.items():
        if old_str in k:
            k = k.replace(old_str, new_str)
            state_dict_new[k] = v
    else:
        state_dict_new[k] = v
torch.save(state_dict_new, 'F:/Projects/pretrain_ckpt/deberta/[IDEA-CCNL-torch]--Erlangshen-DeBERTa-v2-320M-Chinese/bert4torch_pytorch_model.bin')
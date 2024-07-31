'''roformer'''
import pytest
import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_bert4torch_model(model_dir):
    config_path = model_dir + "/bert4torch_config.json"
    if not os.path.exists(config_path):
        config_path = model_dir + "/config.json"
    checkpoint_path = model_dir + '/pytorch_model.bin'
    dict_path = model_dir + '/vocab.txt'
    
    model = build_transformer_model(config_path, checkpoint_path, with_mlm='softmax')  # 建立模型，加载权重
    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    model.eval()
    return model.to(device), tokenizer


@pytest.mark.parametrize("model_dir", ["/data/pretrain_ckpt/roformer/sushen@roformer_v1_base",
                                       "/data/pretrain_ckpt/roformer/sushen@roformer_v2_char_base"])
@torch.inference_mode()
def test_roformer(model_dir):
    query = "今天[MASK]很好，我[MASK]去公园玩。"
    model, tokenizer = get_bert4torch_model(model_dir)

    token_ids, segments_ids = tokenizer.encode(query)
    print(''.join(tokenizer.ids_to_tokens(token_ids)))

    tokens_ids_tensor = torch.tensor([token_ids], device=device)
    segment_ids_tensor = torch.tensor([segments_ids], device=device)

    # 需要传入参数with_mlm
    model.eval()
    with torch.no_grad():
        _, logits = model([tokens_ids_tensor, segment_ids_tensor])

    pred_str = []
    for i, logit in enumerate(logits[0]):
        if token_ids[i] == tokenizer._token_mask_id:
            pred_str.append(torch.argmax(logit, dim=-1).item())
        else:
            pred_str.append(token_ids[i])
    pred_str = tokenizer.decode(pred_str)
    print(pred_str)
    assert pred_str in {'今天天气很好，我想去公园玩。', '今天我很好，我想去公园玩。'}


if __name__=='__main__':
    test_roformer("/data/pretrain_ckpt/roformer/sushen@roformer_v1_base")
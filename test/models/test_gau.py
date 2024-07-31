'''gau'''
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
    dict_path = '/data/pretrain_ckpt/gau/sushen@chinese_GAU-alpha-char_L-24_H-768_torch/vocab.txt'
    
    model = build_transformer_model(config_path, checkpoint_path, with_mlm='softmax')  # 建立模型，加载权重
    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    model.eval()
    return model.to(device), tokenizer


@pytest.mark.parametrize("model_dir", ['/data/pretrain_ckpt/gau/sushen@chinese_GAU-alpha-char_L-24_H-768_torch'])
@torch.inference_mode()
def test_gau(model_dir):
    model, tokenizer = get_bert4torch_model(model_dir)

    token_ids, segments_ids = tokenizer.encode("近期正是上市公司财报密集披露的时间，但有多家龙头公司的业绩令投资者失望")
    token_ids[5] = token_ids[6] = tokenizer._token_mask_id
    print(''.join(tokenizer.ids_to_tokens(token_ids)))

    tokens_ids_tensor = torch.tensor([token_ids], device=device)
    segment_ids_tensor = torch.tensor([segments_ids], device=device)

    # 需要传入参数with_mlm
    model.eval()
    with torch.no_grad():
        _, probas = model([tokens_ids_tensor, segment_ids_tensor])
        result = torch.argmax(probas[0, 5:7], dim=-1).cpu().numpy()
        result = tokenizer.decode(result)
        print(result)
        assert result == '龙头'

if __name__=='__main__':
    test_gau('/data/pretrain_ckpt/gau/sushen@chinese_GAU-alpha-char_L-24_H-768_torch')
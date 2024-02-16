'''测试从huggingface上下载模型'''
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer
import torch
import pytest


@pytest.mark.parametrize("model_name", ["bert-base-chinese"])
def test_hf_download(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = build_transformer_model(checkpoint_path=model_name, with_mlm='softmax')

    inputtext = "今天[MASK]情很好"
    encoded_input = tokenizer(inputtext, return_tensors='pt')
    maskpos = encoded_input['input_ids'][0].tolist().index(103)

    # 需要传入参数with_mlm
    model.eval()
    with torch.no_grad():
        _, probas = model(**encoded_input)
        result = torch.argmax(probas[0, [maskpos]], dim=-1).cpu().numpy()
        pred_token = tokenizer.decode(result)
    print('pred_token: ', pred_token)
    assert pred_token == '心'


if __name__=='__main__':
    test_hf_download("bert-base-chinese")
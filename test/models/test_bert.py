'''测试bert和transformer的结果比对'''
import pytest
import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from transformers import BertConfig, BertTokenizer, BertModel
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_bert4torch_model(model_dir):
    config_path = model_dir + "/bert4torch_config.json"
    if not os.path.exists(config_path):
        config_path = model_dir + "/config.json"
    checkpoint_path = model_dir + '/pytorch_model.bin'

    model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重
    return model.to(device)


def get_hf_model(model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertModel.from_pretrained(model_dir)
    return model.to(device), tokenizer


@pytest.mark.parametrize("model_dir", ["E:/pretrain_ckpt/bert/google@bert-base-chinese",
                                       "E:/pretrain_ckpt/bert/bert-base-multilingual-cased",
                                       "E:/pretrain_ckpt/bert/hfl@macbert-base"])
@torch.inference_mode()
def test_bert_output(model_dir):
    model = get_bert4torch_model(model_dir)
    model_hf, tokenizer = get_hf_model(model_dir)

    model.eval()
    model_hf.eval()

    inputs = tokenizer('语言模型', padding=True, return_tensors='pt').to(device)
    sequence_output = model(**inputs)
    sequence_output_hf = model_hf(**inputs).last_hidden_state
    print(f"Output mean diff: {(sequence_output - sequence_output_hf).abs().mean().item()}")

    assert (sequence_output - sequence_output_hf).abs().max().item() < 1e-4


if __name__=='__main__':
    test_bert_output("E:/pretrain_ckpt/bert/hfl@macbert-base")
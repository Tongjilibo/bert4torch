import pytest
import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer
from transformers import BertConfig, BertTokenizer, BertModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_bert4torch_model(model_dir):
    vocab_path = model_dir + "/vocab.txt"
    config_path = model_dir + "/bert4torch_config.json"
    checkpoint_path = model_dir + '/pytorch_model.bin'

    tokenizer = Tokenizer(vocab_path, do_lower_case=True)  # 建立分词器
    model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重
    return model.to(device), tokenizer


def get_hf_model(model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertModel.from_pretrained(model_dir)
    return model.to(device), tokenizer


# @pytest.mark.parametrize("model_dir", ["E:/pretrain_ckpt/bert/google@bert-base-chinese"])
def test_bert_output(model_dir):
    model, _ = get_bert4torch_model(model_dir)
    model_hf, tokenizer = get_hf_model(model_dir)

    model.eval()
    model_hf.eval()

    inputs = tokenizer('语言模型', padding=True, return_tensors='pt').to(device)
    sequence_output = model(**inputs)
    sequence_output_hf = model_hf(**inputs).last_hidden_state
    print(f"Output mean diff: {(sequence_output - sequence_output_hf).abs().mean().item()}")

    assert (sequence_output - sequence_output_hf).abs().max().item() < 1e-4


if __name__=='__main__':
    test_bert_output("E:/pretrain_ckpt/bert/google@bert-base-chinese")
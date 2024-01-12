import pytest
import torch
from bert4torch.models import build_transformer_model



@pytest.mark.parametrize("model_dir", ["E:/pretrain_ckpt/bert/google@bert-base-chinese"])
def test_bert(model_dir):
    config_path = model_dir + "/bert4torch_config.json"
    checkpoint_path = model_dir + '/pytorch_model.bin'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_transformer_model(config_path, checkpoint_path).to(device)  # 建立模型，加载权重
    model.save_pretrained('./pytorch_model.bin')

    # 重新加载保存的预训练格式的权重
    build_transformer_model(config_path, checkpoint_path='./pytorch_model.bin')


if __name__=='__main__':
    test_bert("E:/pretrain_ckpt/bert/google@bert-base-chinese")
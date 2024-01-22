'''测试model.save_pretrained能否按照transformer格式进行保存和加载'''
import pytest
import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.nn.functional import softmax
import os
import shutil

def _ignore_copy_files(path, content):
        to_ignore = []
        for file_ in content:
            if ('.bin' in file_) or ('.safetensors' in file_):
                to_ignore.append(file_)
        return to_ignore


def get_configs(model_dir):
    if os.path.exists(model_dir + "/bert4torch_config.json"):
        config_path = model_dir + "/bert4torch_config.json"
    else:
        config_path = model_dir + "/config.json"
    
    checkpoint_path = model_dir + '/pytorch_model.bin'
    configs = {'with_pool': True, 'with_mlm': True}
    # 部分逻辑单独处理
    if 'albert' in model_dir:
         configs['model'] = 'albert'
    elif 'DeBERTa' in model_dir:
         configs.pop('with_pool')
         configs['model'] = 'deberta_v2'
    elif 'ernie' in model_dir:
         configs.pop('with_pool')
         configs['model'] = 'ERNIE'
    elif 'macbert' in model_dir:
         configs.pop('with_pool')
    return config_path, checkpoint_path, configs
     
@pytest.mark.parametrize("model_dir", ["E:/pretrain_ckpt/bert/google@bert-base-chinese",
                                       "E:/pretrain_ckpt/albert/brightmart@albert_base_zh",
                                       "E:/pretrain_ckpt/deberta/IDEA-CCNL@Erlangshen-DeBERTa-v2-320M-Chinese",
                                       "E:/pretrain_ckpt/ernie/baidu@ernie-1-base-zh",
                                       "E:/pretrain_ckpt/bert/hfl@macbert-base",
                                       'E:/pretrain_ckpt/roberta/hfl@chinese-roberta-wwm-ext-base/',
                                       "E:/pretrain_ckpt/bert/sushen@wobert_chinese_plus_base"])
@torch.inference_mode()
def test_encoder_model(model_dir):
    root_model_path = './pytorch_model'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_path, checkpoint_path, configs = get_configs(model_dir)
    inputtext = "今天[MASK]情很好"

    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(checkpoint_path))
    encoded_input = tokenizer(inputtext, return_tensors='pt').to(device)
    # =======================bert4torch加载和预测=======================
    model = build_transformer_model(config_path, checkpoint_path, return_dict=True, **configs).to(device)
    maskpos = encoded_input['input_ids'][0].tolist().index(tokenizer.mask_token_id)
    prediction_scores = model.predict(**encoded_input)['mlm_scores']
    predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
    predicted_token1 = tokenizer.convert_ids_to_tokens([predicted_index])[0]

    # =======================保存预训练的权重=======================
    checkpoint_path = os.path.join(root_model_path, 'pytorch_model.bin')
    model.save_pretrained(checkpoint_path)

    # =======================用保存预训练的权重bert4torch推理=======================
    model = build_transformer_model(config_path, checkpoint_path, return_dict=True, **configs).to(device)
    maskpos = encoded_input['input_ids'][0].tolist().index(tokenizer.mask_token_id)
    prediction_scores = model.predict(**encoded_input)['mlm_scores']
    predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
    predicted_token2 = tokenizer.convert_ids_to_tokens([predicted_index])[0]

    # =======================transformer加载和预测=======================
    model = AutoModelForMaskedLM.from_pretrained(root_model_path).to(device)
    prediction_scores = model(**encoded_input)['logits']

    predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
    predicted_token3 = tokenizer.convert_ids_to_tokens([predicted_index])[0]

    assert predicted_token1 == predicted_token2
    assert predicted_token2 == predicted_token3

    if os.path.exists(root_model_path):
        shutil.rmtree(root_model_path)


if __name__=='__main__':
    test_encoder_model("E:/pretrain_ckpt/bert/google@bert-base-chinese")
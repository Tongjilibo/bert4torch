'''测试model.save_pretrained能否按照transformer格式进行保存和加载'''
import pytest
import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.nn.functional import softmax
import shutil
import os


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
def test_encoder_model(model_dir):
    root_model_path = './pytorch_model'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_path, checkpoint_path, configs = get_configs(model_dir)

    # =======================保存预训练的权重=======================
    model = build_transformer_model(config_path, checkpoint_path, **configs).to(device)
    shutil.copytree(model_dir, root_model_path, ignore=_ignore_copy_files)
    model.save_pretrained(os.path.join(root_model_path, 'pytorch_model.bin'))

    # =======================transformer加载和预测=======================
    tokenizer = AutoTokenizer.from_pretrained(root_model_path)
    model = AutoModelForMaskedLM.from_pretrained(root_model_path)

    inputtext = "今天[MASK]情很好"

    encoded_input = tokenizer(inputtext, return_tensors='pt')
    maskpos = encoded_input['input_ids'][0].tolist().index(tokenizer.mask_token_id)
    prediction_scores = model(**encoded_input)['logits']

    logit_prob = softmax(prediction_scores[0, maskpos],dim=-1).data.tolist()
    predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(f'{model_dir}'.center(100, '='))
    print(predicted_token, logit_prob[predicted_index])
    
    if os.path.exists(root_model_path):
         shutil.rmtree(root_model_path)


if __name__=='__main__':
    test_encoder_model("E:/pretrain_ckpt/bert/google@bert-base-chinese")
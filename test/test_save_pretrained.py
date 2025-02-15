'''测试model.save_pretrained能否按照transformer格式进行保存和加载'''
import pytest
import torch
from bert4torch.models import build_transformer_model
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from torch.nn.functional import softmax
import os
import shutil


device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
     
@pytest.mark.parametrize("model_dir", ["E:/data/pretrain_ckpt/google-bert/bert-base-chinese",
                                       "E:/data/pretrain_ckpt/voidful/albert_chinese_base",
                                       "E:/data/pretrain_ckpt/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese",
                                       "E:/data/pretrain_ckpt/nghuyong/ernie-1.0-base-zh",
                                       "E:/data/pretrain_ckpt/hfl/chinese-macbert-base",
                                       'E:/data/pretrain_ckpt/hfl/chinese-roberta-wwm-ext/',
                                       "E:/data/pretrain_ckpt/junnyu/wobert_chinese_plus_base"])
@torch.inference_mode()
def test_encoder_model(model_dir):
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
    save_dir = './pytorch_model'
    model.save_pretrained(save_dir)

    # =======================用保存预训练的权重bert4torch推理=======================
    model = build_transformer_model(config_path, save_dir, return_dict=True, **configs).to(device)
    maskpos = encoded_input['input_ids'][0].tolist().index(tokenizer.mask_token_id)
    prediction_scores = model.predict(**encoded_input)['mlm_scores']
    predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
    predicted_token2 = tokenizer.convert_ids_to_tokens([predicted_index])[0]

    # =======================transformer加载和预测=======================
    model = AutoModelForMaskedLM.from_pretrained(save_dir).to(device)
    prediction_scores = model(**encoded_input)['logits']

    predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
    predicted_token3 = tokenizer.convert_ids_to_tokens([predicted_index])[0]

    assert predicted_token1 == predicted_token2
    assert predicted_token2 == predicted_token3

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)


@pytest.mark.parametrize("model_dir", ['E:/data/pretrain_ckpt/Qwen/Qwen-1_8B-Chat'])
@torch.inference_mode()
def test_qwen(model_dir):
    im_start, im_end = "<|im_start|>", "<|im_end|>"
    tempelate = "\n{}user\n你好{}\n{}assistant\n"
    query = tempelate.format(im_start, im_end, im_start)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer_encode_config = {'allowed_special': {"<|im_start|>", "<|im_end|>", '<|endoftext|>'}}
    tokenizer_decode_config = {'skip_special_tokens': True}

    config_path = model_dir + "/bert4torch_config.json"
    if not os.path.exists(config_path):
        config_path = model_dir + "/config.json"

    model = build_transformer_model(config_path, model_dir).to(device)
    model.eval()

    if 'Chat' in model_dir:
        eos_token_id = [tokenizer.im_start_id, tokenizer.im_end_id]
    else:
        eos_token_id = tokenizer.encode("<|endoftext|>", **tokenizer_encode_config)

    generation_config = {
        'tokenizer': tokenizer,
        'tokenizer_config': {**tokenizer_encode_config, **tokenizer_decode_config}, 
        'eos_token_id': eos_token_id, 
        'mode': 'random_sample', 
        'max_length': 20, 
        'default_rtype': 'logits',
        'use_states': True,
        'top_k': 1, 
    }
    sequence_output = model.generate(query, **generation_config)
    print(sequence_output)

    save_dir = './qwen'
    model.save_pretrained(save_dir)

    model_hf = AutoModelForCausalLM.from_pretrained(save_dir, trust_remote_code=True).half().to(device)
    model_hf.eval()
    inputs = tokenizer.encode(query, return_tensors="pt", **tokenizer_encode_config).to(device)
    sequence_output_hf = model_hf.generate(inputs, top_k=1, max_length=20)
    sequence_output_hf = tokenizer.decode(sequence_output_hf[0].cpu(), skip_special_tokens=True)
    sequence_output_hf = sequence_output_hf[len(tempelate.format('','','')):].strip()
    print(sequence_output_hf)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)


if __name__=='__main__':
    # test_encoder_model("E:/data/pretrain_ckpt/google-bert/bert-base-chinese")
    test_qwen('E:/data/pretrain_ckpt/Qwen/Qwen-1_8B-Chat')
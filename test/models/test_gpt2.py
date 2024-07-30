'''测试bert和transformer的结果比对'''
import pytest
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import cuda_empty_cache
from bert4torch.tokenizers import Tokenizer
from transformers import TextGenerationPipeline, AutoTokenizer, AutoModelWithLMHead
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_bert4torch_model(model_dir):
    config_path = model_dir + "/bert4torch_config.json"
    if not os.path.exists(config_path):
        config_path = model_dir + "/config.json"
    checkpoint_path = model_dir

    model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重
    model.eval()
    return model.to(device)


def get_hf_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelWithLMHead.from_pretrained(model_dir)

    model.eval()
    return model.to(device), tokenizer


@pytest.mark.parametrize("model_dir", ["E:/pretrain_ckpt/gpt/cpm@cpm_lm_2.6b/",
                                       'E:/pretrain_ckpt/gpt/uer@gpt2-chinese-cluecorpussmall/'])
@torch.inference_mode()
def test_gpt2(model_dir):
    query = '你好'

    model_hf, tokenizer = get_hf_model(model_dir)
    text_generator = TextGenerationPipeline(model_hf, tokenizer, device=device)
    sequence_output_hf = text_generator(query, max_length=50, do_sample=True, top_k=1)
    sequence_output_hf = sequence_output_hf[0]['generated_text'].replace(' ', '')
    del model_hf
    cuda_empty_cache()


    model = get_bert4torch_model(model_dir)
    generation_config = {
        'tokenizer': tokenizer,
        'tokenizer_config': {'skip_special_tokens': True, 'clean_up_tokenization_spaces': True, 'add_special_tokens': False},
        'bos_token_id': None, 
        'eos_token_id': tokenizer.eos_token_id or 50256, 
        'mode': 'random_sample',
        'max_length': 50, 
        'default_rtype': 'logits', 
        'use_states': True,
        'top_k': 1,
        'include_input': True
    }   
    sequence_output = model.generate(query, **generation_config).replace(' ', '')

    print(sequence_output, '    ====>    ', sequence_output_hf)
    assert sequence_output==sequence_output_hf


@pytest.mark.parametrize("model_dir", ['E:/pretrain_ckpt/gpt/imcaspar@gpt2-ml_15g_corpus_torch'])
@torch.inference_mode()
def test_gpt2_ml(model_dir):
    query = '你好'
    config_path = os.path.join(model_dir, 'bert4torch_config.json')
    checkpoint_path = os.path.join(model_dir, 'pytorch_model.bin')
    dict_path = os.path.join(model_dir, 'vocab.txt')

    tokenizer = Tokenizer(dict_path, token_start=None, token_end=None, do_lower_case=True)  # 建立分词器
    model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重
    model.eval()

    generation_config = {
        'tokenizer': tokenizer,
        'bos_token_id': None, 
        'eos_token_id': 511, 
        'mode': 'random_sample',
        'max_length': 50, 
        'default_rtype': 'logits', 
        'use_states': True,
        'top_k': 1,
        'include_input': True
    }   
    sequence_output = model.generate(query, **generation_config).replace(' ', '')
    print(sequence_output)
    assert sequence_output == '你好，我是一名大三的学生，我的专业是电气工程及其自动化，我的大学室友是一名大三的学生，他们都是大一的'


if __name__=='__main__':
    test_gpt2_ml('E:/pretrain_ckpt/gpt/imcaspar@gpt2-ml_15g_corpus_torch')
'''测试bert和transformer的结果比对'''
import pytest
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import cuda_empty_cache
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
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


@pytest.mark.parametrize("model_dir", ['E:/pretrain_ckpt/llama/Baichuan-7B',
                                       'E:/pretrain_ckpt/llama/Baichuan-13B',
                                       'E:/pretrain_ckpt/llama/Baichuan2-7B-Chat',
                                       'E:/pretrain_ckpt/llama/Baichuan-13B-Chat'
                                       ])
@torch.inference_mode()
def test_baichuan(model_dir):
    query = '你好'

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model_hf = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).half().to(device)
    model_hf.eval()
    inputs = tokenizer.encode(query, return_tensors="pt").to(device)
    sequence_output_hf = model_hf.generate(inputs, top_k=1, max_length=20)
    sequence_output_hf = tokenizer.decode(sequence_output_hf[0].cpu(), skip_special_tokens=True)
    sequence_output_hf = sequence_output_hf.replace(' ', '')
    del model_hf
    cuda_empty_cache()

    model = get_bert4torch_model(model_dir)
    generation_config = {
        'tokenizer': tokenizer,
        'end_id': 2, 
        'mode':'random_sample', 
        'tokenizer_config': {'skip_special_tokens': True},
        'max_length': 20, 
        'default_rtype': 'logits', 
        'use_states': True,
        'topk': 1, 
        'include_input': True
    }
    sequence_output = model.generate(query, **generation_config).replace(' ', '')

    print(sequence_output, '    ====>    ', sequence_output_hf)
    assert sequence_output==sequence_output_hf


@pytest.mark.parametrize("model_dir", ['E:/pretrain_ckpt/llama/belle-llama-7b-2m'])
@torch.inference_mode()
def test_belle(model_dir):
    query = f"Human: 你好 \n\nAssistant: "

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = get_bert4torch_model(model_dir)
    generation_config = {
        'tokenizer': tokenizer,
        'end_id': 2, 
        'mode': 'random_sample',
        'max_length': 50, 
        'default_rtype': 'logits', 
        'use_states': True,
        'topk': 1, 
    }
    sequence_output = model.generate(query, **generation_config)

    print(sequence_output)
    assert sequence_output=='我是Belle，一个人工智能语言模型。你需要我做什么？'


@pytest.mark.parametrize("model_dir", ['E:/pretrain_ckpt/llama/hfl@chinese_alpaca_plus_7b',
                                       'E:/pretrain_ckpt/llama/hfl@chinese_llama_plus_7b'])
@torch.inference_mode()
def test_chinese_llama_alpaca(model_dir):
    query = "你好"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = get_bert4torch_model(model_dir)
    generation_config = {
        'tokenizer': tokenizer,
        'end_id': 2, 
        'mode': 'random_sample', 
        'tokenizer_config': {'skip_special_tokens': True},
        'max_length': 50, 
        'default_rtype': 'logits', 
        'use_states': True,
        'include_input': True,
        'topk': 1
    }
    sequence_output = model.generate(query, **generation_config)

    print(sequence_output)
    assert sequence_output in {'你好，我是来自中国的。我最近在学习英语，希望通过学习英语来提高自己的能力。我正在学习英语口语，希望通过练习来提高自己的口语能力。',
                               '你好，我是一名高三学生，我今年高考成绩不是很理想，我想请问一下，我可以复读吗？你好，我是一名高三学生，我今年高考成绩不是很理想，我想请问一下，我可以复读吗？你好'}


@pytest.mark.parametrize("model_dir", ['E:/pretrain_ckpt/llama/lmsys@vicuna-7b-v1.5'])
@torch.inference_mode()
def test_vicuna(model_dir):
    query = '你好'

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_hf = AutoModelForCausalLM.from_pretrained(model_dir).half().to(device)
    inputs = tokenizer.encode(query, return_tensors="pt").to(device)
    sequence_output_hf = model_hf.generate(inputs, top_k=1, max_length=20)
    sequence_output_hf = tokenizer.decode(sequence_output_hf[0].cpu(), skip_special_tokens=True)
    sequence_output_hf = sequence_output_hf.replace(' ', '')
    del model_hf
    cuda_empty_cache()

    model = get_bert4torch_model(model_dir)
    generation_config = {
        'tokenizer': tokenizer,
        'end_id': 2, 
        'mode':'random_sample', 
        'tokenizer_config': {'skip_special_tokens': True},
        'max_length': 20, 
        'default_rtype': 'logits', 
        'use_states': True,
        'topk': 1, 
        'include_input': True
    }
    sequence_output = model.generate(query, **generation_config).replace(' ', '')

    print(sequence_output, '    ====>    ', sequence_output_hf)
    assert sequence_output==sequence_output_hf


@pytest.mark.parametrize("model_dir", ['E:/pretrain_ckpt/llama/IDEA-CCNL@Ziya-LLaMA-13B-v1.1',
                                       'E:/pretrain_ckpt/llama/IDEA-CCNL@Ziya-LLaMA-13B-v1',
                                       'E:/pretrain_ckpt/llama/IDEA-CCNL@Ziya-LLaMA-13B_pretrain'])
@torch.inference_mode()
def test_ziya(model_dir):
    query = '你好'

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_hf = AutoModelForCausalLM.from_pretrained(model_dir).half().to(device)
    inputs = tokenizer.encode(query, return_tensors="pt").to(device)
    sequence_output_hf = model_hf.generate(inputs, top_k=1, max_length=20)
    sequence_output_hf = tokenizer.decode(sequence_output_hf[0].cpu(), skip_special_tokens=True)
    sequence_output_hf = sequence_output_hf.replace(' ', '')
    del model_hf
    cuda_empty_cache()

    model = get_bert4torch_model(model_dir)
    generation_config = {
        'tokenizer': tokenizer,
        'end_id': 2, 
        'mode':'random_sample', 
        'tokenizer_config': {'skip_special_tokens': True},
        'max_length': 20, 
        'default_rtype': 'logits', 
        'use_states': True,
        'topk': 1, 
        'include_input': True
    }
    sequence_output = model.generate(query, **generation_config).replace(' ', '')

    print(sequence_output, '    ====>    ', sequence_output_hf)
    assert sequence_output==sequence_output_hf


@pytest.mark.parametrize("model_dir", ['E:/pretrain_ckpt/llama/llama-2-7b',
                                       'E:/pretrain_ckpt/llama/llama-2-7b-chat',
                                       'E:/pretrain_ckpt/llama/llama-2-13b',
                                       'E:/pretrain_ckpt/llama/llama-2-13b-chat'])
@torch.inference_mode()
def test_llama2(model_dir):
    query = '你好'

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_hf = AutoModelForCausalLM.from_pretrained(model_dir).half().to(device)
    inputs = tokenizer.encode(query, return_tensors="pt").to(device)
    sequence_output_hf = model_hf.generate(inputs, top_k=1, max_length=20)
    sequence_output_hf = tokenizer.decode(sequence_output_hf[0].cpu(), skip_special_tokens=True)
    sequence_output_hf = sequence_output_hf.replace(' ', '')
    del model_hf
    cuda_empty_cache()

    model = get_bert4torch_model(model_dir)
    generation_config = {
        'tokenizer': tokenizer,
        'end_id': 2, 
        'mode':'random_sample', 
        'tokenizer_config': {'skip_special_tokens': True},
        'max_length': 20, 
        'default_rtype': 'logits', 
        'use_states': True,
        'topk': 1, 
        'include_input': True
    }
    sequence_output = model.generate(query, **generation_config).replace(' ', '')

    print(sequence_output, '    ====>    ', sequence_output_hf)
    assert sequence_output==sequence_output_hf


@pytest.mark.parametrize("model_dir", ['E:/pretrain_ckpt/llama/llama-7b',
                                       'E:/pretrain_ckpt/llama/llama-13b'])
@torch.inference_mode()
def test_llama(model_dir):
    query = '你好'

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_hf = AutoModelForCausalLM.from_pretrained(model_dir).half().to(device)
    inputs = tokenizer.encode(query, return_tensors="pt").to(device)
    sequence_output_hf = model_hf.generate(inputs, top_k=1, max_length=20)
    sequence_output_hf = tokenizer.decode(sequence_output_hf[0].cpu(), skip_special_tokens=True)
    sequence_output_hf = sequence_output_hf.replace(' ', '')
    del model_hf
    cuda_empty_cache()

    model = get_bert4torch_model(model_dir)
    generation_config = {
        'tokenizer': tokenizer,
        'end_id': 2, 
        'mode':'random_sample', 
        'tokenizer_config': {'skip_special_tokens': True},
        'max_length': 20, 
        'default_rtype': 'logits', 
        'use_states': True,
        'topk': 1, 
        'include_input': True
    }
    sequence_output = model.generate(query, **generation_config).replace(' ', '')

    print(sequence_output, '    ====>    ', sequence_output_hf)
    assert sequence_output==sequence_output_hf

@pytest.mark.parametrize("model_dir", ["E:/pretrain_ckpt/llama/01-ai@Yi-6B"])
@torch.inference_mode()
def test_yi(model_dir):
    query = '你好'

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model_hf = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).half().to(device)
    inputs = tokenizer.encode(query, return_tensors="pt").to(device)
    sequence_output_hf = model_hf.generate(inputs, top_k=1, max_length=20)
    sequence_output_hf = tokenizer.decode(sequence_output_hf[0].cpu(), skip_special_tokens=True)
    sequence_output_hf = sequence_output_hf.replace(' ', '')
    del model_hf
    cuda_empty_cache()

    model = get_bert4torch_model(model_dir)
    generation_config = {
        'tokenizer': tokenizer,
        'end_id': tokenizer.eos_token_id, 
        'mode':'random_sample', 
        'tokenizer_config': {'skip_special_tokens': True},
        'max_length': 20, 
        'default_rtype': 'logits', 
        'use_states': True,
        'topk': 1, 
        'include_input': True
    }
    sequence_output = model.generate(query, **generation_config).replace(' ', '')

    print(sequence_output, '    ====>    ', sequence_output_hf)
    assert sequence_output==sequence_output_hf


if __name__=='__main__':
    test_baichuan('E:/pretrain_ckpt/llama/Baichuan-7B')
    test_belle('E:/pretrain_ckpt/llama/belle-llama-7b-2m')
    test_chinese_llama_alpaca('E:/pretrain_ckpt/llama/hfl@chinese_alpaca_plus_7b')
    test_vicuna('E:/pretrain_ckpt/llama/lmsys@vicuna-7b-v1.5')
    test_ziya('E:/pretrain_ckpt/llama/IDEA-CCNL@Ziya-LLaMA-13B-v1.1')
    test_llama2('E:/pretrain_ckpt/llama/llama-2-7b-chat')
    test_llama('E:/pretrain_ckpt/llama/llama-7b')
    test_yi("E:/pretrain_ckpt/llama/01-ai@Yi-6B")
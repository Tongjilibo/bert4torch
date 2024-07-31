'''测试bert和transformer的结果比对'''
import pytest
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import cuda_empty_cache
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).half().to(device)
    model.eval()
    return model.to(device), tokenizer


@pytest.mark.parametrize("model_dir", ['E:/data/pretrain_ckpt/Qwen/Qwen-7B-Chat',
                                       'E:/data/pretrain_ckpt/Qwen/Qwen-7B',
                                       'E:/data/pretrain_ckpt/Qwen/Qwen-1_8B-Chat'])
@torch.inference_mode()
def test_qwen(model_dir):
    im_start, im_end = "<|im_start|>", "<|im_end|>"
    tempelate = "\n{}user\n你好{}\n{}assistant\n"
    query = tempelate.format(im_start, im_end, im_start)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer_encode_config = {'allowed_special': {"<|im_start|>", "<|im_end|>", '<|endoftext|>'}}
    tokenizer_decode_config = {'skip_special_tokens': True}
    inputs = tokenizer.encode(query, return_tensors="pt", **tokenizer_encode_config).to(device)

    model_hf = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).half().to(device)
    model_hf.eval()
    sequence_output_hf = model_hf.generate(inputs, top_k=1, max_length=20)
    sequence_output_hf = tokenizer.decode(sequence_output_hf[0].cpu(), skip_special_tokens=True)
    sequence_output_hf = sequence_output_hf[len(tempelate.format('','','')):].strip()
    del model_hf
    cuda_empty_cache()

    model = get_bert4torch_model(model_dir)

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

    minlen = min(len(sequence_output), len(sequence_output_hf))
    sequence_output = sequence_output[:minlen]
    sequence_output_hf = sequence_output_hf[:minlen]
    print(sequence_output, '    ====>    ', sequence_output_hf)
    assert sequence_output == sequence_output_hf


if __name__=='__main__':
    test_qwen('E:/data/pretrain_ckpt/Qwen/Qwen-1_8B-Chat')
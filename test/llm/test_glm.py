'''测试bert和transformer的结果比对'''
import pytest
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import cuda_empty_cache
from transformers import AutoTokenizer, AutoModel
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
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().to(device)
    model.eval()
    return model.to(device), tokenizer


@pytest.mark.parametrize("model_dir", ["E:/pretrain_ckpt/glm/chatglm-6b",
                                    #    "E:/pretrain_ckpt/glm/chatglm-6b-int4",
                                    #    "E:/pretrain_ckpt/glm/chatglm-6b-int8",
                                       "E:/pretrain_ckpt/glm/chatglm2-6b",
                                    #    "E:/pretrain_ckpt/glm/chatglm2-6b-int4",
                                       "E:/pretrain_ckpt/glm/chatglm2-6b-32k",
                                       "E:/pretrain_ckpt/glm/chatglm3-6b",
                                       "E:/pretrain_ckpt/glm/chatglm3-6b-32k"])
@torch.inference_mode()
def test_glm(model_dir):
    query = '你好'

    model_hf, tokenizer = get_hf_model(model_dir)
    inputs = tokenizer.encode(query, return_tensors="pt").to(device)
    sequence_output_hf = model_hf.generate(inputs, top_k=1, max_length=20)
    sequence_output_hf = tokenizer.decode(sequence_output_hf[0].cpu(), skip_special_tokens=True)
    sequence_output_hf = sequence_output_hf.replace('[gMASK]sop', '').replace(' ', '')
    del model_hf
    cuda_empty_cache()


    model = get_bert4torch_model(model_dir)
    generation_config = {
        'tokenizer': tokenizer,
        'tokenizer_config': {'skip_special_tokens': True},
        'start_id': None, 
        'end_id': tokenizer.eos_token_id, 
        'mode': 'random_sample',
        'max_length': 20, 
        'default_rtype': 'logits', 
        'use_states': True,
        'top_k': 1,
        'include_input': True
    }    
    sequence_output = model.generate(query, **generation_config).replace(' ', '')

    print(sequence_output, '    ====>    ', sequence_output_hf)
    assert sequence_output==sequence_output_hf


if __name__=='__main__':
    test_glm("E:/pretrain_ckpt/glm/chatglm3-6b-32k")
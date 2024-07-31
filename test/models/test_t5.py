'''t5'''
import pytest
import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import Tokenizer, SpTokenizer, load_vocab
import os
import jieba
import re


device = 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.mark.parametrize("model_dir", ['E:/data/pretrain_ckpt/t5/ClueAI@ClueAI-ChatYuan-large-v1/'])
@torch.inference_mode()
def test_chatyuan(model_dir):
    config_path = model_dir + 'bert4torch_config.json'
    checkpoint_path = model_dir + 'pytorch_model.bin'
    spm_path = model_dir + 'spiece.model'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载并精简词表，建立分词器
    tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>', keep_accents=True)
    model = build_transformer_model(config_path, checkpoint_path, pad_token_id=-1).to(device)

    generation_config = {
        'tokenizer': tokenizer,
        'bos_token_id': 0, 
        'eos_token_id': tokenizer._token_end_id, 
        'max_length': 512,
        'top_k': 1
    }
    res = model.generate("用户：你能干什么\\n小元：", **generation_config)
    print(res)
    assert res == '您好!我是元语AI。我可以回答您的问题、写文章、写作业、翻译，对于一些法律等领域的问题我也可以给你提供信息。'


@pytest.mark.parametrize("model_dir", ['E:/data/pretrain_ckpt/t5/ClueAi@PromptCLUE-base-v1-5/'])
@torch.inference_mode()
def test_PromptCLUE(model_dir):
    config_path = model_dir + 'bert4torch_config.json'
    checkpoint_path = model_dir + 'pytorch_model.bin'
    spm_path = model_dir + 'spiece.model'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载并精简词表，建立分词器
    tokenizer = SpTokenizer(spm_path, token_start=None, token_end='</s>', keep_accents=True)
    model = build_transformer_model(config_path, checkpoint_path, pad_token_id=-1).to(device)

    generation_config = {
        'tokenizer': tokenizer,
        'bos_token_id': 0, 
        'eos_token_id': tokenizer._token_end_id, 
        'max_length': 512,
        'top_k': 1
    }
    res = model.generate("生成与下列文字相同意思的句子： 白云遍地无人扫 答案：", **generation_config)
    print(res)
    assert res == '白云遍地无人扫。'


@pytest.mark.parametrize("model_dir", ['E:/data/pretrain_ckpt/t5/sushen@chinese_t5_pegasus_small_torch/',
                                       'E:/data/pretrain_ckpt/t5/sushen@chinese_t5_pegasus_base_torch/'])
@torch.inference_mode()
def test_t5_pegasus(model_dir):
    config_path = model_dir + 'bert4torch_config.json'
    checkpoint_path = model_dir + 'pytorch_model.bin'
    dict_path = model_dir + 'vocab.txt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载并精简词表，建立分词器
    tokenizer = Tokenizer(
        dict_path,
        do_lower_case=True,
        pre_tokenize=lambda s: jieba.cut(s, HMM=False)
    )
    model = build_transformer_model(config_path, checkpoint_path).to(device)

    generation_config = {
        'tokenizer': tokenizer,
        'bos_token_id': tokenizer._token_start_id, 
        'eos_token_id': tokenizer._token_end_id, 
        'max_length': 512,
        'top_k': 1
    }
    res = model.generate("今天天气不错啊", **generation_config)
    print(res)
    assert res in {'我是个女的，我想知道我是怎么想的', '请问明天的天气怎么样啊？'}


@pytest.mark.parametrize("model_dir", ['E:/data/pretrain_ckpt/t5/uer@t5-small-chinese-cluecorpussmall/',
                                       'E:/data/pretrain_ckpt/t5/uer@t5-base-chinese-cluecorpussmall/'])
@torch.inference_mode()
def test_t5_ner(model_dir):
    config_path = model_dir + 'bert4torch_config.json'
    checkpoint_path = model_dir + 'pytorch_model.bin'
    dict_path = model_dir + 'vocab.txt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载并精简词表，建立分词器
    token_dict = load_vocab(
        dict_path=dict_path,
        simplified=False,
        startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )
    tokenizer = Tokenizer(token_dict, do_lower_case=True)
    model = build_transformer_model(config_path, checkpoint_path).to(device)

    generation_config = {
        'tokenizer': tokenizer,
        'bos_token_id': tokenizer._token_start_id, 
        'eos_token_id': 1, 
        'max_length': 32,
        'top_k': 1
    }
    res = model.generate("中国的首都是extra0京", **generation_config)
    res = re.sub('extra[0-9]+', '', res).strip()
    print(res)
    assert res == '北'


if __name__=='__main__':
    test_chatyuan('E:/data/pretrain_ckpt/t5/ClueAI@ClueAI-ChatYuan-large-v1/')
    test_PromptCLUE('E:/data/pretrain_ckpt/t5/ClueAi@PromptCLUE-base-v1-5/')
    test_t5_pegasus('E:/data/pretrain_ckpt/t5/sushen@chinese_t5_pegasus_small_torch/')
    test_t5_ner('E:/data/pretrain_ckpt/t5/uer@t5-base-chinese-cluecorpussmall/')
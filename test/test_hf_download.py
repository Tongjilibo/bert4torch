'''测试从huggingface上下载模型'''
# import os
# os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from bert4torch.models import build_transformer_model
from bert4torch.snippets import get_config_path
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import pytest


@pytest.mark.parametrize("model_name", ['google-bert/bert-base-chinese',
                                        "hfl/chinese-bert-wwm-ext"])
def test_hf_download(model_name):
    model = build_transformer_model(checkpoint_path=model_name, with_mlm='softmax')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputtext = "今天[MASK]情很好"
    encoded_input = tokenizer(inputtext, return_tensors='pt')
    maskpos = encoded_input['input_ids'][0].tolist().index(103)

    # 需要传入参数with_mlm
    model.eval()
    with torch.no_grad():
        _, probas = model(**encoded_input)
        result = torch.argmax(probas[0, [maskpos]], dim=-1).cpu().numpy()
        pred_token = tokenizer.decode(result)
    print('pred_token: ', pred_token)
    assert pred_token == '心'


@pytest.mark.parametrize("model_name", ['google-bert/bert-base-chinese',
                                        'Tongjilibo/bert-chinese_L-12_H-768_A-12',
                                        'hfl/chinese-bert-wwm-ext',
                                        'google-bert/bert-base-multilingual-cased',
                                        'hfl/chinese-macbert-base',
                                        'hfl/chinese-macbert-large',
                                        'junnyu/wobert_chinese_base',
                                        'junnyu/wobert_chinese_plus_base',
                                        'hfl/chinese-roberta-wwm-ext',
                                        'hfl/chinese-roberta-wwm-ext-large',
                                        'Tongjilibo/chinese_roberta_L-4_H-312_A-12',
                                        'Tongjilibo/chinese_roberta_L-6_H-384_A-12',
                                        'FacebookAI/roberta-base',
                                        'ethanyt/guwenbert-base',
                                        'voidful/albert_chinese_tiny',
                                        'voidful/albert_chinese_small',
                                        'voidful/albert_chinese_base',
                                        'voidful/albert_chinese_large',
                                        'voidful/albert_chinese_xlarge',
                                        'voidful/albert_chinese_xxlarge',
                                        'sijunhe/nezha-cn-base',
                                        'sijunhe/nezha-cn-large',
                                        'sijunhe/nezha-base-wwm',
                                        'sijunhe/nezha-large-wwm',
                                        'Tongjilibo/nezha_gpt_dialog',
                                        'hfl/chinese-xlnet-base',
                                        'transfo-xl/transfo-xl-wt103',
                                        'IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese',
                                        'IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese',
                                        'IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese',
                                        'hfl/chinese-electra-base-discriminator',
                                        'nghuyong/ernie-1.0-base-zh',
                                        'nghuyong/ernie-3.0-base-zh',
                                        'junnyu/roformer_chinese_base',
                                        'junnyu/roformer_v2_chinese_char_base',
                                        'Tongjilibo/simbert-chinese-base',
                                        'Tongjilibo/simbert-chinese-small',
                                        'Tongjilibo/simbert-chinese-tiny',
                                        'junnyu/roformer_chinese_sim_char_base',
                                        'junnyu/roformer_chinese_sim_char_ft_base',
                                        'junnyu/roformer_chinese_sim_char_small',
                                        'junnyu/roformer_chinese_sim_char_ft_small',
                                        'Tongjilibo/chinese_GAU-alpha-char_L-24_H-768',
                                        'Tongjilibo/uie-base',
                                        'thu-coai/CDial-GPT_LCCC-base',
                                        'thu-coai/CDial-GPT_LCCC-large',
                                        'TsinghuaAI/CPM-Generate',
                                        'Tongjilibo/chinese_nezha_gpt_L-12_H-768_A-12',
                                        'uer/gpt2-chinese-cluecorpussmall',
                                        'Tongjilibo/gpt2-ml_15g_corpus',
                                        'Tongjilibo/gpt2-ml_30g_corpus',
                                        'fnlp/bart-base-chinese',
                                        'fnlp/bart-base-chinese-v1.0',
                                        'uer/t5-small-chinese-cluecorpussmall',
                                        'uer/t5-base-chinese-cluecorpussmall',
                                        'google/mt5-base',
                                        'Tongjilibo/chinese_t5_pegasus_small',
                                        'Tongjilibo/chinese_t5_pegasus_base',
                                        'ClueAI/ChatYuan-large-v1',
                                        'ClueAI/ChatYuan-large-v2',
                                        'ClueAI/PromptCLUE-base',
                                        'THUDM/chatglm-6b',
                                        'THUDM/chatglm-6b-int8',
                                        'THUDM/chatglm-6b-int4',
                                        # 'THUDM/chatglm-6b-v0.1.0',
                                        'THUDM/chatglm2-6b',
                                        'THUDM/chatglm2-6b-int4',
                                        'THUDM/chatglm2-6b-32k',
                                        'THUDM/chatglm3-6b',
                                        'THUDM/chatglm3-6b-32k',
                                        # 'meta-llama/llama-7b',
                                        # 'meta-llama/llama-13b',
                                        # 'meta-llama/Llama-2-7b-hf',
                                        # 'meta-llama/Llama-2-7b-chat-hf',
                                        # 'meta-llama/Llama-2-13b-hf',
                                        # 'meta-llama/Llama-2-13b-chat-hf',
                                        'hfl/chinese-alpaca-plus-7b',
                                        'hfl/chinese-llama-plus-7b',
                                        'IDEA-CCNL/Ziya-LLaMA-13B-v1',
                                        'IDEA-CCNL/Ziya-LLaMA-13B-v1.1',
                                        'baichuan-inc/Baichuan-7B',
                                        'baichuan-inc/Baichuan-13B-Base',
                                        'baichuan-inc/Baichuan-13B-Chat',
                                        'baichuan-inc/Baichuan2-7B-Base',
                                        'baichuan-inc/Baichuan2-7B-Chat',
                                        'baichuan-inc/Baichuan2-13B-Base',
                                        'baichuan-inc/Baichuan2-13B-Chat',
                                        'lmsys/vicuna-7b-v1.5',
                                        '01-ai/Yi-6B',
                                        '01-ai/Yi-6B-200K',
                                        'bigscience/bloom-560m',
                                        'bigscience/bloomz-560m',
                                        'Qwen/Qwen-1_8B',
                                        'Qwen/Qwen-1_8B-Chat',
                                        'Qwen/Qwen-7B',
                                        'Qwen/Qwen-7B-Chat',
                                        'internlm/internlm-chat-7b',
                                        'internlm/internlm-7b',
                                        'tiiuae/falcon-rw-1b',
                                        'tiiuae/falcon-7b',
                                        'tiiuae/falcon-7b-instruct',
                                        'deepseek-ai/deepseek-moe-16b-base',
                                        'deepseek-ai/deepseek-moe-16b-chat',
                                        'shibing624/text2vec-base-chinese',
                                        'moka-ai/m3e-base',
                                        'BAAI/bge-large-en-v1.5',
                                        'BAAI/bge-large-zh-v1.5',
                                        'thenlper/gte-large-zh',
                                        'thenlper/gte-base-zh'
                                        ])
def test_download_config(model_name):
    config_b4t = get_config_path(model_name)
    # config = AutoConfig.from_pretrained(model_name)
    print('Done.')


if __name__=='__main__':
    test_hf_download('google-bert/bert-base-chinese')
    # AutoModel.from_pretrained('hfl/chinese-bert-wwm-ext')
    # test_download_config('hfl/chinese-roberta-wwm-ext-large')
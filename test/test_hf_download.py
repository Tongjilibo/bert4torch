'''测试从huggingface上下载模型'''
from bert4torch.models import build_transformer_model
from bert4torch.snippets import get_config_path
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import pytest


# @pytest.mark.parametrize("model_name", ["bert-base-chinese",
#                                         "hfl/chinese-bert-wwm-ext"])
# def test_hf_download(model_name):
#     model = build_transformer_model(checkpoint_path=model_name, with_mlm='softmax')
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     inputtext = "今天[MASK]情很好"
#     encoded_input = tokenizer(inputtext, return_tensors='pt')
#     maskpos = encoded_input['input_ids'][0].tolist().index(103)

#     # 需要传入参数with_mlm
#     model.eval()
#     with torch.no_grad():
#         _, probas = model(**encoded_input)
#         result = torch.argmax(probas[0, [maskpos]], dim=-1).cpu().numpy()
#         pred_token = tokenizer.decode(result)
#     print('pred_token: ', pred_token)
#     assert pred_token == '心'


@pytest.mark.parametrize("model_name", ["bert-base-chinese",
                                        "hfl/chinese-bert-wwm-ext",
                                        "bert-base-multilingual-cased",
                                        "hfl/chinese-macbert-base", 
                                        "hfl/chinese-macbert-large",
                                        "junnyu/wobert_chinese_plus_base",
                                        "junnyu/wobert_chinese_base",
                                        "hfl/chinese-roberta-wwm-ext-base",
                                        "hfl/chinese-roberta-wwm-ext-large",
                                        "roberta-base",
                                        "ethanyt/guwenbert-base",
                                        "hfl/chinese-xlnet-base",
                                        "junnyu/roformer_chinese_base",
                                        "junnyu/roformer_v2_chinese_char_base",
                                        "Tongjilibo/simbert-chinese-base",
                                        "Tongjilibo/simbert-chinese-small",
                                        "Tongjilibo/simbert-chinese-tiny",
                                        "junnyu/roformer_chinese_sim_char_base",
                                        "junnyu/roformer_chinese_sim_char_ft_base",
                                        "junnyu/roformer_chinese_sim_char_small",
                                        "junnyu/roformer_chinese_sim_char_ft_small",
                                        # "thu-coai/CDial-GPT_LCCC-base",
                                        # "thu-coai/CDial-GPT_LCCC-large",
                                        "TsinghuaAI/CPM-Generate",
                                        "uer/gpt2-chinese-cluecorpussmall",
                                        "fnlp/bart-base-chinese",
                                        "uer/t5-base-chinese-cluecorpussmall",
                                        "uer/t5-small-chinese-cluecorpussmall",
                                        "google/mt5-base",
                                        "ClueAI/ChatYuan-large-v1",
                                        "ClueAI/PromptCLUE-base",
                                        "THUDM/chatglm-6b",
                                        "THUDM/chatglm-6b-int8",
                                        "THUDM/chatglm-6b-int4",
                                        "THUDM/chatglm2-6b",
                                        "THUDM/chatglm2-6b-int4",
                                        "THUDM/chatglm2-6b-32k",
                                        "THUDM/chatglm3-6b",
                                        "THUDM/chatglm3-6b-32k",
                                        "llama-7b",
                                        "llama-13b",
                                        "meta-llama/Llama-2-7b-hf",
                                        "meta-llama/Llama-2-7b-chat-hf",
                                        "meta-llama/Llama-2-13b-hf",
                                        "meta-llama/Llama-2-13b-chat-hf",
                                        "chinese_alpaca_plus_7b",
                                        "chinese_llama_plus_7b",
                                        "BelleGroup/BELLE-LLaMA-7B-2M-enc",
                                        "Ziya-LLaMA-13B-v1",
                                        "Ziya-LLaMA-13B-v1.1",
                                        "baichuan-inc/Baichuan-7B",
                                        "baichuan-inc/Baichuan-13B-Base",
                                        "baichuan-inc/Baichuan-13B-Chat",
                                        "baichuan-inc/Baichuan2-7B-Base",
                                        "baichuan-inc/Baichuan2-7B-Chat",
                                        "baichuan-inc/Baichuan2-13B-Base",
                                        "baichuan-inc/Baichuan2-13B-Chat",
                                        "lmsys/vicuna-7b-v1.5",
                                        "01-ai/Yi-6B",
                                        "01-ai/Yi-6B-200K",
                                        "bigscience/bloom-560m",
                                        "bigscience/bloomz-560m",
                                        "Qwen/Qwen-1_8B",
                                        "Qwen/Qwen-1_8B-Chat",
                                        "Qwen/Qwen-7B",
                                        "Qwen/Qwen-7B-Chat",
                                        "internlm/internlm-7b",
                                        "internlm/internlm-chat-7b",
                                        "tiiuae/falcon-rw-1b",
                                        "tiiuae/falcon-7b",
                                        "tiiuae/falcon-7b-instruct",
                                        "deepseek-ai/deepseek-moe-16b-base",
                                        "deepseek-ai/deepseek-moe-16b-chat",
                                        "shibing624/text2vec-base-chinese",
                                        "moka-ai/m3e-base",
                                        "BAAI/bge-large-en-v1.5",
                                        "BAAI/bge-large-zh-v1.5",
                                        "thenlper/gte-base-zh",
                                        "thenlper/gte-large-zh"])
def test_download_config(model_name):
    config_b4t = get_config_path(model_name)
    config = AutoConfig.from_pretrained(model_name)
    print('Done.')


if __name__=='__main__':
    # test_hf_download('hfl/chinese-bert-wwm-ext')
    # AutoModel.from_pretrained('hfl/chinese-bert-wwm-ext')

    test_download_config('THUDM/chatglm-6b')
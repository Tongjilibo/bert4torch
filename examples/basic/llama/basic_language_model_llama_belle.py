#! -*- coding: utf-8 -*-
# 基本测试：belle-llama-7b模型的单论对话测试
# 源项目链接：https://github.com/LianjiaTech/BELLE
# LLaMA模型不允许发布调优后的完整模型权重，但是可以发布原始的模型的diff。因此需要先用脚本合并llama官方权重bell_llama的模型diff
# 模型说明： https://github.com/LianjiaTech/BELLE/tree/main/models
# belle_llama模型：https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc
# 使用前需要先使用转换脚本转换下权重：https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_llama_hf.py


import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import SpTokenizer
from transformers import AutoTokenizer
from bert4torch.generation import AutoRegressiveDecoder, SeqGeneration
import platform
import os

dir_path = 'E:/pretrain_ckpt/llama/belle-llama-7b-2m'
config_path = f'{dir_path}/bert4torch_config.json'
checkpoint_path = f'{dir_path}/bert4torch_pytorch_model.bin'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 可使用bert4torch的tokenizer
use_hf_tokenize = False
if use_hf_tokenize:
    tokenizer = AutoTokenizer.from_pretrained(dir_path, use_fast=False)
else:
    tokenizer = SpTokenizer(dir_path+'/tokenizer.model', token_start='<s>', token_end=None, keep_accents=True, remove_space=False)
print('Loading tokenizer done...')

model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='llama').half()
model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8).to(device)  # 建立模型，加载权重

# 第一种方式
# class ArticleCompletion(AutoRegressiveDecoder):
#     @AutoRegressiveDecoder.wraps(default_rtype='logits')
#     def predict(self, inputs, output_ids, states):
#         token_ids = torch.cat([inputs[0], output_ids], 1)
#         logits = model.predict([token_ids])
#         return logits[:, -1, :]

#     def generate(self, text, n=1, topk=30, topp=0.85, temperature=0.5):
#         if use_hf_tokenize:
#             token_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
#         else:
#             token_ids = tokenizer.encode(text)[0]
#         results = self.random_sample([token_ids], n=n, topk=topk, topp=topp, temperature=temperature)  # 基于随机采样
#         return [tokenizer.decode(ids.cpu().numpy()) for ids in results][0]
# generation = ArticleCompletion(start_id=None, end_id=2, maxlen=512, device=device)

# 第二种方式：调用封装好的接口，可使用cache
class Chat(SeqGeneration):
    def pre_process(self, text):
        if use_hf_tokenize:
            token_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        else:
            token_ids = tokenizer.encode(text)[0]
        return [token_ids]
    def post_process(self, output_ids):
        if use_hf_tokenize:
            return tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        else:
            return tokenizer.decode(output_ids[0].cpu().numpy())
generation = Chat(model, tokenizer, start_id=None, end_id=2, mode='random_sample',
                  maxlen=512, default_rtype='logits', use_states=True)


if __name__ == '__main__':
    os_name = platform.system()
    print("欢迎使用 Belle 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\nUser：")
        if query == "stop":
            break
        if query == "clear":
            command = 'cls' if os_name == 'Windows' else 'clear'
            os.system(command)
            print("欢迎使用 belle 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        # 必须指定human + assistant的prompt
        query = f"Human: {query} \n\nAssistant: "
        response = generation.generate(query, topk=30, topp=0.85, temperature=0.5)
        torch.cuda.empty_cache()  # 清理显存
        print(f"\nBelle：{response}")
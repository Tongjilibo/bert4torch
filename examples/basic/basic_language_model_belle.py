#! -*- coding: utf-8 -*-
# 基本测试：belle-7b模型的基本测试
# belle模型：https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import SpTokenizer
from transformers import AutoTokenizer
from bert4torch.snippets import AutoRegressiveDecoder, SeqGeneration
import platform
import os

dir_path = 'F:/Projects/pretrain_ckpt/llama/belle-llama-7b-2m'
config_path = f'{dir_path}/bert4torch_config.json'
checkpoint_path = f'{dir_path}/bert4torch_pytorch_model.bin'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 经过比对自带的SpTokenizer和transformer中的不完全一致，因此建议使用transformer的（但是from_pretrained这步很慢）
use_hf_tokenize = True
if use_hf_tokenize:
    tokenizer = AutoTokenizer.from_pretrained(dir_path)
else:
    tokenizer = SpTokenizer(dir_path+'/tokenizer.model', token_start='<s>', token_end=None, keep_accents=True)
print('Loading tokenizer done...')

model = build_transformer_model(config_path=config_path,
                                checkpoint_path=checkpoint_path,
                                model='llama').half().quantize(8).to(device)  # 建立模型，加载权重

class ArticleCompletion(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids = torch.cat([inputs[0], output_ids], 1)
        logits = model.predict([token_ids])
        return logits[:, -1, :]

    def generate(self, text, n=1, topk=30, topp=0.85, temperature=0.5, add_input=True):
        if use_hf_tokenize:
            token_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        else:
            token_ids = tokenizer.encode(text)[0]
        results = self.random_sample([token_ids], n, topk=topk, topp=topp, temperature=temperature)  # 基于随机采样
        text = text if add_input else ''
        return [text + tokenizer.decode(ids.cpu().numpy()) for ids in results][0]

generation = ArticleCompletion(
    start_id=1,
    end_id=2,  # </s>标记
    maxlen=256,
    device=device
)


if __name__ == '__main__':
    os_name = platform.system()
    print("Welcome to use belle model，type `clear` to clear history，type `stop` to stop program")
    while True:
        query = input("\nUser：")
        if query == "stop":
            break
        if query == "clear":
            command = 'cls' if os_name == 'Windows' else 'clear'
            os.system(command)
            print("Welcome to use belle model，type `clear` to clear history，type `stop` to stop program")
            continue
        # 必须指定human + assistant的prompt
        query = f"Human: {query} \n\nAssistant: "
        response = generation.generate(query, topk=30, topp=0.85, temperature=0.5, add_input=False)
        torch.cuda.empty_cache()  # 清理显存
        print(f"\nbelle：{response}")
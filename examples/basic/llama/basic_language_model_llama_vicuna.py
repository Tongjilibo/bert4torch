#! -*- coding: utf-8 -*-
# 基本测试：vicuna的7b模型的测试
# bert4torch_config.json文件见readme

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import SpTokenizer
from bert4torch.generation import AutoRegressiveDecoder, SeqGeneration
import platform
import os


dir_path = 'E:/pretrain_ckpt/llama/lmsys@vicuna-7b-v1.5/'
config_path = dir_path + 'bert4torch_config.json'
checkpoint_path = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('.bin')]
spm_path = dir_path + 'tokenizer.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = SpTokenizer(spm_path, token_start='<s>', token_end=None, keep_accents=True)

model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path).half()
model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8).to(device)  # 建立模型，加载权重


# 第一种方式
class ArticleCompletion(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states):
        token_ids = torch.cat([inputs[0], output_ids], 1)
        logits = model.predict([token_ids])
        return logits[:, -1, :]

    def generate(self, text, n=1, topp=0.95):
        token_ids, _ = tokenizer.encode(text)
        results = self.random_sample([token_ids], n=n, topp=topp)  # 基于随机采样
        return [tokenizer.decode(ids.cpu().numpy()) for ids in results]


article_completion = ArticleCompletion(
    bos_token_id=None,
    eos_token_id=2,  # </s>标记
    max_new_tokens=256,
    device=device
)

# 第二种方式
article_completion = SeqGeneration(model, tokenizer, start_id=None, end_id=2, mode='random_sample',
                                   maxlen=256, default_rtype='logits', use_states=True)

if __name__ == '__main__':
    os_name = platform.system()
    print("Welcome to use vicuna-7b model，type `clear` to clear history，type `stop` to stop program")
    while True:
        query = input("\nUser：")
        if query == "stop":
            break
        if query == "clear":
            command = 'cls' if os_name == 'Windows' else 'clear'
            os.system(command)
            print("Welcome to use vicuna-7b model，type `clear` to clear history，type `stop` to stop program")
            continue
        response = article_completion.generate(query, include_input=True)
        torch.cuda.empty_cache()  # 清理显存
        print(f"\nvicuna-7b：{response}")

#! -*- coding: utf-8 -*-
"""
基本测试：baichuan模型的测试 https://github.com/baichuan-inc/Baichuan-7B
使用前需要进行权重转换 https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_llama_hf.py
"""

import torch
from bert4torch.models import build_transformer_model
from bert4torch.tokenizers import SpTokenizer
from bert4torch.generation import SeqGeneration
from transformers import AutoTokenizer
import platform
import os

dir_path = 'E:/pretrain_ckpt/llama/Baichuan-7B/'
config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = dir_path + '/bert4torch_pytorch_model.bin'
spm_path = dir_path + '/tokenizer.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-7B", trust_remote_code=True)
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='llama').half().to(device)
# model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8).to(device)  # 建立模型，加载权重

tokenizer_config = {'skip_special_tokens': True}
article_completion = SeqGeneration(model, tokenizer, start_id=None, end_id=2, mode='random_sample', tokenizer_config=tokenizer_config,
                                   maxlen=64, default_rtype='logits', use_states=True)


if __name__ == '__main__':
    os_name = platform.system()
    print("Welcome to use baichuan model，type `clear` to clear history，type `stop` to stop program")
    while True:
        query = input("\nUser：")
        if query == "stop":
            break
        if query == "clear":
            command = 'cls' if os_name == 'Windows' else 'clear'
            os.system(command)
            print("Welcome to use baichuan model，type `clear` to clear history，type `stop` to stop program")
            continue
        response = article_completion.generate(query, repetition_penalty=1.1, include_input=True)      
        torch.cuda.empty_cache()  # 清理显存
        print(f"\nbaichuan：{response}")
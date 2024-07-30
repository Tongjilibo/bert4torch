#! -*- coding: utf-8 -*-
"""
基本测试：falcon模型的测试

falcon-rw-1b:   https://huggingface.co/tiiuae/falcon-rw-1b
falcon-7b:   https://huggingface.co/tiiuae/falcon-7b
falcon-7b-instruct:   https://huggingface.co/tiiuae/falcon-7b-instruct
"""

import torch
from bert4torch.models import build_transformer_model
from bert4torch.generation import SeqGeneration
from transformers import AutoTokenizer
import platform
import os

model_name = 'falcon-7b-instruct'  # falcon-rw-1b falcon-7b falcon-7b-instruct
model_dir = f'/data/pretrain_ckpt/falcon/{model_name}'
include_input = False if '-instruct' in model_dir else True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir)
# model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8)
model = model.to(device)

generation = SeqGeneration(model, tokenizer, bos_token_id=None, eos_token_id=tokenizer.eos_token_id, mode='random_sample', 
                           tokenizer_config={'skip_special_tokens': True}, max_length=200, default_rtype='logits', use_states=True)


if __name__ == '__main__':
    os_name = platform.system()
    print("Welcome to use falcon model，type `clear` to clear history，type `stop` to stop program")
    while True:
        query = input("\nUser：")
        if query == "stop":
            break
        if query == "clear":
            command = 'cls' if os_name == 'Windows' else 'clear'
            os.system(command)
            print("Welcome to use falcon model，type `clear` to clear history，type `stop` to stop program")
            continue
        # 官方测试用例
        # query = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"
        response = generation.generate(query, top_k=10, include_input=include_input)      
        torch.cuda.empty_cache()  # 清理显存
        print(f"\nfalcon：{response}")
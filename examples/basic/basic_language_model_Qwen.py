#! -*- coding: utf-8 -*-
"""通义千问Qwen的测试
阿里云的通义千问: https://github.com/QwenLM/Qwen-7B
使用前需要进行权重转换 https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_Qwen.py
"""

import torch
from bert4torch.models import build_transformer_model
from typing import Tuple, List, Union, Iterable
from bert4torch.generation import SeqGeneration
from transformers import AutoTokenizer
import platform
import os

choice = 'Qwen-7B-Chat'
if choice == 'Qwen-7B-Chat':
    dir_path = 'E:/pretrain_ckpt/Qwen/Qwen-7B-Chat'
    with_prompt = True
elif choice == 'Qwen-7B':
    dir_path = 'E:/pretrain_ckpt/Qwen/Qwen-7B'
    with_prompt = False
else:
    raise ValueError(f'{choice} not in pre maintained choices')
include_input = not with_prompt

config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = dir_path + '/bert4torch_pytorch_model.bin'
spm_path = dir_path + '/tokenizer.model'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=True)
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path).half()
# model = model.quantize(quantization_method='cpm_kernels', quantization_bit=8)
model = model.to(device)

def make_context(tokenizer, query: str, history: List[Tuple[str, str]] = None, system: str = "", max_window_size: int = 6144, chat_format: str = "chatml"):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"

        def _tokenize_str(role, content):
            return f"{role}\n{content}"

        system_text = _tokenize_str("system", system)
        raw_text = ""

        for turn_query, turn_response in reversed(history):
            query_text = _tokenize_str("user", turn_query)
            response_text = _tokenize_str("assistant", turn_response)
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = len(tokenizer.encode(raw_text, allowed_special={im_start, im_end}))
            if current_context_size < max_window_size:
                raw_text = prev_chat + raw_text
            else:
                break

        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text

def build_prompt(query, response, history):
    prompt = "Welcome to use Qwen model，type `clear` to clear history，type `stop` to stop program"
    for q, a in history:
        prompt += f"\n\nUser：{q}"
        prompt += f"\n\nQwen：{a}"
    prompt += f"\n\nUser：{query}"
    prompt += f"\n\nQwen：{response}"
    return prompt

tokenizer_encode_config = {'allowed_special': {"<|im_start|>", "<|im_end|>", '<|endoftext|>'}}
tokenizer_decode_config = {'skip_special_tokens': True}
end_id = [tokenizer.im_start_id, tokenizer.im_end_id] if with_prompt else tokenizer.encode("<|endoftext|>", **tokenizer_encode_config)
chat = SeqGeneration(model, tokenizer, start_id=None, end_id=end_id, mode='random_sample', 
                     tokenizer_encode_config=tokenizer_encode_config, tokenizer_decode_config=tokenizer_decode_config,
                     maxlen=256, default_rtype='logits', use_states=True)


if __name__ == '__main__':
    history = []
    os_name = platform.system()
    clear_command = 'cls' if os_name == 'Windows' else 'clear'
    print("Welcome to use Qwen model，type `clear` to clear history，type `stop` to stop program")
    while True:
        query_input = query = input("\nUser：")
        if query == "stop":
            break
        if query == "clear":
            history = []
            os.system(clear_command)
            print("Welcome to use Qwen model，type `clear` to clear history，type `stop` to stop program")
            continue
        if with_prompt:
            query = make_context(tokenizer, query, history, system='You are a helpful assistant.')
        else:
            query = make_context(tokenizer, query, None, chat_format='raw')

        for response in chat.stream_generate(query, include_input=include_input):
            os.system(clear_command)
            print(build_prompt(query_input, response, history), flush=True)

        if with_prompt:
            history.append((query_input, response))
        torch.cuda.empty_cache()  # 清理显存

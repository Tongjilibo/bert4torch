#! -*- coding: utf-8 -*-
"""DeepSeek OCR
"""
from bert4torch import build_transformer_model, AutoTokenizer
import torch


prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = '/data/pretrain_ckpt/deepseek-ai/DeepSeek-OCR-2/image.png'

def chat_demo1():
    model_dir = '/data/pretrain_ckpt/deepseek-ai/DeepSeek-OCR'
    from bert4torch.models.deepseek_ocr import process_vision_info
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    res = process_vision_info(tokenizer, prompt=prompt, image_file=image_file, base_size = 1024, image_size = 768, crop_mode=True)

    model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir)
    model = model.eval().cuda().to(torch.bfloat16)
    output = model.generate(**res, max_new_tokens=1024, pad_token_id=2, eos_token_id=1, no_repeat_ngram_size=20, do_sample=False)
    print(tokenizer.batch_decode(output, skip_special_tokens=True)[0])


def chat_demo2():
    model_dir = '/data/pretrain_ckpt/deepseek-ai/DeepSeek-OCR-2'
    from bert4torch.models.deepseek_ocr2 import process_vision_info_v2
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    res = process_vision_info_v2(tokenizer, prompt=prompt, image_file=image_file, base_size = 1024, image_size = 768, crop_mode=True)

    model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir)
    model = model.eval().cuda().to(torch.bfloat16)
    output = model.generate(**res, max_new_tokens=1024, pad_token_id=2, eos_token_id=1, no_repeat_ngram_size=20, do_sample=False)
    print(tokenizer.batch_decode(output, skip_special_tokens=True)[0])


if __name__ == "__main__":
    chat_demo1()
    chat_demo2()
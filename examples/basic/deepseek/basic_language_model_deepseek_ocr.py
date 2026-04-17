#! -*- coding: utf-8 -*-
"""DeepSeek OCR
"""
from bert4torch import build_transformer_model, AutoProcessor
from transformers import AutoTokenizer
import torch


model_dir = '/data/pretrain_ckpt/deepseek-ai/DeepSeek-OCR'
# model_dir = '/data/pretrain_ckpt/deepseek-ai/DeepSeek-OCR-2'
prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = '/data/pretrain_ckpt/deepseek-ai/DeepSeek-OCR-2/image.png'
device = 'cuda'

def chat_demo1():
    from bert4torch.models.deepseek_ocr import process_vision_info
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    res = process_vision_info(tokenizer, prompt=prompt, image_file=image_file, base_size = 1024, image_size = 640, crop_mode=True, crop_thread=640)

    model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir)
    model = model.eval().cuda().to(torch.bfloat16)
    output = model.generate(**res, max_new_tokens=1024, pad_token_id=2, eos_token_id=1, no_repeat_ngram_size=20, do_sample=False)
    print(tokenizer.batch_decode(output, skip_special_tokens=True)[0])


def chat_demo2():
    from bert4torch.models.deepseek_ocr2 import process_vision_info_v2
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    res = process_vision_info_v2(tokenizer, prompt=prompt, image_file=image_file, base_size = 1024, image_size = 768, crop_mode=True)

    model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir)
    model = model.eval().cuda().to(torch.bfloat16)
    output = model.generate(**res, max_new_tokens=1024, pad_token_id=2, eos_token_id=1, no_repeat_ngram_size=20, do_sample=False)
    print(tokenizer.batch_decode(output, skip_special_tokens=True)[0])


if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained(model_dir)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    # "max_pixels": 512 * 512,
                },
                {"type": "text", "text": 'Convert the document to markdown.'},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)
    print(inputs)
    
    # chat_demo1()
    # chat_demo2()
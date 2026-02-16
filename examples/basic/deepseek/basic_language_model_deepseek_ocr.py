#! -*- coding: utf-8 -*-
"""DeepSeek OCR
"""

from bert4torch.models import build_transformer_model
import torch

model_dir = '/data/pretrain_ckpt/deepseek-ai/DeepSeek-OCR-2'
prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = '/data/pretrain_ckpt/deepseek-ai/DeepSeek-OCR-2/image.png'
output_path = '/data/pretrain_ckpt/deepseek-ai/DeepSeek-OCR-2/output'

def chat_demo1():
    from transformers import AutoModel, AutoTokenizer
    from bert4torch.models.deepseek.deepseek_ocr2 import process_vision_info
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    res = process_vision_info(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 768, crop_mode=True, save_results = True)

    model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir).to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            output = model.generate(**res, max_new_tokens=256)
    print(tokenizer.batch_decode(output, skip_special_tokens=True)[0])


if __name__ == "__main__":
    chat_demo1()
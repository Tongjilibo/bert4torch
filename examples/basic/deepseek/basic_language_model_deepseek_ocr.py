#! -*- coding: utf-8 -*-
"""DeepSeek OCR系列
"""
from bert4torch import build_transformer_model, AutoProcessor, AutoTokenizer
import torch


image_file = '/data/pretrain_ckpt/deepseek-ai/DeepSeek-OCR-2/image.png'
device = 'cuda'
# model_dir = '/data/pretrain_ckpt/deepseek-ai/DeepSeek-OCR'
model_dir = '/data/pretrain_ckpt/deepseek-ai/DeepSeek-OCR-2'


if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained(model_dir)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_file,
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

    model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir)
    model = model.eval().cuda().to(torch.bfloat16)
    output = model.generate(**inputs, max_new_tokens=1024, pad_token_id=2, eos_token_id=1, no_repeat_ngram_size=20, do_sample=False)
    print(processor.batch_decode(output, skip_special_tokens=True)[0])
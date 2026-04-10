#! -*- coding: utf-8 -*-
"""PaddleOCR-VL的测试
"""

from PIL import Image
from bert4torch.pipelines import Chat
from bert4torch import build_transformer_model, AutoProcessor
import torch

root_dir = '/data/pretrain_ckpt/PaddlePaddle'
# model_dir = f'{root_dir}/PaddleOCR-VL'
model_dir = f'{root_dir}/PaddleOCR-VL-1.5'
image_path = "/home/lb/projects/tongjilibo/bert4torch/test_local/images/表格1.png"


def demo_generate():
    '''直接调用model.generate进行生成'''
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CHOSEN_TASK = "ocr"  # Options: 'ocr' | 'table' | 'chart' | 'formula'
    PROMPTS = {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
        "spotting": "Spotting:",  # 1.5新增
        "seal": "Seal Recognition:",  # 1.5新增
    }

    image = Image.open(image_path).convert("RGB")

    processor = AutoProcessor.from_pretrained(model_dir) #, trust_remote_code=True)
    model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir).to("cuda")

    messages = [
        {"role": "user",
        "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPTS[CHOSEN_TASK]},
            ]
        }
    ]
    inputs = processor.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 	
        return_dict=True,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = model.generate(**inputs, max_new_tokens=1024)
    outputs = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(outputs)


def demo_chat():
    demo = Chat(model_dir, mode='gradio')
    demo.run()


if __name__ == '__main__':
    demo_generate()
    # demo_chat()
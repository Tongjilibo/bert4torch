#! -*- coding: utf-8 -*-
"""glmocr的测试
"""

from bert4torch.pipelines import Chat
from bert4torch import build_transformer_model, AutoProcessor
import torch

model_dir = '/data/pretrain_ckpt/zai-org/GLM-OCR'
image_path = "/data/pretrain_ckpt/zai-org/GLM-OCR/image.png"


def demo_generate():
    '''直接调用model.generate进行生成'''
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {
                    "type": "text",
                    "text": "Text Recognition:"
                }
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    inputs = processor.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 	
        return_dict=True,
        return_tensors="pt"
    ).to(DEVICE)

    model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir).to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=512)
    outputs = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(outputs)


def demo_chat():
    demo = Chat(model_dir, mode='gradio')
    demo.run()


if __name__ == '__main__':
    demo_generate()
    # demo_chat()
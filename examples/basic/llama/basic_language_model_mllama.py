#! -*- coding: utf-8 -*-
"""
基本测试: 原生llama模型的测试
"""

from bert4torch.models import build_transformer_model
import requests
from PIL import Image
from transformers import AutoProcessor
from bert4torch.pipelines import ChatVL


model_dir = 'E:/data/pretrain_ckpt/llama/Llama-3.2-11B-Vision-Instruct'

def chat_demo1():
    device = 'cuda'

    processor = AutoProcessor.from_pretrained(model_dir)

    url = "https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    model = build_transformer_model(checkpoint_path=model_dir, device_map="auto")

    while True:
        query = input('\nUser: ')
        # query = 'If I had to write a haiku for this one, it would be: '
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": query}  # If I had to write a haiku for this one, it would be: 
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(device)

        output = model.generate(**inputs, max_new_tokens=30)
        print(processor.decode(output[0]))


def chat_demo2():
    demo = ChatVL(model_dir, 
                # quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
                mode='gradio'
                )
    demo.run()


if __name__ == '__main__':
    chat_demo1()
    # chat_demo2()

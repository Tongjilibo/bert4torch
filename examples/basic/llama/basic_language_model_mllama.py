#! -*- coding: utf-8 -*-
"""
基本测试: 原生llama模型的测试
"""

from bert4torch import build_transformer_model, AutoProcessor
from PIL import Image
from bert4torch.pipelines import Chat

model_dir = '/data/pretrain_ckpt/meta-llama/Llama-3.2-11B-Vision-Instruct'

def chat_demo1():
    device = 'cuda'

    processor = AutoProcessor.from_pretrained(model_dir)

    image = Image.open('./data/images/rabbit.jpg').convert('RGB')

    model = build_transformer_model(model_dir, device_map="auto")

    while True:
        query = input('\nUser: ')
        if query == '':
            query = 'If I had to write a haiku for this one, it would be: '
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": query}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(device)

        output = model.generate(**inputs, max_new_tokens=256)
        print('Bot: ', processor.decode(output[0]))


def chat_demo2():
    demo = Chat(model_dir, mode='gradio')
    demo.run()


if __name__ == '__main__':
    chat_demo1()
    # chat_demo2()

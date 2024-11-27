#! -*- coding: utf-8 -*-
"""通义千问Qwen2VL的测试
阿里云的通义千问: https://github.com/QwenLM/Qwen2-VL
"""

from bert4torch.pipelines import ChatVL, ChatOpenaiClient
from bert4torch.models import build_transformer_model
from bert4torch.models.qwen2_vl import process_vision_info
from transformers import AutoProcessor
from PIL import Image
import base64
import io

# Qwen2-VL-2B-Instruct
# Qwen2-VL-7B-Instruct
model_dir = 'E:/data/pretrain_ckpt/Qwen/Qwen2-VL-2B-Instruct'

def chat_demo1():
    device = 'cuda'
    model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir).to(device)

    processor = AutoProcessor.from_pretrained(model_dir)

    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


    # Inference: Generation of the output
    while True:
        query = input('User: ')
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                        "max_pixels": 512 * 512,
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # 一次性输出
        # generated_ids = model.generate(**inputs, max_new_tokens=128, top_k=1, pad_token_id=151643, eos_token_id=[151645, 151643])
        # output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(f'Bot: {output_text}\n')

        # 流式输出
        print('Bot: ', end='')
        last_len = 0
        for generated_ids in model.stream_generate(**inputs, max_new_tokens=128, top_k=1, pad_token_id=151643, eos_token_id=[151645, 151643]):
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(output_text[last_len:], flush=True, end='')
            last_len = len(output_text)
        print('\n')

def chat_demo2():
    demo = ChatVL(model_dir, 
                # quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
                mode='openai',
                template='qwen2_vl'
                )
    demo.run()

def call_openai():
    '''调用openai接口的多模态模型
    '''
    image = Image.open('E:/Github/bert4torch/test_local/images/资料概要.png').convert('RGB')

    # save by BytesIO and convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="png")
    im_b64 = base64.b64encode(buffered.getvalue()).decode()

    # First round chat 
    question = "图片中的基金名称叫什么"

    messages = [{'role': 'user', 
                'content': [
                    {'type': 'text', 'text': question}, 
                    {'type': 'image_url', 'url': im_b64}
                    ]
                }]

    # messages = [
    #              {"content": "你好", "role": "user"},
    #              {"content": "你好，我是AI大模型，有什么可以帮助您的？", "role": "assistant"},
    #              {"content": "你可以做什么？", "role": "user"}
    #              ]

    client = ChatOpenaiClient('http://127.0.0.1:8000', api_key='EMPTY')
    for token in client.stream_chat(messages):
        print(token, end='', flush=True)

    # 非流式
    print(client.chat(messages))

if __name__ == '__main__':
    # chat_demo1()
    chat_demo2()
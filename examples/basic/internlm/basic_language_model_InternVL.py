#! -*- coding: utf-8 -*-
"""书生浦语InternLM的测试
Github: https://github.com/InternLM/InternLM
bert4torch_config.json见readme

[1] internlm-chat-7b模型：https://huggingface.co/internlm/internlm-chat-7b
"""

from bert4torch.pipelines import Chat
import re
from bert4torch.models import build_transformer_model

# InternVL2_5-1B
# InternVL2_5-2B
# InternVL2_5-4B
# InternVL2_5-8B
model_dir = 'E:/data/pretrain_ckpt/internlm/InternVL2_5-1B'


def chat_demo1():
    device = 'cuda'
    model = build_transformer_model(config_path=model_dir, checkpoint_path=model_dir).to(device)

    # # Inference: Generation of the output
    # while True:
    #     query = input('User: ')
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "image",
    #                     "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    #                     "max_pixels": 512 * 512,
    #                 },
    #                 {"type": "text", "text": query},
    #             ],
    #         }
    #     ]

    #     # Preparation for inference
    #     text = processor.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    #     image_inputs, video_inputs = process_vision_info(messages)
    #     inputs = processor(
    #         text=[text],
    #         images=image_inputs,
    #         videos=video_inputs,
    #         padding=True,
    #         return_tensors="pt",
    #     ).to(device)

    #     # 一次性输出
    #     # generated_ids = model.generate(**inputs, max_new_tokens=128, top_k=1, pad_token_id=151643, eos_token_id=[151645, 151643])
    #     # output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #     # print(f'Bot: {output_text}\n')

    #     # 流式输出
    #     print('Bot: ', end='')
    #     last_len = 0
    #     for generated_ids in model.stream_generate(**inputs, max_new_tokens=128, top_k=1, pad_token_id=151643, eos_token_id=[151645, 151643]):
    #         output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #         print(output_text[last_len:], flush=True, end='')
    #         last_len = len(output_text)
    #     print('\n')


def chat_demo2():
    generation_config = {
        'top_p': 0.8, 
        'temperature': 1,
        'repetition_penalty': 1.005, 
        'top_k': 40,
        'include_input': False if re.search('chat|instruct', model_dir) else True
    }

    system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
    - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
    - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."""

    cli_demo = Chat(model_dir, 
                    system=system_prompt,
                    generation_config=generation_config,
                    mode='cli'
                    )
    cli_demo.run()


if __name__ == '__main__':
    chat_demo1()
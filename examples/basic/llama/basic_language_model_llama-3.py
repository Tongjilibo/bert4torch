#! -*- coding: utf-8 -*-
"""
基本测试：原生llama3模型的推理

bert4torch_config.json链接
- https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Meta-Llama-3-8B
- https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Meta-Llama-3-8B-Instruct
"""
from bert4torch.pipelines import Chat

# Meta-Llama-8B-Instruct  Meta-Llama-8B-Instruct
model_dir = 'E:/data/pretrain_ckpt/llama/Meta-Llama-3-8B-Instruct'
with_prompt = True if 'Instruct' in model_dir else False


generation_config = {
    'max_length': 512, 
    'include_input': not with_prompt,
    'temperature': 0.6,
    'top_p': 0.9
}

cli_demo = Chat(model_dir, generation_config=generation_config)


if __name__ == '__main__':
    if with_prompt:
        # chat模型
        cli_demo.run()
    else:
        # 预训练模型
        while True:
            query = input('\nUser:')
            response = cli_demo.generate(query)
            print(f'Llama: {response}')

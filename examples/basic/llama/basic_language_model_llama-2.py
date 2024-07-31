#! -*- coding: utf-8 -*-
"""
基本测试：原生llama模型的测试 https://github.com/facebookresearch/llama
"""
from bert4torch.pipelines import Chat


# llama-2-7b  llama-2-7b-chat  llama-2-13b  llama-2-13b-chat
dir_path = 'E:/data/pretrain_ckpt/llama/llama-2-7b-chat'
with_prompt = True if 'chat' in dir_path else False

generation_config = {'max_length': 512, 'include_input': not with_prompt}
cli_demo = Chat(dir_path, generation_config=generation_config)


if __name__ == '__main__':
    if with_prompt:
        # chat模型
        cli_demo.run()
    else:
        # 预训练模型
        while True:
            query = input('\n输入:')
            response = cli_demo.generate(query)
            print(f'续写: {response}')

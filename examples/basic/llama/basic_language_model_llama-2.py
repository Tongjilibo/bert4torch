#! -*- coding: utf-8 -*-
"""
基本测试：原生llama模型的测试 https://github.com/facebookresearch/llama
"""


choice = 'llama-2-7b'
if choice == 'llama-2-7b':
    dir_path = 'E:/pretrain_ckpt/llama/llama-2-7b'
    with_prompt = False
elif choice == 'llama-2-7b-chat':
    dir_path = 'E:/pretrain_ckpt/llama/llama-2-7b-chat'
    with_prompt = True
elif choice == 'llama-2-13b':
    dir_path = 'E:/pretrain_ckpt/llama/llama-2-13b'
    with_prompt = False
elif choice == 'llama-2-13b-chat':
    dir_path = 'E:/pretrain_ckpt/llama/llama-2-13b-chat'
    with_prompt = True        
else:
    raise ValueError(f'{choice} not in pre maintained choices')
include_input = not with_prompt


from bert4torch.pipelines import ChatLLaMA2Cli
generation_config = {
    'tokenizer_config':  {'skip_special_tokens': True, 'add_special_tokens': False},
    'end_id': 2,
    'mode': 'random_sample', 
    'maxlen': 512, 
    'default_rtype': 'logits', 
    'use_states': True,
    'include_input': include_input
}


cli_demo = ChatLLaMA2Cli(dir_path, generation_config=generation_config)

if __name__ == '__main__':
    if with_prompt:
        # chat模型
        cli_demo.run()
    else:
        # 预训练模型
        while True:
            query = input('\n输入:')
            response = cli_demo.model.generate(query, **generation_config)
            print(f'续写: {response}')

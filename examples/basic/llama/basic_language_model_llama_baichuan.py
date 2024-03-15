#! -*- coding: utf-8 -*-
"""
基本测试：baichuan模型的测试 https://github.com/baichuan-inc/Baichuan-7B
"""
from bert4torch.pipelines import ChatBaichuanCli

choice = 'Baichuan2-7B-Chat'

if choice == 'Baichuan-7B':
    dir_path = 'E:\\pretrain_ckpt\\llama\\Baichuan-7B'
    with_prompt = False
    maxlen = 64
    topk, topp, temperature, repetition_penalty = 50, 1, 1, 1.1
elif choice == 'Baichuan-13B':
    dir_path = 'E:\\pretrain_ckpt\\llama\\Baichuan-13B'
    with_prompt = False
    maxlen = 64
    topk, topp, temperature, repetition_penalty = 50, 1, 1, 1.1
elif choice == 'Baichuan-13B-Chat':
    dir_path = 'E:\\pretrain_ckpt\\llama\\Baichuan-13B-Chat'
    with_prompt = True
    maxlen = 4096
    topk, topp, temperature, repetition_penalty = 5, 0.85, 0.3, 1.1
elif choice == 'Baichuan2-7B-Chat':
    dir_path = 'E:\\pretrain_ckpt\\llama\\Baichuan2-7B-Chat'
    with_prompt = True
    maxlen = 2048
    topk, topp, temperature, repetition_penalty = 5, 0.85, 0.3, 1.05
else:
    raise ValueError(f'{choice} not in pre maintained choices')
include_input = not with_prompt


generation_config = {
    'end_id': 2, 
    'mode':'random_sample', 
    'tokenizer_config': {'skip_special_tokens': True},
    'max_length':maxlen, 
    'default_rtype': 'logits', 
    'use_states': True,
    'topk': topk, 
    'topp': topp, 
    'temperature': temperature, 
    'repetition_penalty': repetition_penalty,
    'include_input': include_input
}

cli_demo = ChatBaichuanCli(
    dir_path, generation_config=generation_config,
    quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
    )


if __name__ == '__main__':
    if with_prompt:
        # chat模型
        cli_demo.run(stream=True)
    else:
        # 预训练模型
        while True:
            query = input('\n输入:')
            response = cli_demo.model.generate(query, **generation_config)
            print(f'续写: {response}')

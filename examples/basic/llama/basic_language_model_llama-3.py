#! -*- coding: utf-8 -*-
"""
基本测试：原生llama3模型的推理

bert4torch_config.json链接
- https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Meta-Llama-3-8B
- https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Meta-Llama-3-8B-Instruct
"""
from bert4torch.pipelines import ChatLLaMA3Cli
from transformers import AutoTokenizer

choice = 'Meta-Llama-8B-Instruct'
if choice == 'Meta-Llama-8B':
    dir_path = '/data/pretrain_ckpt/llama/Meta-Llama-3-8B'
    with_prompt = False
elif choice == 'Meta-Llama-8B-Instruct':
    dir_path = '/data/pretrain_ckpt/llama/Meta-Llama-3-8B-Instruct'
    with_prompt = True
else:
    raise ValueError(f'{choice} not in pre maintained choices')
include_input = not with_prompt


tokenizer = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=True)
generation_config = {
    'tokenizer_config':  {'skip_special_tokens': True, 'add_special_tokens': False},
    'end_id': [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    'mode': 'random_sample', 
    'max_length': 512, 
    'default_rtype': 'logits', 
    'use_states': True,
    'include_input': include_input,
    'temperature': 0.6,
    'top_p': 0.9
}


cli_demo = ChatLLaMA3Cli(dir_path, generation_config=generation_config)

if __name__ == '__main__':
    if with_prompt:
        # chat模型
        cli_demo.run()
    else:
        # 预训练模型
        while True:
            query = input('\nUser:')
            response = cli_demo.model.generate(query, **generation_config)
            print(f'Llama: {response}')

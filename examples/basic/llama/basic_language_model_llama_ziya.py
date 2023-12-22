#! -*- coding: utf-8 -*-
"""
基本测试：ziya系列模型的测试, bert4torch_config.json见readme

Ziya-LLaMA-13B_v1.1: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1
Ziya-LLaMA-13B_v1: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
Ziya-LLaMA-13B_pretrain: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1
"""


choice = 'Ziya-LLaMA-13B_v1.1'
if choice == 'Ziya-LLaMA-13B_v1.1':
    dir_path = 'E:/pretrain_ckpt/llama/IDEA-CCNL@Ziya-LLaMA-13B-v1.1'
    with_prompt = True
elif choice == 'Ziya-LLaMA-13B_v1':
    dir_path = 'E:/pretrain_ckpt/llama/IDEA-CCNL@Ziya-LLaMA-13B-v1'
    with_prompt = True
elif choice == 'Ziya-LLaMA-13B_pretrain':
    dir_path = 'E:/pretrain_ckpt/llama/IDEA-CCNL@Ziya-LLaMA-13B_pretrain'
    with_prompt = False
else:
    raise ValueError(f'{choice} not in pre maintained choices')
include_input = not with_prompt


from bert4torch.chat import CliDemoZiya
tokenizer_config = {'skip_special_tokens': True}
generation_config = {
    'end_id': 2, 
    'mode': 'random_sample', 
    'tokenizer_config': tokenizer_config,
    'maxlen': 256, 
    'default_rtype': 'logits', 
    'use_states': True,
    'include_input': include_input,
}


cli_demo = CliDemoZiya(
    dir_path, generation_config=generation_config,
    quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
    )

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

#! -*- coding: utf-8 -*-
"""
基本测试：ziya系列模型的测试, bert4torch_config.json见readme

Ziya-LLaMA-13B_v1.1: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1
Ziya-LLaMA-13B_v1: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
Ziya-LLaMA-13B_pretrain: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1
"""
from bert4torch.pipelines import Chat


# Ziya-LLaMA-13B_v1.1 Ziya-LLaMA-13B_v1 Ziya-LLaMA-13B_pretrain
dir_path = '/data/pretrain_ckpt/llama/IDEA-CCNL@Ziya-LLaMA-13B-v1.1'
with_prompt = False if '_pretrain' in dir_path else True

generation_config = {'include_input': not with_prompt}


cli_demo = Chat(
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
            response = cli_demo.generate(query)
            print(f'续写: {response}')

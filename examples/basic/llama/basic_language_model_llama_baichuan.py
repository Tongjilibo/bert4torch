#! -*- coding: utf-8 -*-
"""
基本测试：baichuan模型的测试 https://github.com/baichuan-inc/Baichuan-7B
"""
from bert4torch.pipelines import ChatBaichuanCli

# Baichuan-7B Baichuan-13B-Base Baichuan-13B-Chat Baichuan2-7B-Base Baichuan2-7B-Chat Baichuan2-13B-Base Baichuan2-13B-Chat
model_dir = 'E:/pretrain_ckpt/llama/Baichuan2-7B-Chat'
with_prompt = True if 'Chat' in model_dir else False


generation_config = {
    'max_length': 1024, 
    'topk': 40, 
    'topp': 0.9, 
    'temperature': 0.9, 
    'repetition_penalty': 1,
    'include_input': not with_prompt
}

cli_demo = ChatBaichuanCli(
    model_dir, generation_config=generation_config,
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
            response = cli_demo.generate(query)
            print(f'续写: {response}')

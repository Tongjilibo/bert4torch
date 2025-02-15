#! -*- coding: utf-8 -*-
"""
基本测试：chinese_llama_apaca模型的测试 https://github.com/ymcui/Chinese-LLaMA-Alpaca
"""

# chinese-alpaca-plus-7b，chinese-llama-plus-7b
model_dir = 'E:/data/pretrain_ckpt/llama/chinese-alpaca-plus-7b'
with_prompt = True if 'alpaca' in model_dir else False


from bert4torch.pipelines import Chat
generation_config = {
    'max_length': 256, 
    'include_input': not with_prompt,
    'top_k': 40, 
    'top_p': 0.9, 
    'temperature': 0.2, 
    'repetition_penalty': 1.3
}


cli_demo = Chat(
    model_dir, 
    generation_config=generation_config,
    # quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
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

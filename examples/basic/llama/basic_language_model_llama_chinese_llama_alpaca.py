#! -*- coding: utf-8 -*-
"""
基本测试：chinese_llama_apaca模型的测试 https://github.com/ymcui/Chinese-LLaMA-Alpaca
"""

# dir_path = 'E:/pretrain_ckpt/llama/hfl@chinese_llama_plus_7b'
# with_prompt = False

dir_path = 'E:/pretrain_ckpt/llama/hfl@chinese_alpaca_plus_7b'
with_prompt = True
include_input = not with_prompt


from bert4torch.pipelines import ChatChineseAlphaLLaMACli
generation_config = {
    'end_id': 2, 
    'mode': 'random_sample', 
    'tokenizer_config': {'skip_special_tokens': True},
    'max_length': 256, 
    'default_rtype': 'logits', 
    'use_states': True,
    'include_input': include_input,
    'topk': 40, 
    'topp': 0.9, 
    'temperature': 0.2, 
    'repetition_penalty': 1.3
}


cli_demo = ChatChineseAlphaLLaMACli(
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

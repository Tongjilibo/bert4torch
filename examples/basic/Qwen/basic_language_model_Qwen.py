#! -*- coding: utf-8 -*-
"""通义千问Qwen的测试
阿里云的通义千问: https://github.com/QwenLM/Qwen-7B
bert4torch_config.json见readme
"""
from transformers import AutoTokenizer
from bert4torch.pipelines import ChatQwenCli


choice = 'Qwen-1_8B-Chat'
if choice == 'Qwen-7B-Chat':
    dir_path = 'E:/pretrain_ckpt/Qwen/Qwen-7B-Chat'
    with_prompt = True
elif choice == 'Qwen-7B':
    dir_path = 'E:/pretrain_ckpt/Qwen/Qwen-7B'
    with_prompt = False
elif choice == 'Qwen-1_8B-Chat':
    dir_path = 'E:/pretrain_ckpt/Qwen/Qwen-1_8B-Chat'
    with_prompt = True
else:
    raise ValueError(f'{choice} not in pre maintained choices')
include_input = not with_prompt


tokenizer = AutoTokenizer.from_pretrained(dir_path, trust_remote_code=True)
tokenizer_encode_config = {'allowed_special': {"<|im_start|>", "<|im_end|>", '<|endoftext|>'}}
tokenizer_decode_config = {'skip_special_tokens': True}
end_id = [tokenizer.im_start_id, tokenizer.im_end_id] if with_prompt else tokenizer.encode("<|endoftext|>", **tokenizer_encode_config)
generation_config = {
    'end_id': end_id, 
    'mode': 'random_sample', 
    'tokenizer_config': {**tokenizer_encode_config, **tokenizer_decode_config}, 
    'max_length': 256, 
    'default_rtype': 'logits', 
    'use_states': True,
    'include_input': include_input
}

cli_demo = ChatQwenCli(dir_path, system='You are a helpful assistant.', generation_config=generation_config)

if __name__ == '__main__':
    batch = False
    gen_1toN = False

    if batch:
        # chat模型，batch_generate的示例
        res = cli_demo.chat(['你好', '你是谁'])
        print(res)
    elif gen_1toN:
        # 一条输出N跳回复
        cli_demo.generation_config['n'] = 5
        res = cli_demo.chat('你是谁？')
        print(res)
    elif with_prompt:
        # chat模型
        cli_demo.run()
    else:
        # 预训练模型
        while True:
            query = input('\n输入:')
            response = cli_demo.model.generate(query, **generation_config)
            print(f'续写: {response}')
    

#! -*- coding: utf-8 -*-
"""通义千问Qwen的测试
阿里云的通义千问: https://github.com/QwenLM/Qwen
bert4torch_config.json见readme
- [Qwen-7B](https://huggingface.co/Qwen/Qwen-7B)
- [Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)
"""
from transformers import AutoTokenizer
from bert4torch.pipelines import ChatQwenCli


model_name = 'Qwen-7B-Chat'  # Qwen-1_8B  Qwen-1_8B-Chat  Qwen-7B  Qwen-7B-Chat  Qwen-14B  Qwen-14B-Chat
model_dir = f'/data/pretrain_ckpt/Qwen/{model_name}'
with_prompt = True if 'Chat' in model_name else False


generation_config = {'max_length': 256, 'include_input': not with_prompt}
cli_demo = ChatQwenCli(model_dir, system='You are a helpful assistant.', generation_config=generation_config)


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
            response = cli_demo.generate(query)
            print(f'续写: {response}')
    

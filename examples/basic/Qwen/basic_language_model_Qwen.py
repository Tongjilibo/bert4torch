#! -*- coding: utf-8 -*-
"""通义千问Qwen的测试
阿里云的通义千问: https://github.com/QwenLM/Qwen
bert4torch_config.json见readme
- [Qwen-7B](https://huggingface.co/Qwen/Qwen-7B)
- [Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)
"""
from bert4torch.pipelines import ChatQwenCli, ChatQwenOpenaiApi
from bert4torch.pipelines import ChatOpenaiClient, ChatOpenaiClientSseclient
import re


def call_openai(stream=True):
    url = 'http://127.0.0.1:8000'
    messages = [
            {"content": "你好", "role": "user"},
            {"content": "你好，我是法律大模型", "role": "assistant"},
            {"content": "基金从业可以购买股票吗", "role": "user"}
            ]
    client = ChatOpenaiClient(url)
    if stream:
        for token in client.stream_chat(messages):
            print(token, end='', flush=True)
    else:
        print(client.chat(messages))


def call_sseclient():
    url = 'http://127.0.0.1:8000/chat/completions'
    body = {
                "messages": [
                    {"content": "你好", "role": "user"},
                    {"content": "你好，我是法律大模型", "role": "assistant"},
                    {"content": "基金从业可以购买股票吗", "role": "user"}],
                "model": "default",
                "stream": True
            }


    client = ChatOpenaiClientSseclient(url)

    # 测试打印
    client.stream_chat_cli(body)

    # 测试返回
    print('\n-------------------------------------------')
    for token in client.stream_chat(body):
        print(token, end='', flush=True)


def main():
    # Qwen-1_8B  Qwen-1_8B-Chat  Qwen-7B  Qwen-7B-Chat  Qwen-14B  Qwen-14B-Chat
    # Qwen1.5-0.5B  Qwen1.5-0.5B-Chat  Qwen1.5-1.8B  Qwen1.5-1.8B-Chat  Qwen1.5-7B  Qwen1.5-7B-Chat  Qwen1.5-14B  Qwen1.5-14B-Chat
    # Qwen2-0.5B  Qwen2-0.5B-Instruct  Qwen2-1.5B  Qwen2-1.5B-Instruct  Qwen2-7B  Qwen2-7B-Instruct
    model_dir = f'/data/pretrain_ckpt/Qwen/Qwen2-0.5B-Instruct'

    # batch: 同时infer多条query
    # gen_1toN: 为一条query同时生成N条response
    # openai: 启动一个openai的server服务 
    # cli_chat: 命令行聊天
    # cli_continue: 命令行续写
    choice = 'cli_chat'

    generation_config = {'max_length': 256, 'top_k': 1, 'include_input': False if re.search('Chat|Instruct', model_dir) else True}
    Chat =  ChatQwenOpenaiApi if choice == 'openai' else ChatQwenCli
    demo = Chat(model_dir, 
                system='You are a helpful assistant.', 
                generation_config=generation_config,
                # quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
                )

    if choice == 'batch':
        # chat模型，batch_generate的示例
        res = demo.chat(['你好', '你是谁'])
        print(res)

    elif choice == 'gen_1toN':
        # 一条输出N跳回复
        demo.generation_config['n'] = 5
        res = demo.chat('你是谁？')
        print(res)

    elif choice == 'cli_continue':
        # 命令行续写
        while True:
            query = input('\n输入:')
            response = demo.generate(query)
            print(f'续写: {response}')

    elif choice == 'cli_chat':
        # 命令行聊天
        demo.run()


if __name__ == '__main__':
    main()
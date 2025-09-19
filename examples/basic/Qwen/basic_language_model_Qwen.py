#! -*- coding: utf-8 -*-
"""通义千问Qwen的测试
阿里云的通义千问: https://github.com/QwenLM/Qwen
bert4torch_config.json见readme
- [Qwen-7B](https://huggingface.co/Qwen/Qwen-7B)
- [Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)
"""
from bert4torch.pipelines import Chat
from bert4torch.snippets.openai_client import OpenaiClient, OpenaiClientSseclient


model_name = 'default'
messages = [
        {"content": "你好", "role": "user"},
        {"content": "你好，我是法律大模型", "role": "assistant"},
        {"content": "基金从业可以购买股票吗", "role": "user"}
        ]

def call_openai(stream=True):
    url = 'http://127.0.0.1:8000'

    client = OpenaiClient(url)
    if stream:
        for token in client.stream_chat(messages, model=model_name):
            print(token, end='', flush=True)
    else:
        print(client.chat(messages, model=model_name))


def call_sseclient():
    url = 'http://127.0.0.1:8000/chat/completions'

    client = OpenaiClientSseclient(url)

    # 测试打印
    client.stream_chat_cli(messages, model=model_name)

    # 测试返回
    print('\n-------------------------------------------')
    for token in client.stream_chat(messages, model=model_name):
        print(token, end='', flush=True)


def main():
    # Qwen-1_8B	Qwen-7B	Qwen-14B
    # Qwen-1_8B-Chat	Qwen-7B-Chat	Qwen-14B-Chat
        
    # Qwen1.5-0.5B	Qwen1.5-1.8B	Qwen1.5-7B	Qwen1.5-14B
    # Qwen1.5-0.5B-Chat	Qwen1.5-1.8B-Chat	Qwen1.5-7B-Chat	Qwen1.5-14B-Chat
        
    # Qwen2-0.5B	Qwen2-1.5B	Qwen2-7B
    # Qwen2-0.5B-Instruct	Qwen2-1.5B-Instruct	Qwen2-7B-Instruct
        
    # Qwen2.5-0.5B	Qwen2.5-1.5B	Qwen2.5-3B	Qwen2.5-7B	Qwen2.5-14B
    # Qwen2.5-0.5B-Instruct	Qwen2.5-1.5B-Instruct	Qwen2.5-3B-Instruct	Qwen2.5-7B-Instruct	Qwen2.5-14B-Instruct
        
    # Qwen3-0.6B-Base	Qwen3-1.7B-Base	Qwen3-4B-Base	Qwen3-8B-Base	Qwen3-14B-Base
    # Qwen3-0.6B	Qwen3-0.6B-GPTQ-Int8	Qwen3-1.7B	Qwen3-4B    Qwen3-4B-AWQ	Qwen3-8B	Qwen3-14B	Qwen3-32B
    # Qwen3-4B-Instruct-2507	Qwen3-4B-Thinking-2507	Qwen3-30B-A3B-Instruct-2507	Qwen3-30B-A3B-Thinking-2507
    model_dir = 'E:/data/pretrain_ckpt/Qwen/Qwen3-4B-AWQ'

    # batch: 同时infer多条query
    # gen_1toN: 为一条query同时生成N条response
    # cli: 命令行聊天
    # openai: 启动一个openai的server服务
    # gradio: web demo
    # streamlit: web demo  [启动命令]: streamlit run app.py --server.address 0.0.0.0 --server.port 8001
    choice = 'cli'

    generation_config = {
        'repetition_penalty': 1.1, 
        'temperature':0.8,
        'top_k': 40,
        'top_p': 0.8,
        'max_new_tokens': 512,
        }
    demo = Chat(
        model_dir, 
        system = None,
        mode = 'cli' if choice in {'batch', 'gen_1toN'} else choice,
        generation_config = generation_config,
        # quantization_config={'quant_method': 'cpm_kernels', 'quantization_bit':8},
        # offload_when_nocall='disk',  # offload到哪里
        # offload_max_callapi_interval=30,  # 超出该时间段无调用则offload
        # offload_scheduler_interval=3,  # 检查的间隔
        # enable_thinking=False,
        # device_map='auto'
    )

    if choice == 'batch':
        # chat模型，batch_generate的示例
        res = demo.chat(['你好', '上海的天气怎么样'])
        print(res)

    elif choice == 'gen_1toN':
        # 一条输出N条回复
        demo.generation_config['n'] = 5
        res = demo.chat('你是谁？')
        print(res)

    else:
        demo.run()


if __name__ == '__main__':
    main()
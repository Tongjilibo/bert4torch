#! -*- coding: utf-8 -*-
"""文心一言的测试
"""
from bert4torch.pipelines import Chat


def main():
    model_dir = "E:/data/pretrain_ckpt/baidu/ERNIE-4.5-0.3B-PT"

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
        'do_sample': False
        }
    demo = Chat(model_dir, 
                mode = 'cli' if choice in {'batch', 'gen_1toN'} else choice,
                # system='You are a helpful assistant.', 
                generation_config=generation_config,
                # quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8},
                # offload_when_nocall='disk',  # offload到哪里
                # offload_max_callapi_interval=30,  # 超出该时间段无调用则offload
                # offload_scheduler_interval=3,  # 检查的间隔
                # enable_thinking=False
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
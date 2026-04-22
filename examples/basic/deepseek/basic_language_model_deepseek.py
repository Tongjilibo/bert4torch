
from bert4torch.pipelines import Chat

# deepseek-moe-16b-base
model_name = 'deepseek-moe-16b-chat'

# deepseek-llm-7b-base
# deepseek-llm-7b-chat

# deepseek-coder-1.3b-base
model_name = 'deepseek-coder-1.3b-instruct'
# deepseek-coder-6.7b-base
# deepseek-coder-6.7b-instruct
# deepseek-coder-7b-base-v1.5
# deepseek-coder-7b-instruct-v1.5

# DeepSeek-V2-Lite
model_name = 'DeepSeek-V2-Lite-Chat'

model_name = 'DeepSeek-R1-Distill-Qwen-1.5B'
# DeepSeek-R1-Distill-Qwen-7B
# DeepSeek-R1-Distill-Llama-8B
# DeepSeek-R1-Distill-Qwen-14B
# DeepSeek-R1-0528-Qwen3-8B

model_dir = f'/data/pretrain_ckpt/deepseek-ai/{model_name}'

generation_config = {
    'max_length': 512,
}

demo = Chat(model_dir, 
            system='You are a helpful assistant.',
            mode='cli',
            generation_config=generation_config,
            device_map='auto'
            )

if __name__ == '__main__':
    demo.run()

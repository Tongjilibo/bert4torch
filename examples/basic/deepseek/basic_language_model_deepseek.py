
from bert4torch.pipelines import Chat

# deepseek-moe-16b-base
# deepseek-moe-16b-chat

# deepseek-llm-7b-base
# deepseek-llm-7b-chat

# deepseek-coder-1.3b-base
# deepseek-coder-1.3b-instruct
# deepseek-coder-6.7b-base
# deepseek-coder-6.7b-instruct
# deepseek-coder-7b-base-v1.5
# deepseek-coder-7b-instruct-v1.5

# DeepSeek-V2-Lite
# DeepSeek-V2-Lite-Chat

# DeepSeek-R1-Distill-Qwen-1.5B
# DeepSeek-R1-Distill-Qwen-7B
# DeepSeek-R1-Distill-Llama-8B
# DeepSeek-R1-Distill-Qwen-14B
# DeepSeek-R1-0528-Qwen3-8B
model_dir = 'E:/data/pretrain_ckpt/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
generation_config = {
    'max_length': 512,
}

demo = Chat(model_dir, 
            system='You are a helpful assistant.',
            mode='cli',
            # route_api='/v1/chat/completions',
            generation_config=generation_config,
            device_map='auto',
            # quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
            )

if __name__ == '__main__':
    demo.run()



from bert4torch.pipelines import Chat

# deepseek-ai@deepseek-moe-16b-base
# deepseek-ai@deepseek-moe-16b-chat
# deepseek-ai@deepseek-llm-7b-chat

# deepseek-ai@deepseek-coder-1.3b-base
# deepseek-ai@deepseek-coder-1.3b-instruct
# deepseek-ai@deepseek-coder-6.7b-base
# deepseek-ai@deepseek-coder-6.7b-instruct
# deepseek-ai@deepseek-coder-7b-base-v1.5
# deepseek-ai@deepseek-coder-7b-instruct-v1.5

# deepseek-ai@DeepSeek-V2-Lite-Chat
# model_dir = 'E:/data/pretrain_ckpt/deepseek/deepseek-ai@deepseek-coder-1.3b-instruct'
model_dir = 'E:/data/pretrain_ckpt/deepseek/deepseek-ai@DeepSeek-V2-Lite-Chat'
generation_config = {
    'max_length': 512,
    'top_k': 1
}

demo = Chat(model_dir, 
            generation_config=generation_config,
            # device_map='auto',
            quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
            )

if __name__ == '__main__':
    demo.run()


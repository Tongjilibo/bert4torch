
from bert4torch.pipelines import Chat
import re

# deepseek-ai@deepseek-llm-7b-chat
# deepseek-ai@deepseek-coder-1.3b-instruct
model_dir = 'E:/data/pretrain_ckpt/deepseek/deepseek-ai@deepseek-coder-1.3b-instruct'
generation_config = {
    'max_length': 512
}

demo = Chat(model_dir, 
            generation_config=generation_config,
            # quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
            )

if __name__ == '__main__':
    demo.run()


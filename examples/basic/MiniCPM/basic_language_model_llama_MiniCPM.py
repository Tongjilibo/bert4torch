from bert4torch.pipelines import Chat

# MiniCPM-2B-sft-bf16
# MiniCPM-2B-dpo-bf16
# MiniCPM-1B-sft-bf16
# MiniCPM-2B-128k
model_dir = 'E:/data/pretrain_ckpt/MiniCPM/MiniCPM-1B-sft-bf16'
generation_config = {
    'max_length': 512, 
    'top_k': 40,
    'top_p': 0.8,
    'repetition_penalty': 1.1
}

demo = Chat(model_dir, 
            generation_config=generation_config,
            # quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
            )


if __name__ == '__main__':
    demo.run()

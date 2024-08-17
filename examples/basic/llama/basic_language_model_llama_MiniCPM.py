from bert4torch.pipelines import Chat


model_dir = 'E:/data/pretrain_ckpt/MiniCPM/MiniCPM-2B-sft-bf16'
generation_config = {
    'max_length': 512, 
}

demo = Chat(model_dir, 
            generation_config=generation_config,
            # quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
            )


if __name__ == '__main__':
    demo.run()

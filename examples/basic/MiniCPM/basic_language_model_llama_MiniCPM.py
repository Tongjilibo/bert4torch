from bert4torch.pipelines import Chat

# MiniCPM-1B-sft-bf16
# model_name = 'MiniCPM-2B-sft-bf16'
model_name = 'MiniCPM-2B-dpo-bf16'
# MiniCPM-2B-128k
model_dir = f'/data/pretrain_ckpt/openbmb/{model_name}'
generation_config = {
    'max_length': 512, 
    'top_k': 40,
    'top_p': 0.8,
    'repetition_penalty': 1.1
}

demo = Chat(model_dir, 
            mode='cli',
            generation_config=generation_config
            )


if __name__ == '__main__':
    demo.run()

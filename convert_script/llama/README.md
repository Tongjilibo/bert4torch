```text
[1] belle-llama模型：https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc
    模型说明： https://github.com/LianjiaTech/BELLE/tree/main/models
    LLaMA模型不允许发布调优后的完整模型权重，但是可以发布原始的模型的diff。因此需要先用脚本合并llama官方权重bell_llama的模型diff

[2] llama_vicuna模型：https://huggingface.co/AlekseyKorshuk/vicuna-7b
    模型说明： https://github.com/lm-sys/FastChat

[3]. Ziya-LLaMA-13B_v1.1: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1
[4]. Ziya-LLaMA-13B_v1: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1
[5]. Ziya-LLaMA-13B_pretrain: https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1
    1）下载llama-13b-hf权重：https://huggingface.co/decapoda-research/llama-13b-hf
    2）用项目中脚本https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/utils/apply_delta.py合并权重
        python3 -m apply_delta 
        --base E:/pretrain_ckpt/llama/13B-hf 
        --target E:/pretrain_ckpt/llama/[IDEA-CCNL]--Ziya-LLaMA-13B-v1.1 
        --delta E:/pretrain_ckpt/llama/[IDEA-CCNL]--Ziya-LLaMA-13B-v1.1-delta
    3）转换为bert4torch的适配权重

[6]. Baichuan-7B：https://huggingface.co/baichuan-inc/Baichuan-7B
[7]. Baichuan-13B：https://huggingface.co/baichuan-inc/Baichuan-13B
[8]. Baichuan-13B-Chat：https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
[9]. Baichuan2-7B-Base: https://huggingface.co/baichuan-inc/Baichuan2-7B-Base
[10]. Baichuan2-7B-Chat: https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat
[11]. Baichuan2-13B-Base: https://huggingface.co/baichuan-inc/Baichuan-13B-Base
[12]. Baichuan2-13B-Chat: https://huggingface.co/baichuan-inc/Baichuan-13B-Chat
    1) 其实baichuan-7b就是llama架构，baichuan-13b是把rope相对编码换成了alibi位置编码

[13]. Llama-2-7B: https://huggingface.co/meta-llama/Llama-2-7b-hf
[14]. Llama-2-7B-Chat: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
[15]. Llama-2-13B: https://huggingface.co/meta-llama/Llama-2-13b-hf
[16]. Llama-2-13B-Chat: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
```
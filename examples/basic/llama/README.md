## [llama](https://github.com/facebookresearch/llama)
- [huggingface](https://huggingface.co/huggyllama)
- [torrent](https://pan.baidu.com/s/1yBaYZK5LHIbJyCCbtFLW3A?pwd=phhd)

## llama2
- [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [Llama-2-7B-Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [Llama-2-13B](https://huggingface.co/meta-llama/Llama-2-13b-hf)
- [Llama-2-13B-Chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)

## [Baichuan](https://github.com/baichuan-inc)
其实baichuan-7b就是llama架构，baichuan-13b是把rope相对编码换成了alibi位置编码

- [Baichuan-7B](https://huggingface.co/baichuan-inc/Baichuan-7B)
- [Baichuan-13B](https://huggingface.co/baichuan-inc/Baichuan-13B)
- [Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)
- [Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)
- [Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)
- [Baichuan2-13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base)
- [Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)

## Ziya-LLaMA
- [Ziya-LLaMA-13B_v1.1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1)
- [Ziya-LLaMA-13B_v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1)
- [Ziya-LLaMA-13B_pretrain](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1)
1. 下载llama-13b-hf权重
2. 用项目中[脚本](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/utils/apply_delta.py)合并权重
```python
python3 -m apply_delta 
    --base /data/pretrain_ckpt/llama/13B-hf 
    --target /data/pretrain_ckpt/llama/IDEA-CCNL@Ziya-LLaMA-13B-v1.1 
    --delta /data/pretrain_ckpt/llama/IDEA-CCNL@Ziya-LLaMA-13B-v1.1-delta
```

## [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [hfl@chinese_llama_plus_7b](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
1. 用transformer脚本转换facebook的llama模型, 如直接下载的是hf版的llama则此步骤忽略；
```python
python D:/ProgramData/Anaconda3/Lib/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py  
    --input_dir /data/pretrain_ckpt/llama  
    --model_size 7B  
    --output_dir /data/pretrain_ckpt/llama/7B-hf
```
2. 用项目中脚本合并lora权重；
```python
python scripts/merge_llama_with_chinese_lora.py 
    --base_model /data/pretrain_ckpt/llama/7B-hf  
    --lora_model /data/pretrain_ckpt/llama/chinese-llama/hfl@chinese_llama_plus_lora_7b  
    --output_type huggingface
    --output_dir /data/pretrain_ckpt/llama/chinese-llama/hfl@chinese_llama_plus_7b 
```


- [hfl@chinese_alpaca_plus_7b](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
转换同上，只是合并lora权重需要合并多个lora权重
```python
python scripts/merge_llama_with_chinese_lora.py 
    --base_model /data/pretrain_ckpt/llama/7B-hf 
    --lora_model /data/pretrain_ckpt/llama/hfl@chinese_llama_plus_lora_7b,/data/pretrain_ckpt/llama/hfl@chinese_alpaca_plus_lora_7b  
    --output_type huggingface 
    --output_dir /data/pretrain_ckpt/llama/hfl@chinese_alpaca_plus_7b 
```

## [belle-llama](https://github.com/LianjiaTech/BELLE/tree/main/models)
- [BELLE-LLaMA-7B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc)
- LLaMA模型不允许发布调优后的完整模型权重，但是可以发布原始的模型的diff。因此需要先用脚本合并llama官方权重bell_llama的模型diff

## [llama_vicuna模型](https://github.com/lm-sys/FastChat)
- [vicuna-7b](https://huggingface.co/AlekseyKorshuk/vicuna-7b)
- [lmsys@vicuna-7b-v1.5](https://hf-mirror.com/lmsys/vicuna-7b-v1.5)

## [Yi](https://github.com/01-ai/Yi)
- [6B](https://huggingface.co/01-ai/Yi-6B)
- [6B-200K](https://huggingface.co/01-ai/Yi-6B-200K)
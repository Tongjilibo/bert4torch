#! -*- coding: utf-8 -*-
# 基本测试：belle-llama-7b模型的单论对话测试
# 源项目链接：https://github.com/LianjiaTech/BELLE
# LLaMA模型不允许发布调优后的完整模型权重，但是可以发布原始的模型的diff。因此需要先用脚本合并llama官方权重bell_llama的模型diff
# 模型说明： https://github.com/LianjiaTech/BELLE/tree/main/models
# belle_llama模型：https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc
# bert4torch_config.json见readme
from bert4torch.pipelines import Chat


model_dir = 'E:/data/pretrain_ckpt/llama/belle-llama-7b-2m'
generation_config = {'max_length': 512}


cli_demo = Chat(
    model_dir, 
    generation_config=generation_config,
    quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
    )


if __name__ == '__main__':
    cli_demo.run()

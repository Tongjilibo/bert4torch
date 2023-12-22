#! -*- coding: utf-8 -*-
# 基本测试：belle-llama-7b模型的单论对话测试
# 源项目链接：https://github.com/LianjiaTech/BELLE
# LLaMA模型不允许发布调优后的完整模型权重，但是可以发布原始的模型的diff。因此需要先用脚本合并llama官方权重bell_llama的模型diff
# 模型说明： https://github.com/LianjiaTech/BELLE/tree/main/models
# belle_llama模型：https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc
# bert4torch_config.json见readme



dir_path = 'E:/pretrain_ckpt/llama/belle-llama-7b-2m'
config_path = f'{dir_path}/bert4torch_config.json'
checkpoint_path = f'{dir_path}/pytorch_model.bin'


from bert4torch.chat import CliDemoBelle
generation_config = {
    'end_id': 2, 
    'mode': 'random_sample',
    'maxlen': 512, 
    'default_rtype': 'logits', 
    'use_states': True
}


cli_demo = CliDemoBelle(
    dir_path, generation_config=generation_config,
    quantization_config={'quantization_method': 'cpm_kernels', 'quantization_bit':8}
    )


if __name__ == '__main__':
    cli_demo.run()

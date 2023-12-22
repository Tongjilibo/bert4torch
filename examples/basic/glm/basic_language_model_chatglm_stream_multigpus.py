#! -*- coding: utf-8 -*-
# 基本测试：chatglm的对话测试(多卡加载)

from typing import Dict, Tuple, Union, Optional
from torch.nn import Module
from basic_language_model_chatglm import cli_demo

def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # embeddings.word_embeddings 占用1层
    # LayerNormFinal 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 embeddings.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果embeddings.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将embeddings.word_embeddings,LayerNormFinal,lm_head都放到第一张卡上
    device_map = {'embeddings.word_embeddings': 0, 'LayerNormFinal': 0, 'lm_head': 0}
    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'encoderLayer.{i}'] = gpu_target
        used += 1

    return device_map

def load_model_on_gpus(model, num_gpus: int = 2, device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        return model
    else:
        from accelerate import dispatch_model
        if device_map is None:
            device_map = auto_configure_device_map(num_gpus)
        model = dispatch_model(model, device_map=device_map)
    return model

# 多卡部署
cli_demo.model = load_model_on_gpus(cli_demo.model, num_gpus=2)

if __name__ == '__main__':
    cli_demo.run()
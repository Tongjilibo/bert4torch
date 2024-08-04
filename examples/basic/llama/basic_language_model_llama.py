#! -*- coding: utf-8 -*-
"""
基本测试: 原生llama模型的测试(需要搭配bert4torch_config.json, 见Github readme)

# llama: https://github.com/facebookresearch/llama
    权重下载：[Github](https://github.com/facebookresearch/llama)
    [huggingface](https://huggingface.co/huggyllama)
    [torrent](https://pan.baidu.com/s/1yBaYZK5LHIbJyCCbtFLW3A?pwd=phhd)

# llama2: https://huggingface.co/collections/meta-llama/llama-2-family-661da1f90a9d678b6f55773b

# llama3: https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6

# llama3.1: https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f
"""

from bert4torch.pipelines import Chat
import re

# llama-7b, llama-13b
# llama-2-7b  llama-2-7b-chat  llama-2-13b  llama-2-13b-chat
# Meta-Llama-8B-Instruct  Meta-Llama-8B-Instruct
# Meta-Llama-3.1-8B  Meta-Llama-3.1-8B-Instruct
model_dir = 'E:/data/pretrain_ckpt/llama/Meta-Llama-3.1-8B-Instruct'
generation_config = {
    'max_length': 512, 
    'include_input': False if re.search('chat|Instruct', model_dir) else True,
}

demo = Chat(model_dir, generation_config=generation_config)


if __name__ == '__main__':
    demo.run()

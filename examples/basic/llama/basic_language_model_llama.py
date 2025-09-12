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

# llama3.2: https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf
"""

from bert4torch.pipelines import Chat
import re

# llama-7b, llama-13b
# llama-2-7b  llama-2-7b-chat  llama-2-13b  llama-2-13b-chat
# Meta-Llama-3-8B  Meta-Llama-3-8B-Instruct
# Meta-Llama-3.1-8B  Meta-Llama-3.1-8B-Instruct
# Llama-3.2-1B  Llama-3.2-1B-Instruct  Llama-3.2-3B  Llama-3.2-3B-Instruct
model_dir = 'E:/data/pretrain_ckpt/meta-llama/Meta-Llama-3.1-8B-Instruct'
generation_config = {
    'max_length': 512, 
    'include_input': False if re.search('chat|Instruct', model_dir) else True
}

demo = Chat(model_dir, 
            mode='cli',
            generation_config=generation_config,
            # quantization_config={'quant_method': 'cpm_kernels', 'quantization_bit':8}
            )

functions = [{
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": "Get the current temperature at a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the temperature for, in the format \"City, Country\""
                }
            },
            "required": [
                "location"
            ]
        },
        "return": {
            "type": "number",
            "description": "The current temperature at the specified location in the specified units, as a float."
        }
    }
}]


if __name__ == '__main__':
    demo.run(
        # functions=functions  # llama3.1支持function call
        )

#! -*- coding: utf-8 -*-
# 基本测试：chatglm的对话测试, 使用内置


from bert4torch.chat import ChatCliDemo
import re


choice = 'default'  # v1.1.0, default, int4, int8
if choice == 'default':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B"
    quantization_config = {'quantization_method': 'cpm_kernels', 'quantization_bit': 8}
elif choice == 'v1.1.0':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-v1_1_0"
    quantization_config = {'quantization_method': 'cpm_kernels', 'quantization_bit': 8}
elif choice == 'int4':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-int4"
elif choice == 'int8':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-int8"
else:
    raise ValueError(f'{choice} not in pre maintained choices')

generation_config = {'mode': 'random_sample',
                     'maxlen': 2048, 
                     'default_rtype':'logits', 
                     'use_states':True}

class Demo(ChatCliDemo):
    def build_prompt(self, query, history) -> str:
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        return prompt
    
    def process_response(self, response):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response

demo = Demo(dir_path, generation_config=generation_config)
demo.run()
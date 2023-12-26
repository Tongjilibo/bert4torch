#! -*- coding: utf-8 -*-
# 基本测试：chatglm的对话测试
# bert4torch_config.json文件请访问readme

# 官方项目：https://github.com/THUDM/ChatGLM-6B
# hf链接：https://huggingface.co/THUDM/chatglm-6b
# fp16半精度下显存占用14G
# 20230406 官方项目对20000个和图像相关的进行的裁剪，因此本项目之前裁剪及tokenize的作废，使用最新的tokenize不需要进行offset


from bert4torch.pipelines import ChatGlmCli


choice = 'default'  # v1.1.0, default, int4, int8
quantization_config = None
if choice == 'default':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B"
    # quantization_config = {'quantization_method': 'cpm_kernels', 'quantization_bit': 8}
elif choice == 'v1.1.0':
    dir_path = "E:/pretrain_ckpt/glm/chatglm-6B-v1_1_0"
    # quantization_config = {'quantization_method': 'cpm_kernels', 'quantization_bit': 8}
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

cli_demo = ChatGlmCli(dir_path, generation_config=generation_config, quantization_config=quantization_config)


if __name__ == '__main__':
    cli_demo.run(stream=True)
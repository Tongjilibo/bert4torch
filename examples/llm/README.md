# 支持的大模型

- 目前支持的大模型：`chatglm | belle | llama | moss`

## 调用
- [examples/basic](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic)

## 其他
- [instruct_gpt](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/instruct_gpt): 按照三个步骤复现rlhf的实现
- [task_chatglm_nbce.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm_nbce.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, 使用朴素贝叶斯增加LLM的Context处理长度。
- [eval](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/eval): 大模型的评估eval

## 微调
- [task_chatglm_ptuning_v2.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm_ptuning_v2.py): [chatglm-6b](https://github.com/THUDM/ChatGLM-6B)的ptuning_v2微调。
- [task_chatglm_lora.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm_lora.py): [chatglm-6b](https://github.com/THUDM/ChatGLM-6B)的lora微调(基于peft)。
- [task_chatglm2_ptuning_v2.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm2_ptuning_v2.py): [chatglm2-6b](https://github.com/THUDM/ChatGLM2-6B)的ptuning_v2微调。
- [task_chatglm2_lora.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm2_lora.py): [chatglm2-6b](https://github.com/THUDM/ChatGLM2-6B)的lora微调(基于peft)。
- [task_llama-2_lora.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_llama-2_lora.py): [llama-2](https://github.com/facebookresearch/llama)的lora微调(基于peft)。
- [task_chatglm_deepspeed](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm_deepspeed): [chatglm](https://github.com/THUDM/ChatGLM-6B)的lora微调(peft+deepspeed)。
- [task_llama_deepspeed](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_llama_deepspeed): [llama-2](https://github.com/facebookresearch/llama)的lora微调(peft+deepspeed)。

### 微调指标

- [ADGEN数据集](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1)(广告生成) 

|            chatglm              |  gpu      | Time/epoch(s)|    Rouge-L    |   Rouge-1   |   Rouge-2   |   BLEU    | comment |
| ----------------------          | --------- | ------------ | ------------- | ----------- | ----------- | --------- | ------- |
| hf+pt2 official+v100-int4-bs1   |   ——      |      ——      |     24.97     |    31.12    |     7.11    |    8.10   |         |
| hf+pt2 reappear+v100-int4-bs1   |   ——      |      ——      |     24.80     |    30.97    |     6.98    |    7.85   |         |
| b4t+pt2+v100+int4+bs1           |   7G      |      ——      |     24.58     |    30.76    |     7.12    |    8.12   |         |
| b4t+pt2+v100+int4+bs1           |   7G      |      ——      |     24.98     |    31.16    |     7.17    |    8.23   |   复跑   |
| b4t+pt2+T4-int8-bs1             |  10G      |     1470     |     24.87     |    30.83    |     7.14    |    8.05   |         |
| b4t+pt2+A100(pcie 40G)-fp16-bs1 |  15G      |     287      |     25.10     |    31.43    |     7.30    |    8.28   |         |
| b4t+pt2+A100(pcie 40G)-fp16-bs8 |  22G      |     705      |     25.22     |    31.22    |     7.38    |    8.35   |         |
| b4t+pt2+A100(pcie 40G)-fp32-bs1 |  29G      |     760      |     24.83     |    30.95    |     7.18    |    8.08   |         |
| b4t+pt2+A100(pcie 40G)-fp32-bs4 |  32G      |     2600     |     25.12     |    31.55    |     7.21    |    8.02   |         |
| b4t+lora+V100-fp16-bs16         |  28G      |     2570     |     24.89     |    31.38    |     7.17    |    8.15   |         |
| b4t+qlora+V100-bs16             |  26G      |     5381     |     23.99     |    29.52    |     6.47    |    7.74   |         |


|            chatglm2             |  gpu      | Time/epoch(s)|    Rouge-L    |   Rouge-1   |   Rouge-2   |   BLEU    | comment |
| ----------------------          | --------- | ------------ | ------------- | ----------- | ----------- | --------- | ------- |
| b4t+pt2+v100+int4+bs1           |   7G      |      ——      |     24.36     |    29.97    |     6.66    |    7.89   |         |

# 支持的大模型

- 目前支持的大模型：`chatglm | belle | llama | moss`

## 调用
- [basic_language_model_llama.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_llama.py): 测试[llama](https://github.com/facebookresearch/llama)模型。
- [basic_language_model_belle.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_belle.py): 测试[belle](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M)模型。
- [basic_language_model_chatglm.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型。
- [basic_language_model_chatglm_stream.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_stream.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, stream方式输出。
- [basic_language_model_chatglm_webdemo.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_webdemo.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, web方式输出。
- [basic_language_model_chatglm_batch.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_batch.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, batch方式输出。
- [basic_language_model_moss.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_moss.py): 测试[moss](https://github.com/OpenLMLab/MOSS)模型, int8低成本部署。
- [basic_language_model_vicuna.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_vicuna.py): 测试[vicuna](https://huggingface.co/AlekseyKorshuk/vicuna-7b)模型。

## 微调
- [task_chatglm_ptuning_v2.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm_ptuning_v2.py): [chatglm-6b](https://github.com/THUDM/ChatGLM-6B)的ptuning_v2微调。
- [task_chatglm_lora.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm_lora.py): [chatglm-6b](https://github.com/THUDM/ChatGLM-6B)的lora微调(基于peft)。

### 微调指标

- [ADGEN数据集](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1)(广告生成) 

|            chatglm              |  gpu      | Time/epoch(s)|    Rouge-L    |   Rouge-1   |   Rouge-2   |   BLEU    | comment |
| ----------------------          | --------- | ------------ | ------------- | ----------- | ----------- | --------- | ------- |
| hf+pt2 official+v100            |   ——      |      ——      |     24.97     |    31.12    |     7.11    |    8.10   |         |
| hf+pt2 reappear+v100            |   ——      |      ——      |     24.80     |    30.97    |     6.98    |    7.85   |         |
| b4t+chatglm+ptuning_v2+v100     |   ——      |      ——      |     24.58     |    30.76    |     7.12    |    8.12   |         |
| b4t+pt2+T4-int8-bs1             |  10G      |     1470     |     24.87     |    30.83    |     7.14    |    8.05   |         |
| b4t+pt2+A100(pcie 40G)-fp16-bs1 |  15G      |     287      |     25.10     |    31.43    |     7.30    |    8.28   |         |
| b4t+pt2+A100(pcie 40G)-fp16-bs8 |  22G      |     705      |     25.22     |    31.22    |     7.38    |    8.35   |         |
| b4t+pt2+A100(pcie 40G)-fp32-bs1 |  29G      |     760      |     24.83     |    30.95    |     7.18    |    8.08   |         |
| b4t+pt2+A100(pcie 40G)-fp32-bs4 |  32G      |     2600     |     25.12     |    31.55    |     7.21    |    8.02   |         |
| b4t+lora+V100-fp16-bs16         |  28G      |     2570     |     24.89     |    31.38    |     7.17    |    8.15   |         |

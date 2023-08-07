# 支持的大模型

- 目前支持的大模型：`chatglm | belle | llama | moss`

## 调用
- [basic_language_model_llama.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama/basic_language_model_llama.py): 测试[llama](https://github.com/facebookresearch/llama)模型。
- [basic_language_model_llama-2.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama/basic_language_model_llama-2.py): 测试[llama-2](https://github.com/facebookresearch/llama)模型。
- [basic_language_model_llama_vicuna.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama/basic_language_model_llama_vicuna.py): 测试[vicuna](https://huggingface.co/AlekseyKorshuk/vicuna-7b)模型。
- [basic_language_model_llama_belle.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama/basic_language_model_llama_belle.py): 测试[belle](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M)模型。
- [basic_language_model_llama_chinese_llama_alpaca.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama/basic_language_model_llama_chinese_llama_alpaca.py): 测试[chinese_llama_alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)模型。
- [basic_language_model_llama_ziya.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama/basic_language_model_llama_ziya.py): 测试[ziya](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1)模型。
- [basic_language_model_llama_baichuan.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama/basic_language_model_llama_baichuan.py): 测试[baichuan](https://github.com/baichuan-inc/Baichuan-7B)模型。
- [basic_language_model_llama_baichuan_stream.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama/basic_language_model_llama_baichuan_stream.py): 测试[baichuan](https://github.com/baichuan-inc/Baichuan-7B)模型, stream形式。
- [basic_language_model_chatglm.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型。
- [basic_language_model_chatglm_api.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_api.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, api形式。
- [basic_language_model_chatglm_stream.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_stream.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, stream方式输出。
- [basic_language_model_chatglm_stream_multigpus.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_stream_multigpus.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, stream方式输出(多卡加载)。
- [basic_language_model_chatglm_webdemo.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_webdemo.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, web方式输出。
- [basic_language_model_chatglm_batch.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_batch.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, batch方式输出。
- [basic_language_model_chatglm_nbce.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_nbce.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, 使用朴素贝叶斯增加LLM的Context处理长度。
- [basic_language_model_chatglm2_stream.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm2_stream.py): 测试[chatglm2-6b](https://github.com/THUDM/ChatGLM2-6B)模型, stream方式输出。
- [basic_language_model_moss.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_moss.py): 测试[moss](https://github.com/OpenLMLab/MOSS)模型, int4和int8低成本部署。

## 微调
- [task_chatglm_ptuning_v2.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm_ptuning_v2.py): [chatglm-6b](https://github.com/THUDM/ChatGLM-6B)的ptuning_v2微调。
- [task_chatglm_lora.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm_lora.py): [chatglm-6b](https://github.com/THUDM/ChatGLM-6B)的lora微调(基于peft)。
- [task_chatglm2_ptuning_v2.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm2_ptuning_v2.py): [chatglm2-6b](https://github.com/THUDM/ChatGLM2-6B)的ptuning_v2微调。
- [task_chatglm2_lora.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm2_lora.py): [chatglm2-6b](https://github.com/THUDM/ChatGLM2-6B)的lora微调(基于peft)。
- [task_llama-2_lora.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_llama-2_lora.py): [llama-2](https://github.com/facebookresearch/llama)的lora微调(基于peft)。
- [task_chatglm_deepspeed.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm_deepspeed.py): [chatglm](https://github.com/THUDM/ChatGLM-6B)的lora微调(peft+deepspeed)。
- [task_llama_deepspeed.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_llama_deepspeed.py): [llama-2](https://github.com/facebookresearch/llama)的lora微调(peft+deepspeed)。

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

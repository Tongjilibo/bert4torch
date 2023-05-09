# 支持的大模型

- 目前支持的大模型：`chatglm | belle | llama | moss`
- 需要安装最新版的bert4toch: `pip install git+https://github.com/Tongjilibo/bert4torch`

## 调用
- [basic_language_model_llama.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_llama.py): 测试[llama](https://github.com/facebookresearch/llama)模型。
- [basic_language_model_belle.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_belle.py): 测试[belle](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M)模型。
- [basic_language_model_chatglm.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型。
- [basic_language_model_chatglm_stream.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_stream.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, stream方式输出。
- [basic_language_model_chatglm_webdemo.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_webdemo.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, web方式输出。
- [basic_language_model_chatglm_batch.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_batch.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, batch方式输出。
- [basic_language_model_moss.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_moss.py): 测试[moss](https://github.com/OpenLMLab/MOSS)模型, int8低成本部署。

## 微调
- [task_chatglm_ptuning_v2.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm_ptuning_v2.py): [chatglm-6b](https://github.com/THUDM/ChatGLM-6B)的ptuning_v2微调。
- [task_chatglm_lora.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm_lora.py): [chatglm-6b](https://github.com/THUDM/ChatGLM-6B)的lora微调(基于peft)。

### 微调指标

- [ADGEN数据集](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1)(广告生成) 

|            solution           |    Rouge-L    |   Rouge-1   |   Rouge-2   |    BLEU   | comment |
| ------------ | ---------------| ------------- | ----------- | ----------- | --------- |
| hf+chatglm+ptuning_v2官方     |     24.97     |    31.12    |     7.11    |    8.10   |         |
| hf+chatglm+ptuning_v2复现     |     24.80     |    30.97    |     6.98    |    7.85   |         |
| bert4torch+chatglm+ptuning_v2 |     24.58     |    30.76    |     7.12    |   8.12    |         |

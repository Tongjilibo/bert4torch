# 支持的大模型

chatglm | llama | moss

## 调用
- [basic_language_model_llama.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_llama.py): 测试[llama](https://github.com/facebookresearch/llama)模型。
- [basic_language_model_belle.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_belle.py): 测试[belle](https://huggingface.co/BelleGroup/BELLE-LLAMA-7B-2M)模型。
- [basic_language_model_chatglm.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型。
- [basic_language_model_chatglm_stream.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_stream.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, stream方式输出。
- [basic_language_model_chatglm_webdemo.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_chatglm_webdemo.py): 测试[chatglm-6b](https://github.com/THUDM/ChatGLM-6B)模型, web方式输出。
- [basic_language_model_moss.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/basic_language_model_moss.py): 测试[moss](https://github.com/OpenLMLab/MOSS)模型, int8低成本部署。

## 微调
- [task_chatglm_ptuning_v2.py](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm/task_chatglm_ptuning_v2.py): [chatglm-6b](https://github.com/THUDM/ChatGLM-6B)的ptuning_v2微调。

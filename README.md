![bert4torch](https://github.com/Tongjilibo/bert4torch/blob/master/docs/pics/bert4torch.png)

[![licence](https://img.shields.io/github/license/Tongjilibo/bert4torch.svg?maxAge=3600)](https://github.com/Tongjilibo/bert4torch/blob/master/LICENSE)
[![GitHub release](https://img.shields.io/github/release/Tongjilibo/bert4torch.svg?maxAge=3600)](https://github.com/Tongjilibo/bert4torch/releases)
[![PyPI](https://img.shields.io/pypi/v/bert4torch?label=pypi%20package)](https://pypi.org/project/bert4torch/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/bert4torch)](https://pypistats.org/packages/bert4torch)
[![GitHub stars](https://img.shields.io/github/stars/Tongjilibo/bert4torch?style=social)](https://github.com/Tongjilibo/bert4torch)
[![GitHub Issues](https://img.shields.io/github/issues/Tongjilibo/bert4torch.svg)](https://github.com/Tongjilibo/bert4torch/issues)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Tongjilibo/bert4torch/issues)
[![Generic badge](https://img.shields.io/badge/wechat-join-green.svg?logo=wechat)](https://github.com/Tongjilibo/bert4torch/blob/master/docs/pics/wechat_group.jpg)

[Documentation](https://bert4torch.readthedocs.io) |
[Torch4keras](https://github.com/Tongjilibo/torch4keras) |
[Examples](https://github.com/Tongjilibo/bert4torch/blob/master/examples)

## 1. 下载安装

安装稳定版

```shell
pip install bert4torch
```

安装最新版

```shell
pip install git+https://github.com/Tongjilibo/bert4torch
```

- **注意事项**：pip包的发布慢于git上的开发版本，git clone**注意引用路径**，注意权重是否需要转换
- **测试用例**：`git clone https://github.com/Tongjilibo/bert4torch`，修改example中的预训练模型文件路径和数据路径即可启动脚本
- **自行训练**：针对自己的数据，修改相应的数据处理代码块
- **开发环境**：原使用`torch==1.10`版本进行开发，现已切换到`torch2.0`开发，如其他版本遇到不适配，欢迎反馈

## 2. 功能
- **LLM模型**: 加载chatglm、llama、 baichuan、ziya、bloom等开源大模型权重进行推理和微调
- **核心功能**：加载bert、roberta、albert、xlnet、nezha、bart、RoFormer、RoFormer_V2、ELECTRA、GPT、GPT2、T5、GAU-alpha、ERNIE等预训练权重继续进行finetune、并支持在bert基础上灵活定义自己模型
- [**丰富示例**](https://github.com/Tongjilibo/bert4torch/blob/master/examples/)：包含[llm](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm)、[pretrain](https://github.com/Tongjilibo/bert4torch/blob/master/examples/pretrain)、[sentence_classfication](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication)、[sentence_embedding](https://github.com/Tongjilibo/bert4torch/tree/master/examples/sentence_embedding)、[sequence_labeling](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling)、[relation_extraction](https://github.com/Tongjilibo/bert4torch/blob/master/examples/relation_extraction)、[seq2seq](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq)、[serving](https://github.com/Tongjilibo/bert4torch/blob/master/examples/serving/)等多种解决方案
- **实验验证**：已在公开数据集实验验证，使用如下[examples数据集](https://github.com/Tongjilibo/bert4torch/blob/master/examples/DATA.md)
- **易用trick**：集成了常见的[trick](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick)，即插即用
- **其他特性**：[加载transformers库模型](https://github.com/Tongjilibo/bert4torch/blob/master//tutorials/tutorials_load_transformers_model.py)一起使用；调用方式简洁高效；有训练进度条动态展示；配合torchinfo打印参数量；默认Logger和Tensorboard简便记录训练过程；自定义fit过程，满足高阶需求
- **训练过程**：

  ```text
  2022-10-28 23:16:10 - Start Training
  2022-10-28 23:16:10 - Epoch: 1/2
  5000/5000 [==============================] - 13s 3ms/step - loss: 0.1351 - acc: 0.9601
  Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 798.09it/s] 
  test_acc: 0.98045. best_test_acc: 0.98045

  2022-10-28 23:16:27 - Epoch: 2/2
  5000/5000 [==============================] - 13s 3ms/step - loss: 0.0465 - acc: 0.9862
  Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 635.78it/s] 
  test_acc: 0.98280. best_test_acc: 0.98280

  2022-10-28 23:16:44 - Finish Training
  ```

|          功能                | bert4torch |  transformers | 备注 |
|-----------------------------|------------|----------------|--------|
|训练进度条                     | ✅         |      ✅        |进度条打印loss和定义的metrics|
|分布式训练dp/ddp               | ✅         |      ✅        |torch自带dp/ddp|
|各类callbacks                 | ✅         |      ✅        |日志/tensorboard/earlystop/wandb等|
|大模型推理，stream/batch输出    | ✅         |      ✅        |各个模型是通用的，无需单独维护脚本|
|大模型微调                     | ✅         |      ✅        |lora依赖peft库，pv2自带|
|丰富tricks                    | ✅         |      ❌        |对抗训练等tricks即插即用|
|代码简洁易懂，自定义空间大        | ✅         |      ❌        |代码复用度高, keras代码训练风格|
|仓库的维护能力/影响力/使用量/兼容性| ❌         |      ✅        |目前仓库个人维护|


## 3. 快速上手

- [Quick-Start](https://bert4torch.readthedocs.io/en/latest//Quick-Start.html)
- [快速上手教程](https://github.com/Tongjilibo/bert4torch/blob/master//tutorials/README.md)，[教程示例](https://github.com/Tongjilibo/bert4torch/blob/master//tutorials)，[实战示例](https://github.com/Tongjilibo/bert4torch/blob/master/examples)
- [bert4torch介绍(知乎)](https://zhuanlan.zhihu.com/p/486329434)，[bert4torch快速上手(知乎)](https://zhuanlan.zhihu.com/p/508890807)，[bert4torch又双叒叕更新啦(知乎)](https://zhuanlan.zhihu.com/p/560885427?)

## 4. 版本历史

|更新日期| bert4torch | torch4keras | 版本说明 |
|------| ---------------- | ----------------- |----------- |
|20231224| 0.4.3          | 0.1.7|在`chat`中增加常见chat模型, 简化大模型调用的代码逻辑|
|20231219| 0.4.2          | 0.1.7|参数`checkpoint_path`支持传入文件夹地址，增加`chat`模块用于快速发布demo/api, 支持加载`.safetensors`, `meta`的device提示报错|
|20231210| 0.4.1          | 0.1.6.post2|增加longlora, 增加test模块，适配torch4keras==0.1.6(监控fit过程，有报错则发送邮件提醒; 解决torch2.0的compile冲突问题; 修复clip_grad_norm的bug)|
|20231126| 0.4.0          | 0.1.5     |修复flash_attn的bug, stream_generate支持仅输出last_token|

[更多版本](https://github.com/Tongjilibo/bert4torch/blob/master/docs/Update.md)

## 5. 更新历史：
- **20231224**：在`chat`中增加常见chat模型, 简化大模型调用的代码逻辑
- **20231219**：参数`checkpoint_path`支持传入文件夹地址，增加`chat`模块用于快速发布demo/api, 支持加载`.safetensors`, `meta`的device提示报错
- **20231210**：增加longlora, 增加test模块，适配torch4keras==0.1.6(监控fit过程，有报错则发送邮件提醒; 解决torch2.0的compile冲突问题; 修复clip_grad_norm的bug)
- **20231126**：修复flash_attn的bug, stream_generate支持仅输出last_token

[更多历史](https://github.com/Tongjilibo/bert4torch/blob/master/docs/History.md)

## 6. 预训练权重
- 若无说明则使用权重自带的`pytorch_model.bin`和`config.json`

| 模型分类| 模型名称 | 权重来源| 权重链接 | 备注(若有)|
| ----- | ----- | ----- | ----- | ----- |
| bert| bert-base-chinese| 谷歌bert的torch版 | [torch](https://huggingface.co/bert-base-chinese) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/bert/google@bert-base-chinese/bert4torch_config.json) |
|     | chinese_L-12_H-768_A-12| 谷歌 | [github](https://github.com/google-research/bert), [tf](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) | [转换命令](https://huggingface.co/docs/transformers/v4.28.1/en/converting_tensorflow_models), [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/bert/google@chinese_L-12_H-768_A-12/bert4torch_config.json) |
|     | chinese-bert-wwm-ext| HFL | [tf/torch](https://github.com/ymcui/Chinese-BERT-wwm)，[torch](https://huggingface.co/hfl/chinese-bert-wwm-ext)| |
|     | bert-base-multilingual-cased| huggingface | [torch](https://huggingface.co/bert-base-multilingual-cased) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/bert/google@bert-base-chinese/bert4torch_config.json) |
|     | macbert | HFL| [tf/torch](https://github.com/ymcui/MacBERT)，[torch](https://huggingface.co/hfl/chinese-macbert-base) | |
|     | wobert| 追一科技| [tf](https://github.com/ZhuiyiTechnology/WoBERT)，[torch_base](https://huggingface.co/junnyu/wobert_chinese_base)，[torch_plus_base](https://huggingface.co/junnyu/wobert_chinese_plus_base) | |
|     | guwenbert| ethanyt |[torch](https://huggingface.co/ethanyt/guwenbert-base) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/bert/ethanyt@guwenbert-base/bert4torch_config.json)|
|roberta|chinese-roberta-wwm-ext | HFL | [tf/torch](https://github.com/ymcui/Chinese-BERT-wwm)，[torch](https://huggingface.co/hfl/chinese-roberta-wwm-ext) | |
|     |roberta-small/tiny| 追一科技 & UER| [tf](https://github.com/ZhuiyiTechnology/pretrained-models)，[torch](https://huggingface.co/uer) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/roberta/convert_roberta-small.py) |
|     |roberta-base-english| huggingface | [torch](https://huggingface.co/roberta-base) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/roberta/huggingface@roberta-base-english/bert4torch_config.json) |
| albert|albert| brightmart| [tf](https://github.com/brightmart/albert_zh)，[torch](https://huggingface.co/voidful)，[torch](https://github.com/lonePatient/albert_pytorch) | |
| nezha|NEZHA | 华为| [tf](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)，[torch](https://github.com/lonePatient/NeZha_Chinese_PyTorch) | |
| xlnet|chinese-xlnet | HFL | [tf/torch](https://github.com/ymcui/Chinese-XLNet) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/xlnet/hfl@chinese-xlnet-base)|
|deberta| Erlangshen-DeBERTa-v2| IDEA | [torch](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese/tree/main) | |
| electra|Chinese-ELECTRA | HFL | [tf](https://github.com/ymcui/Chinese-ELECTRA)，[torch](https://huggingface.co/hfl/chinese-electra-base-discriminator) | |
| ernie|ernie | 百度文心| [paddle](https://github.com/PaddlePaddle/ERNIE)，[torch](https://huggingface.co/nghuyong)| |
| roformer|roformer| 追一科技| [tf](https://github.com/ZhuiyiTechnology/roformer)，[torch](https://huggingface.co/junnyu/roformer_chinese_base) | |
|         |roformer_v2 | 追一科技| [tf](https://github.com/ZhuiyiTechnology/roformer-v2)，[torch](https://huggingface.co/junnyu/roformer_v2_chinese_char_base)| |
| simbert|simbert | 追一科技| [tf](https://github.com/ZhuiyiTechnology/simbert)，[torch_base](https://huggingface.co/peterchou/simbert-chinese-base/tree/main) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/simbert/convert_simbert.py) |
|        |simbert_v2/roformer-sim | 追一科技| [tf](https://github.com/ZhuiyiTechnology/roformer-sim)，[torch](https://huggingface.co/junnyu/roformer_chinese_sim_char_base)| |
| gau|GAU-alpha | 追一科技| [tf](https://github.com/ZhuiyiTechnology/GAU-alpha)| [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/gau/convert_GAU_alpha.py) |
| gpt |CDial-GPT| thu-coai| [torch](https://github.com/thu-coai/CDial-GPT) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/gpt/thu-coai@CDial-GPT-LCCC-base/bert4torch_config.json) |
| gpt2| cmp_lm(26亿)|清华 | [torch](https://github.com/TsinghuaAI/CPM-1-Generate)| [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/gpt2/cpm@cpm_lm_2.6b) |
|     | gpt2-chinese-cluecorpussmall|UER | [torch](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/gpt2/uer@gpt2-chinese-cluecorpussmall)|
|     | gpt2-ml|imcaspar | [tf](https://github.com/imcaspar/gpt2-ml)，[torch](https://github.com/ghosthamlet/gpt2-ml-torch) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/gpt2/imcaspar@gpt2-ml_15g_corpus_torch) |
| bart| bart_base_chinese|复旦fnlp| [torch](https://github.com/fastnlp/CPT), [v1.0](https://huggingface.co/fnlp/bart-base-chinese/tree/v1.0), [v2.0](https://huggingface.co/fnlp/bart-base-chinese/tree/v2.0)| [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/bart/fnlp@bart-base-chinese/bert4torch_config.json) |
| t5  | t5| UER | [torch](https://huggingface.co/uer/t5-base-chinese-cluecorpussmall)| [config_base](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/uer@t5-base-chinese-cluecorpussmall), [config_small](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/uer@t5-small-chinese-cluecorpussmall)|
|     | mt5 | 谷歌| [torch](https://huggingface.co/google/mt5-base)| [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/google@mt5_torch_base)|
|     | t5_pegasus| 追一科技| [tf](https://github.com/ZhuiyiTechnology/t5-pegasus) | [config_base](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/sushen@chinese_t5_pegasus_base_torch), [config_small](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/sushen@chinese_t5_pegasus_small_torch)|
|     | chatyuan v1&v2| clue-ai | [torch](https://github.com/clue-ai/ChatYuan) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/ClueAI@ClueAI-ChatYuan-large-v1)|
|     | PromptCLUE| clue-ai | [torch](https://github.com/clue-ai/PromptCLUE) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/ClueAI@ClueAI-ChatYuan-large-v1)|
| chatglm   |chatglm-6b | THUDM | [github](https://github.com/THUDM/ChatGLM-6B), [v0.1.0](https://huggingface.co/THUDM/chatglm-6b/tree/v0.1.0), [v1.1.0](https://huggingface.co/THUDM/chatglm-6b/tree/v1.1.0), [int8](https://huggingface.co/THUDM/chatglm-6b-int8), [int4](https://huggingface.co/THUDM/chatglm-6b-int4) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/glm) |
|       |chatglm2-6b | THUDM | [github](https://github.com/THUDM/ChatGLM2-6B), [v2](https://huggingface.co/THUDM/chatglm2-6b), [int4](https://huggingface.co/THUDM/chatglm2-6b-int4), [32k](https://huggingface.co/THUDM/chatglm2-6b-32k) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/glm) |
|       |chatglm3-6b | THUDM | [github](https://github.com/THUDM/ChatGLM3), [v3](https://huggingface.co/THUDM/chatglm3-6b), [32k](https://huggingface.co/THUDM/chatglm3-6b-32k) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/glm) |
| llama | llama | facebook| [github](https://github.com/facebookresearch/llama) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama)|
|       | llama-2 | facebook| [github](https://github.com/facebookresearch/llama), [7b](https://huggingface.co/meta-llama/Llama-2-7b-hf), [7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [13b](https://huggingface.co/meta-llama/Llama-2-13b-hf), [13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama)|
|       | chinese_llama_alpaca|HFL|[github](https://github.com/ymcui/Chinese-LLaMA-Alpaca) |[config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama)|
|       | Belle_llama| LianjiaTech| [github](https://github.com/LianjiaTech/BELLE), [7B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc) | [合成说明](https://github.com/LianjiaTech/BELLE/tree/main/models)、[config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama)|
|       | Ziya | IDEA-CCNL | [v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1), [v1.1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1), [pretrain-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama) |
|       | Baichuan | baichuan-inc | [github](https://github.com/baichuan-inc/Baichuan), [7B](https://huggingface.co/baichuan-inc/Baichuan-7B), [13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base), [13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama) |
|       | Baichuan2 | baichuan-inc | [github](https://github.com/baichuan-inc/Baichuan2), [7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base), [7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat), [13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base), [13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama) |
|       | vicuna | lmsys| [7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama)|
|       | Yi | 01-ai| [github](https://github.com/01-ai/Yi), [6B](https://huggingface.co/01-ai/Yi-6B), [6B-200K](https://huggingface.co/01-ai/Yi-6B-200K) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/llama)|
| bloom |bloom | bigscience | [bloom-560m](https://huggingface.co/bigscience/bloom-560m), [bloomz-560m](https://huggingface.co/bigscience/bloomz-560m) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/bloom) |
| Qwen  |Qwen | 阿里云 | [github](https://github.com/QwenLM/Qwen-7B), [1.8B](https://huggingface.co/Qwen/Qwen-1_8B), [1.8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat), [7B](https://huggingface.co/Qwen/Qwen-7B), [7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/Qwen) |
| InternLM|InternLM | 上海人工智能实验室 | [github](https://github.com/InternLM/InternLM), [7B-Chat](https://huggingface.co/internlm/internlm-chat-7b), [7B](https://huggingface.co/internlm/internlm-7b) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/internlm) |
| Falcon|Falcon | tiiuae | [hf](https://huggingface.co/tiiuae), [RW-1B](https://huggingface.co/tiiuae/falcon-rw-1b), [7B](https://huggingface.co/tiiuae/falcon-7b), [7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/falcon) |
| embedding| text2vec-base-chinese |shibing624| [torch](https://huggingface.co/shibing624/text2vec-base-chinese) | |
|          | m3e |moka-ai| [torch](https://huggingface.co/moka-ai) |[config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/embedding/moka-ai@m3e-base/bert4torch_config.json)|
|          | bge |BAAI| [torch](huggingface.co) |[config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/embedding/moka-ai@m3e-base/bert4torch_config.json)|

## 7. 鸣谢

- 感谢苏神实现的[bert4keras](https://github.com/bojone/bert4keras)，本实现有不少地方参考了bert4keras的源码，在此衷心感谢大佬的无私奉献;
- 其次感谢项目[bert4pytorch](https://github.com/MuQiuJun-AI/bert4pytorch)，也是在该项目的指引下给了我用pytorch来复现bert4keras的想法和思路。

## 8. 引用

```
@misc{bert4torch,
  title={bert4torch},
  author={Bo Li},
  year={2022},
  howpublished={\url{https://github.com/Tongjilibo/bert4torch}},
}
```

## 9. 其他

- Wechat & Star History Chart

<table border="0">
  <tbody>
    <tr align="center" >
      <td>
         <a href="https://github.com/Tongjilibo"><img width="200" height="250" src="./docs/pics/wechat.jpg" alt="pic"></a><br>
         <a href="https://github.com/Tongjilibo">微信号</a> 
      </td>
      <td>
         <a href="https://github.com/Tongjilibo"><img width="190" height="250" src="./docs/pics/wechat_group.jpg" alt="pic"></a><br>
         <a href="https://github.com/Tongjilibo">微信群</a> 
      </td>
      <td>
         <a href="https://star-history.com/#Tongjilibo/bert4torch&Date"><img width="400" height="250" src="https://api.star-history.com/svg?repos=Tongjilibo/bert4torch&type=Date" alt="pic"></a><br>
         <a href="https://star-history.com/#Tongjilibo/bert4torch&Date">Star History Chart</a> 
      </td>    
      </tr>
  </tbody>
</table>

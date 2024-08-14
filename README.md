![bert4torch](./docs/pics/bert4torch.png)

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
[Examples](https://github.com/Tongjilibo/bert4torch/blob/master/examples) |
[build_MiniLLM_from_scratch](https://github.com/Tongjilibo/build_MiniLLM_from_scratch) |
[bert4vector](https://github.com/Tongjilibo/bert4vector)


## 目录
- [目录](#目录)
- [1. 下载安装](#1-下载安装)
- [2. 功能](#2-功能)
- [3. 快速上手](#3-快速上手)
  - [3.1 上手教程](#31-上手教程)
  - [3.2 命令行快速部署大模型服务](#32-命令行快速部署大模型服务)
- [4. 版本和更新历史](#4-版本和更新历史)
  - [4.1 版本历史](#41-版本历史)
  - [4.2 更新历史](#42-更新历史)
- [5. 预训练权重](#5-预训练权重)
- [6. 鸣谢](#6-鸣谢)
- [7. 引用](#7-引用)
- [8. 其他](#8-其他)
  

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
- **LLM模型**: 加载chatglm、llama、 baichuan、ziya、bloom等开源大模型权重进行推理和微调，命令行一行部署大模型
- **核心功能**：加载bert、roberta、albert、xlnet、nezha、bart、RoFormer、RoFormer_V2、ELECTRA、GPT、GPT2、T5、GAU-alpha、ERNIE等预训练权重继续进行finetune、并支持在bert基础上灵活定义自己模型
- [**丰富示例**](https://github.com/Tongjilibo/bert4torch/blob/master/examples/)：包含[llm](https://github.com/Tongjilibo/bert4torch/blob/master/examples/llm)、[pretrain](https://github.com/Tongjilibo/bert4torch/blob/master/examples/pretrain)、[sentence_classfication](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication)、[sentence_embedding](https://github.com/Tongjilibo/bert4torch/tree/master/examples/sentence_embedding)、[sequence_labeling](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling)、[relation_extraction](https://github.com/Tongjilibo/bert4torch/blob/master/examples/relation_extraction)、[seq2seq](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq)、[serving](https://github.com/Tongjilibo/bert4torch/blob/master/examples/serving/)等多种解决方案
- **实验验证**：已在公开数据集实验验证，使用如下[examples数据集](https://github.com/Tongjilibo/bert4torch/blob/master/data/README.md)和[实验指标](https://github.com/Tongjilibo/bert4torch/blob/master/examples/Experiments.md)
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
|一键部署大模型                   |

## 3. 快速上手

### 3.1 上手教程
- [Quick-Start](https://bert4torch.readthedocs.io/en/latest//Quick-Start.html)
- [快速上手教程](https://github.com/Tongjilibo/bert4torch/blob/master//tutorials/README.md)，[教程示例](https://github.com/Tongjilibo/bert4torch/blob/master//tutorials)，[实战示例](https://github.com/Tongjilibo/bert4torch/blob/master/examples)
- [bert4torch介绍(知乎)](https://zhuanlan.zhihu.com/p/486329434)，[bert4torch快速上手(知乎)](https://zhuanlan.zhihu.com/p/508890807)，[bert4torch又双叒叕更新啦(知乎)](https://zhuanlan.zhihu.com/p/560885427?)

### 3.2 命令行快速部署大模型服务
- 本地 / 联网加载
```shell
# 联网下载全部文件
bert4torch-llm-server --checkpoint_path Qwen2-0.5B-Instruct

# 加载本地大模型，联网下载bert4torch_config.json
bert4torch-llm-server --checkpoint_path /data/pretrain_ckpt/Qwen/Qwen2-0.5B-Instruct --config_path Qwen/Qwen2-0.5B-Instruct

# 加载本地大模型，且bert4torch_config.json已经下载并放于同名目录下
bert4torch-llm-server --checkpoint_path /data/pretrain_ckpt/Qwen/Qwen2-0.5B-Instruct
```
- 命令行 / gradio网页 / openai_api
```shell
# 命令行
bert4torch-llm-server --checkpoint_path /data/pretrain_ckpt/Qwen/Qwen2-0.5B-Instruct --mode cli

# gradio网页
bert4torch-llm-server --checkpoint_path /data/pretrain_ckpt/Qwen/Qwen2-0.5B-Instruct --mode gradio

# openai_api
bert4torch-llm-server --checkpoint_path /data/pretrain_ckpt/Qwen/Qwen2-0.5B-Instruct --mode openai
```

## 4. 版本和更新历史
### 4.1 版本历史
|更新日期| bert4torch | torch4keras | 版本说明 |
|------| ---------------- | ----------------- |----------- |
|20240814| 0.5.3          | 0.2.6 | 【新功能】增加llama3.1/Yi1.5；自动选择从hfmirror下载；支持命令行参数`bert4torch-llm-server`|
|20240801| 0.5.2          | 0.2.5 | 【新功能】chatglm/qwen系列支持function call调用, 增加internlm2系列；【小优化】简化pipeline中chat demo的调用，generate的终止token元素允许为列表, 统一rope_scaling参数名，增加rope衍生类；【bug】修复flash_attn2的推理bug, 修复bart的tie_word_embedding的bug|
|20240619| 0.5.1          | 0.2.4 | 增加Qwen1.5, Qwen2, glm4; 增加SWA/convert_lm_logits_dtype；调整各个trainer(重点DPOTrainer), generation中segment_ids, repetition_penalty需带query, RMSNorm中转类型bug|

[更多版本](https://github.com/Tongjilibo/bert4torch/blob/master/docs/Update.md)

### 4.2 更新历史

[更多历史](https://github.com/Tongjilibo/bert4torch/blob/master/docs/History.md)

## 5. 预训练权重
- 预训练模型支持多种代码加载方式
```python
from bert4torch.models import build_transformer_model

# 1. 仅指定config_path: 从头初始化模型结构, 不加载预训练模型
model = build_transformer_model('./model/bert4torch_config.json')

# 2. 仅指定checkpoint_path: 
## 2.1 文件夹路径: 自动寻找路径下的*.bin/*.safetensors权重文件 + 需把bert4torch_config.json下载并放于该目录下
model = build_transformer_model(checkpoint_path='./model')

## 2.2 文件路径/列表: 文件路径即权重路径/列表, bert4torch_config.json会从同级目录下寻找
model = build_transformer_model(checkpoint_path='./pytorch_model.bin')

## 2.3 model_name: hf上预训练权重名称, 会自动下载hf权重以及bert4torch_config.json文件
model = build_transformer_model(checkpoint_path='bert-base-chinese')

# 3. 同时指定config_path和checkpoint_path(本地路径名或model_name排列组合): 
#    本地路径从本地加载，pretrained_model_name会联网下载
config_path = './model/bert4torch_config.json'  # 或'bert-base-chinese'
checkpoint_path = './model/pytorch_model.bin'  # 或'bert-base-chinese'
model = build_transformer_model(config_path, checkpoint_path)
```

| 模型分类| 模型名称 | 权重来源| 权重链接/checkpoint_path | config_path|
| ----- | ----- | ----- | ----- | ----- |
| bert| bert-base-chinese| google-bert | [`bert-base-chinese`](https://huggingface.co/bert-base-chinese) | [`bert-base-chinese`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bert-base-chinese)|
|     | [chinese_L-12_H-768_A-12](https://github.com/google-research/bert) | 谷歌 | [tf](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip), [`Tongjilibo/bert-chinese_L-12_H-768_A-12`](https://huggingface.co/Tongjilibo/bert-chinese_L-12_H-768_A-12) | |
|     | [chinese-bert-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm) | HFL | [`hfl/chinese-bert-wwm-ext`](https://huggingface.co/hfl/chinese-bert-wwm-ext)| [`chinese-bert-wwm-ext`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chinese-bert-wwm-ext) |
|     | bert-base-multilingual-cased| google-bert | [`bert-base-multilingual-cased`](https://huggingface.co/bert-base-multilingual-cased) | [`bert-base-multilingual-cased`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bert-base-multilingual-cased) |
|     | [MacBERT](https://github.com/ymcui/MacBERT) | HFL| [`hfl/chinese-macbert-base`](https://huggingface.co/hfl/chinese-macbert-base), [`hfl/chinese-macbert-large`](https://huggingface.co/hfl/chinese-macbert-large) |[`chinese-macbert-base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chinese-macbert-base), [`chinese-macbert-large`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chinese-macbert-large)|
|     | [WoBERT](https://github.com/ZhuiyiTechnology/WoBERT)| 追一科技|[`junnyu/wobert_chinese_base`](https://huggingface.co/junnyu/wobert_chinese_base)，[`junnyu/wobert_chinese_plus_base`](https://huggingface.co/junnyu/wobert_chinese_plus_base) |[`wobert_chinese_base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/wobert_chinese_base), [`wobert_chinese_plus_base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/wobert_chinese_plus_base)|
|roberta|[chinese-roberta-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm) | HFL | [`hfl/chinese-roberta-wwm-ext`](https://huggingface.co/hfl/chinese-roberta-wwm-ext), [`hfl/chinese-roberta-wwm-ext-large`](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)(mlm权重随机初始化) |[`chinese-roberta-wwm-ext`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chinese-roberta-wwm-ext), [`chinese-roberta-wwm-ext-large`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chinese-roberta-wwm-ext-large) |
|     |[roberta-small/tiny](https://github.com/ZhuiyiTechnology/pretrained-models)| 追一科技| [`Tongjilibo/chinese_roberta_L-4_H-312_A-12`](https://huggingface.co/Tongjilibo/chinese_roberta_L-4_H-312_A-12), [`Tongjilibo/chinese_roberta_L-6_H-384_A-12`](https://huggingface.co/Tongjilibo/chinese_roberta_L-6_H-384_A-12) | |
|     |[roberta-base](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta)| FacebookAI | [`roberta-base`](https://huggingface.co/roberta-base) | [`roberta-base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roberta-base) |
|     | [guwenbert](https://github.com/Ethan-yt/guwenbert) | ethanyt |[`ethanyt/guwenbert-base`](https://huggingface.co/ethanyt/guwenbert-base) | [`guwenbert-base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/guwenbert-base)|
| albert|[albert_zh](https://github.com/brightmart/albert_zh)| brightmart| [torch](https://github.com/lonePatient/albert_pytorch), [`voidful/albert_chinese_tiny`](https://huggingface.co/voidful/albert_chinese_tiny)，[`voidful/albert_chinese_small`](https://huggingface.co/voidful/albert_chinese_small), [`voidful/albert_chinese_base`](https://huggingface.co/voidful/albert_chinese_base), [`voidful/albert_chinese_large`](https://huggingface.co/voidful/albert_chinese_large), [`voidful/albert_chinese_xlarge`](https://huggingface.co/voidful/albert_chinese_xlarge), [`voidful/albert_chinese_xxlarge`](https://huggingface.co/voidful/albert_chinese_xxlarge) | [`albert_chinese_tiny`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/albert_chinese_tiny)，[`albert_chinese_small`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/albert_chinese_small), [`albert_chinese_base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/albert_chinese_base), [`albert_chinese_large`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/albert_chinese_large), [`albert_chinese_xlarge`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/albert_chinese_xlarge), [`albert_chinese_xxlarge`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/albert_chinese_xxlarge)|
| nezha|[NEZHA](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-PyTorch) | huawei_noah| [torch](https://github.com/lonePatient/NeZha_Chinese_PyTorch), [`sijunhe/nezha-cn-base`](https://huggingface.co/sijunhe/nezha-cn-base), [`sijunhe/nezha-cn-large`](https://huggingface.co/sijunhe/nezha-cn-large), [`sijunhe/nezha-base-wwm`](https://huggingface.co/sijunhe/nezha-base-wwm), [`sijunhe/nezha-large-wwm`](https://huggingface.co/sijunhe/nezha-large-wwm)|[`nezha-cn-base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/nezha-cn-base), [`nezha-cn-large`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/nezha-cn-large), [`nezha-base-wwm`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/nezha-base-wwm), [`nezha-large-wwm`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/nezha-large-wwm)|
|      |[nezha_gpt_dialog](https://github.com/bojone/nezha_gpt_dialog)| bojone| [`Tongjilibo/nezha_gpt_dialog`](https://huggingface.co/Tongjilibo/nezha_gpt_dialog) | |
| xlnet|[Chinese-XLNet](https://github.com/ymcui/Chinese-XLNet) | HFL | [`hfl/chinese-xlnet-base`](https://huggingface.co/hfl/chinese-xlnet-base) | [`chinese-xlnet-base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chinese-xlnet-base)|
||[tranformer_xl](https://github.com/kimiyoung/transformer-xl)|huggingface|[`transfo-xl/transfo-xl-wt103`](https://huggingface.co/transfo-xl/transfo-xl-wt103)|[`transfo-xl-wt103`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/transfo-xl-wt103)|
|deberta| [Erlangshen-DeBERTa-v2](https://github.com/IDEA-CCNL/Fengshenbang-LM) | IDEA | [`IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese`](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese), [`IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese`](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese), [`IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese`](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese) |[`Erlangshen-DeBERTa-v2-97M-Chinese`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Erlangshen-DeBERTa-v2-97M-Chinese), [`Erlangshen-DeBERTa-v2-320M-Chinese`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Erlangshen-DeBERTa-v2-320M-Chinese), [`Erlangshen-DeBERTa-v2-710M-Chinese`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Erlangshen-DeBERTa-v2-710M-Chinese) |
| electra|[Chinese-ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | HFL |[`hfl/chinese-electra-base-discriminator`](https://huggingface.co/hfl/chinese-electra-base-discriminator) |[`chinese-electra-base-discriminator`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chinese-electra-base-discriminator)|
| ernie|[ernie](https://github.com/PaddlePaddle/ERNIE) | 百度文心| [`nghuyong/ernie-1.0-base-zh`](https://huggingface.co/nghuyong/ernie-1.0-base-zh), [`nghuyong/ernie-3.0-base-zh`](https://huggingface.co/nghuyong/ernie-3.0-base-zh)| [`ernie-1.0-base-zh`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/ernie-1.0-base-zh), [`ernie-3.0-base-zh`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/ernie-3.0-base-zh)|
| roformer|[roformer](https://github.com/ZhuiyiTechnology/roformer)| 追一科技| [`junnyu/roformer_chinese_base`](https://huggingface.co/junnyu/roformer_chinese_base) |[`roformer_chinese_base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_base) |
|         |[roformer_v2](https://github.com/ZhuiyiTechnology/roformer-v2) | 追一科技| [`junnyu/roformer_v2_chinese_char_base`](https://huggingface.co/junnyu/roformer_v2_chinese_char_base)|[`roformer_v2_chinese_char_base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_v2_chinese_char_base) |
| simbert|[simbert](https://github.com/ZhuiyiTechnology/simbert) | 追一科技|[`Tongjilibo/simbert-chinese-base`](https://huggingface.co/Tongjilibo/simbert-chinese-base), [`Tongjilibo/simbert-chinese-small`](https://huggingface.co/Tongjilibo/simbert-chinese-small), [`Tongjilibo/simbert-chinese-tiny`](https://huggingface.co/Tongjilibo/simbert-chinese-tiny) | |
|        |[simbert_v2/roformer-sim](https://github.com/ZhuiyiTechnology/roformer-sim) | 追一科技|[`junnyu/roformer_chinese_sim_char_base`](https://huggingface.co/junnyu/roformer_chinese_sim_char_base)，[`junnyu/roformer_chinese_sim_char_ft_base`](https://huggingface.co/junnyu/roformer_chinese_sim_char_ft_base)，[`junnyu/roformer_chinese_sim_char_small`](https://huggingface.co/junnyu/roformer_chinese_sim_char_small)，[`junnyu/roformer_chinese_sim_char_ft_small`](https://huggingface.co/junnyu/roformer_chinese_sim_char_ft_small)|[`roformer_chinese_sim_char_base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_base), [`roformer_chinese_sim_char_ft_base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_ft_base), [`roformer_chinese_sim_char_small`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_small), [`roformer_chinese_sim_char_ft_small`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_ft_small) |
| gau|[GAU-alpha](https://github.com/ZhuiyiTechnology/GAU-alpha) | 追一科技| [`Tongjilibo/chinese_GAU-alpha-char_L-24_H-768`](https://huggingface.co/Tongjilibo/chinese_GAU-alpha-char_L-24_H-768) | |
| uie| [uie](https://github.com/universal-ie/UIE) | 百度| [torch](https://github.com/HUSTAI/uie_pytorch), [`Tongjilibo/uie-base`](https://huggingface.co/Tongjilibo/uie-base) | |
| gpt |[CDial-GPT](https://github.com/thu-coai/CDial-GPT)| thu-coai| [`thu-coai/CDial-GPT_LCCC-base`](https://huggingface.co/thu-coai/CDial-GPT_LCCC-base), [`thu-coai/CDial-GPT_LCCC-large`](https://huggingface.co/thu-coai/CDial-GPT_LCCC-large) | [`CDial-GPT_LCCC-base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/CDial-GPT_LCCC-base), [`CDial-GPT_LCCC-large`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/CDial-GPT_LCCC-large) |
|     | [cmp_lm(26亿)](https://github.com/TsinghuaAI/CPM-1-Generate)|清华 | [`TsinghuaAI/CPM-Generate`](https://huggingface.co/TsinghuaAI/CPM-Generate) | [`CPM-Generate`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/CPM-Generate) |
|     |[nezha_gen](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow)|huawei_noah|[`Tongjilibo/chinese_nezha_gpt_L-12_H-768_A-12`](https://huggingface.co/Tongjilibo/chinese_nezha_gpt_L-12_H-768_A-12)|
|     | [gpt2-chinese-cluecorpussmall](https://github.com/dbiir/UER-py/wiki/Modelzoo) |UER | [`uer/gpt2-chinese-cluecorpussmall`](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall) | [`gpt2-chinese-cluecorpussmall`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/gpt2-chinese-cluecorpussmall)|
|     | [gpt2-ml](https://github.com/imcaspar/gpt2-ml) |imcaspar | [torch](https://github.com/ghosthamlet/gpt2-ml-torch), [BaiduYun(84dh)](https://pan.baidu.com/s/16tL4Bmoh6jPy0cOND0YyeA) | [`gpt2-ml_15g_corpus`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/gpt2-ml_15g_corpus), [`gpt2-ml_30g_corpus`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/gpt2-ml_30g_corpus) |
| bart| [bart_base_chinese](https://github.com/fastnlp/CPT) |复旦fnlp| [v1.0](https://huggingface.co/fnlp/bart-base-chinese/tree/v1.0), [`fnlp/bart-base-chinese`](https://huggingface.co/fnlp/bart-base-chinese/tree/main)| [`bart-base-chinese`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bart-base-chinese), [`bart-base-chinese-v1.0`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bart-base-chinese-v1.0) |
| t5  | [t5](https://github.com/dbiir/UER-py/wiki/Modelzoo) | UER | [`uer/t5-small-chinese-cluecorpussmall`](https://huggingface.co/uer/t5-small-chinese-cluecorpussmall), [`uer/t5-base-chinese-cluecorpussmall`](https://huggingface.co/uer/t5-base-chinese-cluecorpussmall) | [`t5-base-chinese-cluecorpussmall`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/t5-base-chinese-cluecorpussmall), [`t5-small-chinese-cluecorpussmall`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/t5-small-chinese-cluecorpussmall)|
|     | mt5 | 谷歌| [`google/mt5-base`](https://huggingface.co/google/mt5-base)| [`mt5-base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/mt5-base)|
|     | [t5_pegasus](https://github.com/ZhuiyiTechnology/t5-pegasus)| 追一科技| [`Tongjilibo/chinese_t5_pegasus_small`](https://huggingface.co/Tongjilibo/chinese_t5_pegasus_small), [`Tongjilibo/chinese_t5_pegasus_base`](https://huggingface.co/Tongjilibo/chinese_t5_pegasus_base)| |
|     | [chatyuan](https://github.com/clue-ai/ChatYuan)| clue-ai |[`ClueAI/ChatYuan-large-v1`](https://huggingface.co/ClueAI/ChatYuan-large-v1), [`ClueAI/ChatYuan-large-v2`](https://huggingface.co/ClueAI/ChatYuan-large-v2)| [`ChatYuan-large-v1`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/ChatYuan-large-v1), [`ChatYuan-large-v2`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/ChatYuan-large-v2)|
|     | [PromptCLUE](https://github.com/clue-ai/PromptCLUE)| clue-ai | [`ClueAI/PromptCLUE-base`](https://huggingface.co/ClueAI/PromptCLUE-base) | [`PromptCLUE-base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/PromptCLUE-base)|
| chatglm   | [chatglm-6b](https://github.com/THUDM/ChatGLM-6B) | THUDM | [`THUDM/chatglm-6b`](https://huggingface.co/THUDM/chatglm-6b), [`THUDM/chatglm-6b-int8`](https://huggingface.co/THUDM/chatglm-6b-int8), [`THUDM/chatglm-6b-int4`](https://huggingface.co/THUDM/chatglm-6b-int4), [v0.1.0](https://huggingface.co/THUDM/chatglm-6b/tree/v0.1.0) | [`chatglm-6b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm-6b), [`chatglm-6b-int8`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm-6b-int8), [`chatglm-6b-int4`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm-6b-int4), [`chatglm-6b-v0.1.0`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm-6b-v0.1.0) |
|       |[chatglm2-6b](https://github.com/THUDM/ChatGLM2-6B) | THUDM | [`THUDM/chatglm2-6b`](https://huggingface.co/THUDM/chatglm2-6b), [`THUDM/chatglm2-6b-int4`](https://huggingface.co/THUDM/chatglm2-6b-int4), [`THUDM/chatglm2-6b-32k`](https://huggingface.co/THUDM/chatglm2-6b-32k) | [`chatglm2-6b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm2-6b), [`chatglm2-6b-int4`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm2-6b-int4), [`chatglm2-6b-32k`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm2-6b-32k) |
|       |[chatglm3-6b](https://github.com/THUDM/ChatGLM3) | THUDM | [`THUDM/chatglm3-6b`](https://huggingface.co/THUDM/chatglm3-6b), [`THUDM/chatglm3-6b-32k`](https://huggingface.co/THUDM/chatglm3-6b-32k) | [`chatglm3-6b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm3-6b), [`chatglm3-6b-32k`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm3-6b-32k) |
|       |[glm4-9b](https://github.com/THUDM/GLM-4) | THUDM | [`THUDM/glm-4-9b`](https://huggingface.co/THUDM/glm-4-9b), [`THUDM/glm-4-9b-chat`](https://huggingface.co/THUDM/glm-4-9b-chat), [`THUDM/glm-4-9b-chat-1m`](https://huggingface.co/THUDM/glm-4-9b-chat-1m) | [`glm-4-9b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/glm-4-9b), [`glm-4-9b-chat`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/glm-4-9b-chat), [`glm-4-9b-chat-1m`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/glm-4-9b-chat-1m)|
| llama | [llama](https://github.com/facebookresearch/llama) | meta|  | [`llama-7b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/llama-7b), [`llama-13b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/llama-13b)|
|       | [llama-2](https://github.com/facebookresearch/llama) | meta| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf), [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf), [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | [`Llama-2-7b-hf`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Llama-2-7b-hf), [`Llama-2-7b-chat-hf`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Llama-2-7b-chat-hf), [`Llama-2-13b-hf`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Llama-2-13b-hf), [`Llama-2-13b-chat-hf`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Llama-2-13b-chat-hf)|
|       | [llama-3](https://github.com/meta-llama/llama3) | meta| [`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B), [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | [`Meta-Llama-3-8B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Meta-Llama-3-8B), [`Meta-Llama-3-8B-Instruct`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Meta-Llama-3-8B-Instruct)|
|       | [llama-3.1](https://github.com/meta-llama/llama-models) | meta | [`meta-llama/Meta-Llama-3.1-8B`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B), [`meta-llama/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) | [`Meta-Llama-3.1-8B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Meta-Llama-3.1-8B), [`Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Meta-Llama-3.1-8B-Instruct) |
|       | [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)|HFL| |[`chinese_alpaca_plus_7b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chinese_alpaca_plus_7b), [`chinese_llama_plus_7b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chinese_llama_plus_7b)|
|       | [Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)|HFL| |待添加|
|       | [Chinese-LLaMA-Alpaca-3](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3)|HFL| |待添加|
|       | [Belle_llama](https://github.com/LianjiaTech/BELLE)| LianjiaTech|[BelleGroup/BELLE-LLaMA-7B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc) | [合成说明](https://github.com/LianjiaTech/BELLE/tree/main/models)、[`BELLE-LLaMA-7B-2M-enc`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/BELLE-LLaMA-7B-2M-enc)|
|       | [Ziya](https://github.com/IDEA-CCNL/Fengshenbang-LM) | IDEA-CCNL | [IDEA-CCNL/Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1), [IDEA-CCNL/Ziya-LLaMA-13B-v1.1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1), [IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1) | [`Ziya-LLaMA-13B-v1`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Ziya-LLaMA-13B-v1), [`Ziya-LLaMA-13B-v1.1`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Ziya-LLaMA-13B-v1.1) |
|       | [Baichuan](https://github.com/baichuan-inc/Baichuan) | baichuan-inc | [`baichuan-inc/Baichuan-7B`](https://huggingface.co/baichuan-inc/Baichuan-7B), [`baichuan-inc/Baichuan-13B-Base`](https://huggingface.co/baichuan-inc/Baichuan-13B-Base), [`baichuan-inc/Baichuan-13B-Chat`](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) | [`Baichuan-7B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Baichuan-7B), [`Baichuan-13B-Base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Baichuan-13B-Base), [`Baichuan-13B-Chat`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Baichuan-13B-Chat) |
|       | [Baichuan2](https://github.com/baichuan-inc/Baichuan2) | baichuan-inc | [`baichuan-inc/Baichuan2-7B-Base`](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base), [`baichuan-inc/Baichuan2-7B-Chat`](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat), [`baichuan-inc/Baichuan2-13B-Base`](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base), [`baichuan-inc/Baichuan2-13B-Chat`](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) | [`Baichuan2-7B-Base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Baichuan2-7B-Base), [`Baichuan2-7B-Chat`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Baichuan2-7B-Chat), [`Baichuan2-13B-Base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Baichuan2-13B-Base), [`Baichuan2-13B-Chat`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Baichuan2-13B-Chat) |
|       | [vicuna](https://github.com/lm-sys/FastChat) | lmsys| [`lmsys/vicuna-7b-v1.5`](https://huggingface.co/lmsys/vicuna-7b-v1.5) | [`vicuna-7b-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/vicuna-7b-v1.5)|
|       | [Yi](https://github.com/01-ai/Yi) | 01-ai| [`01-ai/Yi-6B`](https://huggingface.co/01-ai/Yi-6B), [`01-ai/Yi-6B-200K`](https://huggingface.co/01-ai/Yi-6B-200K), [`01-ai/Yi-9B`](https://huggingface.co/01-ai/Yi-9B), [`01-ai/Yi-9B-200K`](https://huggingface.co/01-ai/Yi-9B-200K) | [`Yi-6B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Yi-6B), [`Yi-6B-200K`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Yi-6B-200K), [`Yi-9B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Yi-9B), [`Yi-9B-200K`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Yi-9B-200K) |
|       | [Yi-1.5](https://github.com/01-ai/Yi) | 01-ai| [`01-ai/Yi-1.5-6B`](https://huggingface.co/01-ai/Yi-1.5-6B), [`01-ai/Yi-1.5-6B-Chat`](https://huggingface.co/01-ai/Yi-1.5-6B-Chat),  [`01-ai/Yi-1.5-9B`](https://huggingface.co/01-ai/Yi-1.5-9B), [`01-ai/Yi-1.5-9B-32K`](https://huggingface.co/01-ai/Yi-1.5-9B-32K), [`01-ai/Yi-1.5-9B-Chat`](https://huggingface.co/01-ai/Yi-1.5-9B-Chat), [`01-ai/Yi-1.5-9B-Chat-16K`](https://huggingface.co/01-ai/Yi-1.5-9B-Chat-16K) | [`Yi-1.5-6B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Yi-1.5-6B), [`Yi-1.5-6B-Chat`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Yi-1.5-6B-Chat),  [`Yi-1.5-9B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Yi-1.5-9B), [`Yi-1.5-9B-32K`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Yi-1.5-9B-32K), [`Yi-1.5-9B-Chat`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Yi-1.5-9B-Chat), [`Yi-1.5-9B-Chat-16K`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Yi-1.5-9B-Chat-16K) |
| bloom |[bloom](https://github.com/bigscience-workshop/xmtf) | bigscience | [`bigscience/bloom-560m`](https://huggingface.co/bigscience/bloom-560m), [`bigscience/bloomz-560m`](https://huggingface.co/bigscience/bloomz-560m) | [`bloom-560m`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bloom-560m), [`bloomz-560m`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bloomz-560m) |
| Qwen  |[Qwen](https://github.com/QwenLM/Qwen) | 阿里云 | [`Qwen/Qwen-1_8B`](https://huggingface.co/Qwen/Qwen-1_8B), [`Qwen/Qwen-1_8B-Chat`](https://huggingface.co/Qwen/Qwen-1_8B-Chat), [`Qwen/Qwen-7B`](https://huggingface.co/Qwen/Qwen-7B), [`Qwen/Qwen-7B-Chat`](https://huggingface.co/Qwen/Qwen-7B-Chat), [`Qwen/Qwen-14B`](https://huggingface.co/Qwen/Qwen-14B), [`Qwen/Qwen-14B-Chat`](https://huggingface.co/Qwen/Qwen-14B-Chat) | [`Qwen-1_8B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen-1_8B), [`Qwen-1_8B-Chat`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen-1_8B-Chat), [`Qwen-7B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen-7B), [`Qwen-7B-Chat`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen-7B-Chat), [`Qwen-14B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen-14B), [`Qwen-14B-Chat`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen-14B-Chat) |
||[Qwen1.5](https://github.com/QwenLM/Qwen1.5)|阿里云|[`Qwen/Qwen1.5-0.5B`](https://huggingface.co/Qwen/Qwen1.5-0.5B), [`Qwen/Qwen1.5-0.5B-Chat`](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat), [`Qwen/Qwen1.5-1.8B`](https://huggingface.co/Qwen/Qwen1.5-1.8B), [`Qwen/Qwen1.5-1.8B-Chat`](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat), [`Qwen/Qwen1.5-7B`](https://huggingface.co/Qwen/Qwen1.5-7B), [`Qwen/Qwen1.5-7B-Chat`](https://huggingface.co/Qwen/Qwen1.5-7B-Chat), [`Qwen/Qwen1.5-14B`](https://huggingface.co/Qwen/Qwen1.5-14B), [`Qwen/Qwen1.5-14B-Chat`](https://huggingface.co/Qwen/Qwen1.5-14B-Chat)|[`Qwen1.5-0.5B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen1.5-0.5B), [`Qwen1.5-0.5B-Chat`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen1.5-0.5B-Chat), [`Qwen1.5-1.8B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen1.5-1.8B), [`Qwen1.5-1.8B-Chat`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen1.5-1.8B-Chat), [`Qwen1.5-7B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen1.5-7B), [`Qwen1.5-7B-Chat`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen1.5-7B-Chat), [`Qwen1.5-14B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen1.5-14B), [`Qwen1.5-14B-Chat`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen1.5-14B-Chat)|
||[Qwen2](https://github.com/QwenLM/Qwen2)|阿里云|[`Qwen/Qwen2-0.5B`](https://huggingface.co/Qwen/Qwen2-0.5B), [`Qwen/Qwen2-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct), [`Qwen/Qwen2-1.5B`](https://huggingface.co/Qwen/Qwen2-1.5B), [`Qwen/Qwen2-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct), [`Qwen/Qwen2-7B`](https://huggingface.co/Qwen/Qwen2-7B), [`Qwen/Qwen2-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-7B-Instruct)|[`Qwen2-0.5B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen2-0.5B), [`Qwen2-0.5B-Instruct`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen2-0.5B-Instruct), [`Qwen2-1.5B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen2-1.5B), [`Qwen2-1.5B-Instruct`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen2-1.5B-Instruct), [`Qwen2-7B`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen2-7B), [`Qwen2-7B-Instruct`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen2-7B-Instruct)|
| InternLM|[InternLM](https://github.com/InternLM/InternLM) | 上海人工智能实验室 | [`internlm/internlm-chat-7b`](https://huggingface.co/internlm/internlm-chat-7b), [`internlm/internlm-7b`](https://huggingface.co/internlm/internlm-7b) | [`internlm-7b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/internlm-7b), [`internlm-chat-7b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/internlm-chat-7b)|
|            |[InternLM2](https://huggingface.co/collections/internlm/internlm2-65b0ce04970888799707893c) | 上海人工智能实验室 | [`internlm/internlm2-1_8b`](https://huggingface.co/internlm/internlm2-1_8b), [`internlm/internlm2-chat-1_8b`](https://huggingface.co/internlm/internlm2-chat-1_8b), [`internlm/internlm2-7b`](https://huggingface.co/internlm/internlm2-7b), [`internlm/internlm2-chat-7b`](https://huggingface.co/internlm/internlm2-chat-7b), [`internlm/internlm2-20b`](https://huggingface.co/internlm/internlm2-20b), [`internlm/internlm2-chat-20b`](https://huggingface.co/internlm/internlm2-chat-20b) |[`internlm2-1_8b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/internlm2-1_8b), [`internlm2-chat-1_8b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/internlm2-chat-1_8b), [`internlm2-7b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/internlm2-7b), [`internlm2-chat-7b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/internlm2-chat-7b)|
|            |[InternLM2.5](https://huggingface.co/collections/internlm/internlm25-66853f32717072d17581bc13) | 上海人工智能实验室 | [`internlm/internlm2_5-7b`](https://huggingface.co/internlm/internlm2_5-7b), [`internlm/internlm2_5-7b-chat`](https://huggingface.co/internlm/internlm2_5-7b-chat), [`internlm/internlm2_5-7b-chat-1m`](https://huggingface.co/internlm/internlm2_5-7b-chat-1m) | [`internlm2_5-7b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/internlm2_5-7b), [`internlm2_5-7b-chat`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/internlm2_5-7b-chat), [`internlm2_5-7b-chat-1m`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/internlm2_5-7b-chat-1m) |
| Falcon|[Falcon](https://huggingface.co/tiiuae) | tiiuae | [`tiiuae/falcon-rw-1b`](https://huggingface.co/tiiuae/falcon-rw-1b), [`tiiuae/falcon-7b`](https://huggingface.co/tiiuae/falcon-7b), [`tiiuae/falcon-7b-instruct`](https://huggingface.co/tiiuae/falcon-7b-instruct) | [`falcon-rw-1b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/falcon-rw-1b), [`falcon-7b`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/falcon-7b), [`falcon-7b-instruct`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/falcon-7b-instruct) |
|  deepseek  |[DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE)|deepseek| [`deepseek-ai/deepseek-moe-16b-base`](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base), [`deepseek-ai/deepseek-moe-16b-chat`](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat) | [`deepseek-moe-16b-base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/deepseek-moe-16b-base), [`deepseek-moe-16b-chat`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/deepseek-moe-16b-chat) |
|          |[DeepSeek-LLM](https://github.com/deepseek-ai/DeepSeek-LLM)|||待添加|
|          |[DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)|||待添加|
|          |[DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder)|||待添加|
|          |[DeepSeek-Coder-V2](https://github.com/deepseek-ai/DeepSeek-Coder-V2)|||待添加|
| embedding| [text2vec-base-chinese](https://github.com/shibing624/text2vec) |shibing624| [`shibing624/text2vec-base-chinese`](https://huggingface.co/shibing624/text2vec-base-chinese) |[`text2vec-base-chinese`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/text2vec-base-chinese) |
|          | [m3e](https://github.com/wangyuxinwhy/uniem) |moka-ai| [`moka-ai/m3e-base`](https://huggingface.co/moka-ai/m3e-base) |[`m3e-base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/m3e-base)|
|          | bge |BAAI| [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5), [`BAAI/bge-large-zh-v1.5`](https://huggingface.co/BAAI/bge-large-zh-v1.5), [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5), [`BAAI/bge-base-zh-v1.5`](https://huggingface.co/BAAI/bge-base-zh-v1.5), [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5), [`BAAI/bge-small-zh-v1.5`](https://huggingface.co/BAAI/bge-small-zh-v1.5) | [`bge-large-en-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-large-en-v1.5), [`bge-large-zh-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-large-zh-v1.5), [`bge-base-en-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-base-en-v1.5), [`bge-base-zh-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-base-zh-v1.5), [`bge-small-en-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-small-en-v1.5), [`bge-small-zh-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-small-zh-v1.5)|
|          | gte |thenlper| [`thenlper/gte-large-zh`](https://huggingface.co/thenlper/gte-large-zh), [`thenlper/gte-base-zh`](https://huggingface.co/thenlper/gte-base-zh) |[`gte-base-zh`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/gte-base-zh), [`gte-large-zh`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/gte-large-zh)|

*注：
1. `高亮格式`(如`bert-base-chinese`)的表示可直接`build_transformer_model()`联网下载
2. 国内镜像网站加速下载
   - `HF_ENDPOINT=https://hf-mirror.com python your_script.py`
   - `export HF_ENDPOINT=https://hf-mirror.com`后再执行python代码
   - 在python代码开头如下设置
    ```python
    import os
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
    ```

## 6. 鸣谢

- 感谢苏神实现的[bert4keras](https://github.com/bojone/bert4keras)，本实现有不少地方参考了bert4keras的源码，在此衷心感谢大佬的无私奉献;
- 其次感谢项目[bert4pytorch](https://github.com/MuQiuJun-AI/bert4pytorch)，也是在该项目的指引下给了我用pytorch来复现bert4keras的想法和思路。

## 7. 引用

```
@misc{bert4torch,
  title={bert4torch},
  author={Bo Li},
  year={2022},
  howpublished={\url{https://github.com/Tongjilibo/bert4torch}},
}
```

## 8. 其他

- Wechat & Star History Chart
- 微信群人数超过200个（有邀请限制），可添加个人微信拉群

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

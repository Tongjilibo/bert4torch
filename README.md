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
|20240204| 0.4.7          | 0.1.9|修改`save_pretrained`用于保存文件夹, 增加GenerateSpeed用于统计token生成速度，修复t5在use_states=True时候的错误, 修改层次编码的bug, 增加deepseek_moe模型，修复generation并发错误，优化大模型耗时|
|20240116| 0.4.6          | 0.1.8|bug修复，增加`save_pretrained`用于保存`transformer`格式的权重, 增加部分`embedding`模型|
|20240111| 0.4.5          | 0.1.7|`training`时候不生成`past_key_values`, 增加`streamlit`的example, 修复句向量`max`时的bug, `batch_generate`合并到`generate`, 修改`generation`的默认参数名(兼容过去的参数名), 多轮对话中可保留`past_key_values`, 把`attention`中的`mask`补齐逻辑移到`apply_embedding`中, 增加`uie`的`pipeline`，增加`PtuningV2Trainer`|

[更多版本](https://github.com/Tongjilibo/bert4torch/blob/master/docs/Update.md)

## 5. 更新历史：

[更多历史](https://github.com/Tongjilibo/bert4torch/blob/master/docs/History.md)

## 6. 预训练权重
- `空`表示可直接使用预训练权重的`config.json`

| 模型分类| 模型名称 | 权重来源| 权重链接 | 配置文件(若有)|
| ----- | ----- | ----- | ----- | ----- |
| bert| bert-base-chinese| 谷歌bert的torch版 | [torch](https://huggingface.co/bert-base-chinese) | [bert-base-chinese](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bert-base-chinese)|
|     | chinese_L-12_H-768_A-12| 谷歌 | [github](https://github.com/google-research/bert), [tf](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) | [转换命令](https://huggingface.co/docs/transformers/v4.28.1/en/converting_tensorflow_models), [bert@chinese_L-12_H-768_A-12](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bert@chinese_L-12_H-768_A-12) |
|     | chinese-bert-wwm-ext| HFL | [tf/torch](https://github.com/ymcui/Chinese-BERT-wwm)，[torch](https://huggingface.co/hfl/chinese-bert-wwm-ext)| 空; [hfl/chinese-bert-wwm-ext](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chinese-bert-wwm-ext) |
|     | bert-base-multilingual-cased| huggingface | [torch](https://huggingface.co/bert-base-multilingual-cased) | [bert-base-multilingual-cased](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bert-base-multilingual-cased) |
|     | macbert | HFL| [tf/torch](https://github.com/ymcui/MacBERT)，[torch](https://huggingface.co/hfl/chinese-macbert-base) |空; [hfl/chinese-macbert-base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chinese-macbert-base); [hfl/chinese-macbert-large](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chinese-macbert-large)|
|     | wobert| 追一科技| [tf](https://github.com/ZhuiyiTechnology/WoBERT)，[torch_base](https://huggingface.co/junnyu/wobert_chinese_base)，[torch_plus_base](https://huggingface.co/junnyu/wobert_chinese_plus_base) |空; [junnyu/wobert_chinese_plus_base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/wobert_chinese_plus_base); [junnyu/wobert_chinese_base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/wobert_chinese_base)|
|roberta|chinese-roberta-wwm-ext | HFL | [tf/torch](https://github.com/ymcui/Chinese-BERT-wwm)，[torch](https://huggingface.co/hfl/chinese-roberta-wwm-ext) |空; [hfl/chinese-roberta-wwm-ext](); [hfl/chinese-roberta-wwm-ext-large]() |
|     |roberta-small/tiny| 追一科技 & UER| [tf](https://github.com/ZhuiyiTechnology/pretrained-models)，[torch](https://huggingface.co/uer) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/roberta/convert_roberta-small.py) |
|     |roberta-base-english| huggingface | [torch](https://huggingface.co/roberta-base) | [roberta-base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roberta-base) |
|     | guwenbert| ethanyt |[torch](https://huggingface.co/ethanyt/guwenbert-base) | [ethanyt/guwenbert-base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/guwenbert-base)|
| albert|albert| brightmart| [tf](https://github.com/brightmart/albert_zh)，[torch](https://huggingface.co/voidful)，[torch](https://github.com/lonePatient/albert_pytorch) | |
| nezha|NEZHA | 华为| [tf](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)，[torch](https://github.com/lonePatient/NeZha_Chinese_PyTorch) | |
| xlnet|chinese-xlnet | HFL | [tf/torch](https://github.com/ymcui/Chinese-XLNet), [base](https://huggingface.co/hfl/chinese-xlnet-base) | [chinese-xlnet-base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chinese-xlnet-base)|
|deberta| Erlangshen-DeBERTa-v2| IDEA | [torch](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese/tree/main) | |
| electra|Chinese-ELECTRA | HFL | [tf](https://github.com/ymcui/Chinese-ELECTRA)，[torch](https://huggingface.co/hfl/chinese-electra-base-discriminator) | |
| ernie|ernie | 百度文心| [paddle](https://github.com/PaddlePaddle/ERNIE)，[torch](https://huggingface.co/nghuyong)| |
| roformer|roformer| 追一科技| [tf](https://github.com/ZhuiyiTechnology/roformer)，[torch](https://huggingface.co/junnyu/roformer_chinese_base) |[junnyu/roformer_chinese_base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_base) |
|         |roformer_v2 | 追一科技| [tf](https://github.com/ZhuiyiTechnology/roformer-v2)，[torch](https://huggingface.co/junnyu/roformer_v2_chinese_char_base)|[junnyu/roformer_v2_chinese_char_base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_v2_chinese_char_base) |
| simbert|simbert | 追一科技| [tf](https://github.com/ZhuiyiTechnology/simbert)，[torch_base](https://huggingface.co/peterchou/simbert-chinese-base/tree/main) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/simbert/convert_simbert.py), [peterchou/simbert-chinese-base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/simbert-chinese-base) |
|        |simbert_v2/roformer-sim | 追一科技| [tf](https://github.com/ZhuiyiTechnology/roformer-sim)，[base](https://huggingface.co/junnyu/roformer_chinese_sim_char_base)，[ft_base](https://huggingface.co/junnyu/roformer_chinese_sim_char_ft_base)，[small](https://huggingface.co/junnyu/roformer_chinese_sim_char_small)，[ft_small](https://huggingface.co/junnyu/roformer_chinese_sim_char_ft_small)|[转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/simbert/convert_roformer-sim.py), [junnyu/roformer_chinese_sim_char_base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_base), [junnyu/roformer_chinese_sim_char_ft_base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_ft_base), [junnyu/roformer_chinese_sim_char_small](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_small), [junnyu/roformer_chinese_sim_char_ft_small](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_ft_small) |
| gau|GAU-alpha | 追一科技| [tf](https://github.com/ZhuiyiTechnology/GAU-alpha)| [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/gau/convert_GAU_alpha.py) |
| uie| uie | 百度| [github](https://github.com/universal-ie/UIE), [torch](https://github.com/HUSTAI/uie_pytorch) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/convert_uie.py) |
| gpt |CDial-GPT| thu-coai| [github](https://github.com/thu-coai/CDial-GPT), [base](https://huggingface.co/thu-coai/CDial-GPT_LCCC-base), [large](https://huggingface.co/thu-coai/CDial-GPT_LCCC-large) | [thu-coai/CDial-GPT_LCCC-base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/CDial-GPT_LCCC-base), [thu-coai/CDial-GPT_LCCC-large](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/CDial-GPT_LCCC-large) |
| gpt2| cmp_lm(26亿)|清华 | [github](https://github.com/TsinghuaAI/CPM-1-Generate), [torch](https://huggingface.co/TsinghuaAI/CPM-Generate) | [TsinghuaAI/CPM-Generate](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/CPM-Generate) |
|     | gpt2-chinese-cluecorpussmall|UER | [torch](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall) | [uer/gpt2-chinese-cluecorpussmall](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/gpt2-chinese-cluecorpussmall)|
|     | gpt2-ml|imcaspar | [tf](https://github.com/imcaspar/gpt2-ml)，[github](https://github.com/ghosthamlet/gpt2-ml-torch) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/gpt2/imcaspar@gpt2-ml_15g_corpus_torch) |
| bart| bart_base_chinese|复旦fnlp| [torch](https://github.com/fastnlp/CPT), [v1.0](https://huggingface.co/fnlp/bart-base-chinese/tree/v1.0), [v2.0](https://huggingface.co/fnlp/bart-base-chinese/tree/v2.0)| [fnlp/bart-base-chinese](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bart-base-chinese) |
| t5  | t5| UER | [small](https://huggingface.co/uer/t5-small-chinese-cluecorpussmall), [base](https://huggingface.co/uer/t5-base-chinese-cluecorpussmall) | [uer/t5-base-chinese-cluecorpussmall](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/t5-base-chinese-cluecorpussmall), [uer/t5-small-chinese-cluecorpussmall](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/t5-small-chinese-cluecorpussmall)|
|     | mt5 | 谷歌| [torch](https://huggingface.co/google/mt5-base)| [google/mt5-base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/mt5-base)|
|     | t5_pegasus| 追一科技| [tf](https://github.com/ZhuiyiTechnology/t5-pegasus) | [base](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/sushen@chinese_t5_pegasus_base_torch), [small](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/t5/sushen@chinese_t5_pegasus_small_torch)|
|     | chatyuan v1&v2| clue-ai | [github](https://github.com/clue-ai/ChatYuan), [v1](https://huggingface.co/ClueAI/ChatYuan-large-v1), [v2](https://huggingface.co/ClueAI/ChatYuan-large-v2) | [ClueAI/ChatYuan-large-v1](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/ChatYuan-large-v1)|
|     | PromptCLUE| clue-ai | [github](https://github.com/clue-ai/PromptCLUE), [base](https://huggingface.co/ClueAI/PromptCLUE-base) | [ClueAI/PromptCLUE-base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/PromptCLUE-base)|
| chatglm   |chatglm-6b | THUDM | [github](https://github.com/THUDM/ChatGLM-6B), [v0.1.0](https://huggingface.co/THUDM/chatglm-6b/tree/v0.1.0), [v1.1.0](https://huggingface.co/THUDM/chatglm-6b/tree/v1.1.0), [int8](https://huggingface.co/THUDM/chatglm-6b-int8), [int4](https://huggingface.co/THUDM/chatglm-6b-int4) | [THUDM/chatglm-6b](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm-6b), [THUDM/chatglm-6b-int8](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm-6b-int8), [THUDM/chatglm-6b-int4](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm-6b-int4) |
|       |chatglm2-6b | THUDM | [github](https://github.com/THUDM/ChatGLM2-6B), [base](https://huggingface.co/THUDM/chatglm2-6b), [int4](https://huggingface.co/THUDM/chatglm2-6b-int4), [32k](https://huggingface.co/THUDM/chatglm2-6b-32k) | [THUDM/chatglm2-6b](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm2-6b), [THUDM/chatglm2-6b-int4](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm2-6b-int4), [THUDM/chatglm2-6b-32k](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm2-6b-32k) |
|       |chatglm3-6b | THUDM | [github](https://github.com/THUDM/ChatGLM3), [base](https://huggingface.co/THUDM/chatglm3-6b), [32k](https://huggingface.co/THUDM/chatglm3-6b-32k) | [THUDM/chatglm3-6b](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm3-6b), [THUDM/chatglm3-6b-32k](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chatglm3-6b-32k) |
| llama | llama | facebook| [github](https://github.com/facebookresearch/llama) | [llama-7b](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/llama-7b), [llama-13b](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/llama-13b)|
|       | llama-2 | facebook| [github](https://github.com/facebookresearch/llama), [7b](https://huggingface.co/meta-llama/Llama-2-7b-hf), [7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [13b](https://huggingface.co/meta-llama/Llama-2-13b-hf), [13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | [meta-llama/Llama-2-7b-hf](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Llama-2-7b-hf), [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Llama-2-7b-chat-hf), [meta-llama/Llama-2-13b-hf](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Llama-2-13b-hf), [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Llama-2-13b-chat-hf)|
|       | chinese_llama_alpaca|HFL|[github](https://github.com/ymcui/Chinese-LLaMA-Alpaca) |[chinese_alpaca_plus_7b](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chinese_alpaca_plus_7b), [chinese_llama_plus_7b](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/chinese_llama_plus_7b)|
|       | Belle_llama| LianjiaTech| [github](https://github.com/LianjiaTech/BELLE), [7B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc) | [合成说明](https://github.com/LianjiaTech/BELLE/tree/main/models)、[BelleGroup/BELLE-LLaMA-7B-2M-enc](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/BELLE-LLaMA-7B-2M-enc)|
|       | Ziya | IDEA-CCNL | [v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1), [v1.1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1), [pretrain-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1) | [Ziya-LLaMA-13B-v1](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Ziya-LLaMA-13B-v1), [Ziya-LLaMA-13B-v1.1](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Ziya-LLaMA-13B-v1.1) |
|       | Baichuan | baichuan-inc | [github](https://github.com/baichuan-inc/Baichuan), [7B](https://huggingface.co/baichuan-inc/Baichuan-7B), [13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base), [13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) | [baichuan-inc/Baichuan-7B](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Baichuan-7B), [baichuan-inc/Baichuan-13B-Base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Baichuan-13B-Base), [baichuan-inc/Baichuan-13B-Chat](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Baichuan-13B-Chat) |
|       | Baichuan2 | baichuan-inc | [github](https://github.com/baichuan-inc/Baichuan2), [7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base), [7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat), [13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base), [13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) | [baichuan-inc/Baichuan2-7B-Base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Baichuan2-7B-Base), [baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Baichuan2-7B-Chat), [Baichuan2-13B-Base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Baichuan2-13B-Base), [baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Baichuan2-13B-Chat) |
|       | vicuna | lmsys| [7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) | [lmsys/vicuna-7b-v1.5](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/vicuna-7b-v1.5)|
|       | Yi | 01-ai| [github](https://github.com/01-ai/Yi), [6B](https://huggingface.co/01-ai/Yi-6B), [6B-200K](https://huggingface.co/01-ai/Yi-6B-200K) | [01-ai/Yi-6B](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Yi-6B), [01-ai/Yi-6B-200K](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Yi-6B-200K)|
| bloom |bloom | bigscience | [bloom-560m](https://huggingface.co/bigscience/bloom-560m), [bloomz-560m](https://huggingface.co/bigscience/bloomz-560m) | [bigscience/bloom-560m](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bloom-560m), [bigscience/bloomz-560m](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bloomz-560m) |
| Qwen  |Qwen | 阿里云 | [github](https://github.com/QwenLM/Qwen-7B), [1.8B](https://huggingface.co/Qwen/Qwen-1_8B), [1.8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat), [7B](https://huggingface.co/Qwen/Qwen-7B), [7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat) | [Qwen/Qwen-1_8B](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen-1_8B), [Qwen/Qwen-1_8B-Chat](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen-1_8B-Chat), [Qwen/Qwen-7B](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen-7B), [Qwen/Qwen-7B-Chat](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/Qwen-7B-Chat) |
| InternLM|InternLM | 上海人工智能实验室 | [github](https://github.com/InternLM/InternLM), [7B-Chat](https://huggingface.co/internlm/internlm-chat-7b), [7B](https://huggingface.co/internlm/internlm-7b) | [internlm/internlm-7b](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/internlm-7b), [internlm/internlm-chat-7b](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/internlm-chat-7b)|
| Falcon|Falcon | tiiuae | [hf](https://huggingface.co/tiiuae), [RW-1B](https://huggingface.co/tiiuae/falcon-rw-1b), [7B](https://huggingface.co/tiiuae/falcon-7b), [7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) | [tiiuae/falcon-rw-1b](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/falcon-rw-1b), [tiiuae/falcon-7b](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/falcon-7b), [tiiuae/falcon-7b-instruct](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/falcon-7b-instruct) |
|  moe  |deeoseek-moe|deepseek| [github](https://github.com/deepseek-ai/DeepSeek-MoE), [moe-16b-base](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base), [moe-16b-chat](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat) | [deepseek-ai/deepseek-moe-16b-base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/deepseek-moe-16b-base), [deepseek-ai/deepseek-moe-16b-chat](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/deepseek-moe-16b-chat) |
| embedding| text2vec-base-chinese |shibing624| [base](https://huggingface.co/shibing624/text2vec-base-chinese) |[shibing624/text2vec-base-chinese](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/text2vec-base-chinese) |
|          | m3e |moka-ai| [base](https://huggingface.co/moka-ai/m3e-base) |[m3e-base](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/m3e-base)|
|          | bge |BAAI| [large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5), [large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) | [BAAI/bge-large-en-v1.5](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-large-en-v1.5), [BAAI/bge-large-zh-v1.5](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-large-zh-v1.5)|
|          | gte |thenlper| [large-zh](https://huggingface.co/thenlper/gte-large-zh), [base-zh](https://huggingface.co/thenlper/gte-base-zh) |[thenlper/gte-base-zh](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/gte-base-zh), [thenlper/gte-large-zh](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/gte-large-zh)|

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

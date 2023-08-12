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
- **实验验证**：已在公开数据集实验验证，使用如下[examples数据集](https://github.com/Tongjilibo/bert4torch/blob/master/examples/README.md)
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

<details><summary><b>4.1 版本历史</b></summary>

|更新日期| bert4torch版本 | torch4keras版本 | 版本说明 |
|------| ---------------- | ----------------- |----------- |
|20230812| 0.3.3          | 0.1.2     |增加大模型deepspeed的使用，增加Qwen模型(增加ntk和logn_attn)，generation的end_id支持多个token_id，修复多文件权重加载资源占用问题|
|20230804| 0.3.2          | 0.1.1     |修改依赖的torch4keras, 主要是进度条和logger, tensorboard的同步|
|20230726| 0.3.1.post2    | 0.1.0.post2     |修改baichuan的alibi逻辑，增加bloom, 简化decoder架构代码(gpt, llama, chatglm均继承decoder)|
|20230716| 0.3.0          | 0.0.9           |修改models和layers为文件夹方便扩展, 增加flash_attention参数控制，修改skip_init逻辑减少显存占用，generation增加repetition_penalty，修复chatglm的pv2的bug，generation支持transformers的tokenize，增加ziya，Baichuan|
|20230705| 0.2.9          | 0.0.8           |使用accelerate来实现skip_init精简代码, 修复add_trainer的代码提示, 增加chatglm的load_in_8bit+lora/qlora的训练, 修复grad_chechpoint, 增加chinese_llama_alpaca, torch2.0默认使用scaled_dot_product_attention加速, 增加chatglm2-6b+pv2+lora微调|
|20230518| 0.2.8          | 0.0.7.post3     |1）新增模型: 增加chatglm-6b/llama-7b/BELLE_llama/vicuna/moss/苏神、uer的roberta-small/Tiny模型以及ChatYuan v2模型/fnlp的bart2.0, 增加量化模块并适配llama，增加skip_init参数加快加载, 增加stream输出/网页demo, 增加ptuning_v2和lora; <br/>2）generation: 生成式解码新增SeqGeneration和Seq2SeqGeneration，单向decoder模型和encoder decoder模型解码增加cache, 增加batch_generate()/stream_generate功能；<br/>3）其他: 修改rope为不使用max_position，修复model.half()类型不一致问题，支持加载多个权重文件, gpt系列默认不加softmax，增加苏神Tiger的pytorch实现, 增加了对attention_key_size的入参支持，把_token_pad_ids重命名为pad_token_ids, tokenizor中重命名部分字段|
|20230310| 0.2.7.post2    | 0.0.6           |增加lion优化器, 修复albert_unshared加载权重, 修复lm系列(gpt, seq2seq)存在的forward参数不对的问题，修复GlobalPointer使用rope的bug|
|20230213| 0.2.7          | 0.0.6           |修复random_sample()的bug，适配v0.0.6的torch4keras：增加resume_from_checkpoint和save_to_checkpoint；增加add_trainer方法，重构了Trainer(BaseModel)的实现，增加了AccelerateCallback|
|20221231| 0.2.6          | 0.0.5           |build_transformer_model需显式指定add_trainer才从BaseModel继承, 增加guwenbert, macbert，text2vec-bert-chinese, wobert预训练模型，允许position_ids从padding开始, transformer.configs支持点操作，可以使用torch4keras的Trainer(net)来初始化, 修复tokenizer的切分subtoken的bug, 允许embedding_size!=hidden_size|
|20221127| 0.2.5          | 0.0.4           |对抗训练从compile转为使用Callback来实现，修复1.7.1版本兼容bug, uie模型内置|
|20221120| 0.2.4          | 0.0.3.post2     |删除SpTokenizer基类中的rematch, 增加deberta_v2模型|
|20221023| 0.2.3          | 0.0.2           |虚拟对抗VAT在多个ouput时支持指定，把Trainer抽象到[torch4keras](https://github.com/Tongjilibo/torch4keras)中，修复DP和DDP出现resume_epoch不存在的bug, tokenizer的never_split去除None, transformer_xl的bug, 增加gradient_checkpoint|
|20220922| 0.2.2         | ——            |修复t5的norm_mode问题，允许hidden_size不整除num_attention_heads，支持多个schedule(如同时ema+warmup)|
|20220905| 0.2.1         | ——            |兼容torch<=1.7.1的torch.div无rounding_mode，增加自定义metrics，支持断点续训，增加默认Logger和Tensorboard日志|
|20220823| 0.2.0         | ——            |兼容torch1.9.0的缺失take_along_dim，修复bart中位置向量514的问题，修复Sptokenizer对符号不转换，打印Epoch开始的时间戳，增加parallel_apply|
|20220808| 0.1.9         | ——            |增加mixup/manifold_mixup/temporal_ensembling策略，修复pgd策略param.grad为空的问题，修改tokenizer支持批量|
|20220717| 0.1.8         | ——            |修复原来CRF训练中loss陡增的问题，修复xlnet的token_type_ids输入显存占用大的问题|
|20220710| 0.1.7         | ——            |增加EarlyStop，CRF中自带转bool类型|
|20220605| 0.1.6         | ——            |增加transformer_xl、xlnet、t5_pegasus模型，prompt、预训练等示例，支持增加embedding输入，EMA策略，修复tokenizer和sinusoid的bug|
|20220504| 0.1.5         | ——            |增加GAU-alpha，混合梯度，梯度裁剪，单机多卡(DP、DDP)|
|20220421| 0.1.4         | ——            |增加了VAT，修复了linux下apply_embedding返回项有问题的情况|
|20220409| 0.1.3         | ——            |初始版本|


</details>

## 5. 更新历史：

<details><summary><b>5.1 即将更新</b></summary>

- [ ] chatgpt三阶段的训练

</details>

<details><summary><b>5.2 更新历史</b></summary>

- **20230812**：增加llama-2的微调, 增加大模型deepspeed的使用，增加Qwen模型(增加ntk和logn_attn)，generation的end_id支持多个token_id，修复多文件权重加载资源占用问题
- **20230726**：修改baichuan的alibi逻辑，增加bloom, 简化decoder架构代码(gpt, llama, chatglm均继承decoder)
- **20230716**：修改models和layers为文件夹方便扩展, 增加flash_attention参数控制，增加chatglm-api示例，修改skip_init逻辑减少显存占用，generation增加repetition_penalty，修复chatglm的pv2的bug，generation支持transformers的tokenize，增加ziya，Baichuan
- **20230705**：使用accelerate来实现skip_init精简代码, 修复add_trainer的代码提示, 增加chatglm的load_in_8bit+lora/qlora的训练, 修复grad_chechpoint, 增加chinese_llama_alpaca, torch2.0默认使用scaled_dot_product_attention加速, 增加chatglm2-6b+pv2+lora微调
- **20230518**：增加vicuna的集成, 增加batch_generate()功能, 把_token_pad_ids重命名为pad_token_ids, tokenizor中重命名部分字段
- **20230408**：增加苏神Tiger的pytorch实现, 集成苏神、uer的roberta-small/Tiny模型以及ChatYuan v2模型, 增加了对attention_key_size的入参支持，单向decoder模型和encoder decoder模型解码增加cache, 更新fnlp的bart2.0, 增加chatglm-6b预训练模型推理, 集成BELLE_llama模型, 增加量化模块并适配llama，增加skip_init参数加快加载, 增加stream输出/网页demo, 增加ptuning_v2，增加moss模型的int4/int8推理
- **20230326**：增加llama-7b预训练模型, 修改rope为不使用max_position, 增加prompt_clue和nezha_gpt_dialog的finetune示例(skykiseki用户)，修复model.half()类型不一致问题，生成式解码新增SeqGeneration和Seq2SeqGeneration, 支持加载多个权重文件, gpt系列默认不加softmax
- **20230310**：增加lion优化器, 修改dp和ddp示例更易用，增加PromptCLUE模型, 修复albert_unshared加载权重, 增加uer-gpt2-chinese预训练模型，修复lm系列(gpt, seq2seq)存在的forward参数不对的问题，修复GlobalPointer使用rope的bug
- **20230212**：兼容accelerate包, 增加ChatYuan v1模型，修复random_sample()的bug
- **20221230**：增加macbert，text2vec-bert-chinese, wobert模型，增加LEAR的ner示例, 增加PGRC、SPN4RE的关系提取示例，transformer.configs支持点操作，可以使用torch4keras的Trainer(net)来初始化, 修复tokenizer的切分subtoken的bug, 允许embedding_size!=hidden_size
- **20221127**：增加deberta_v2模型, 对抗训练从compile转为使用Callback来实现，修复1.7.1版本兼容bug, uie模型内置, 增加triton示例, build_transformer_model需显式指定add_trainer才从BaseModel继承, 增加guwenbert预训练模型，允许position_ids从padding开始
- **20221102**：增加CNN_Nested_NER示例, 删除SpTokenizer基类中的rematch
- **20221022**：修复DP和DDP出现resume_epoch不存在的bug, tokenizer的never_split去除None, transformer_xl的bug, 增加gradient_checkpoint
- **20221011**：虚拟对抗VAT在多个ouput时支持指定，增加elasticsearch示例, 把Trainer抽象到[torch4keras](https://github.com/Tongjilibo/torch4keras)中供更多项目使用，把梯度累积移到compile中
- **20220920**：增加TensorRT示例，支持多个schedule(如同时ema+warmup)，sanic+onnx部署
- **20220910**：增加默认Logger和Tensorboard日志，ONNX推理，增加ERNIE模型，修复t5的norm_mode问题，允许hidden_size不整除num_attention_heads
- **20220828**：增加nl2sql示例，增加自定义metrics，支持断点续训
- **20220821**：增加W2NER和DiffCSE示例，打印Epoch开始的时间戳，增加parallel_apply，兼容torch<=1.7.1的torch.div无rounding_mode
- **20220814**：增加有监督句向量、关系抽取、文本生成实验指标，兼容torch<1.9.0的缺失take_along_dim，修复bart中位置向量514的问题，修复Sptokenizer对符号不转换
- **20220727**：增加mixup/manifold_mixup/temporal_ensembling策略，修复pgd策略param.grad为空的问题，修改tokenizer支持批量，增加uie示例
- **20220716**：修复原来CRF训练中loss陡增的问题，修复xlnet的token_type_ids输入显存占用大的问题
- **20220710**：增加金融中文FAQ示例，天池新闻分类top1案例，增加EarlyStop，CRF中自带转bool类型
- **20220629**：增加ner的实验，测试crf不同初始化的效果，bert-whitening中文实验
- **20220613**：增加seq2seq+前缀树，增加SimCSE/ESimCSE/PromptBert等无监督语义相似度的中文实验
- **20220605**：增加PromptBert、PET、P-tuning示例，修改tokenizer对special_tokens分词错误的问题，增加t5_pegasus
- **20220529**：transformer_xl、xlnet模型，修改sinusoid位置向量被init_weight的bug，EMA，sohu情感分类示例
- **20220517**：增加预训练代码，支持增加embedding输入(如词性，word粒度embedding)
- **20220501**：增加了混合梯度，梯度裁剪，单机多卡训练(DP、DDP)
- **20220425**：增加了VAT、GAU-alpha等示例，增加了梯度累积，自定义fit()示例
- **20220415**：增加了ner_mrc、ner_span、roformer_v2、roformer-sim等示例
- **20220405**：增加了GPLinker、TPlinker、SimBERT等示例
- **20220329**：增加了CoSENT、R-Drop、UDA等示例
- **20220322**：添加GPT、GPT2、T5模型
- **20220312**：初版提交

</details>

## 6. 预训练权重

- 部分权重是要加载修改的[config.json](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/README.md)


| 模型分类| 权重来源| 权重链接 | 备注(若有)|
| ----- | ----- | ----- | ----- |
| bert| 谷歌原版bert(即bert-base-chinese) | [tf](https://github.com/google-research/bert)，[torch](https://huggingface.co/bert-base-chinese) | [tf转pytorch命令](https://huggingface.co/docs/transformers/converting_tensorflow_models)，[转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_bert-base-chinese.py) |
| bert| 哈工大chinese-bert-wwm-ext| [tf/torch](https://github.com/ymcui/Chinese-BERT-wwm)，[torch](https://huggingface.co/hfl/chinese-bert-wwm-ext)| |
| bert-base-multilingual-cased| huggingface | [torch](https://huggingface.co/bert-base-multilingual-cased) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_bert-base-chinese.py) |
| macbert | 哈工大chinese-macbert-base/large| [tf/torch](https://github.com/ymcui/MacBERT)，[torch](https://huggingface.co/hfl/chinese-macbert-base) | |
| roberta | 哈工大chinese-roberta-wwm-ext | [tf/torch](https://github.com/ymcui/Chinese-BERT-wwm)，[torch](https://huggingface.co/hfl/chinese-roberta-wwm-ext) | |
| roberta-small/tiny| 追一科技 & UER| [tf](https://github.com/ZhuiyiTechnology/pretrained-models)，[torch](https://huggingface.co/uer) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_roberta-small.py) |
| roberta-base (english)| huggingface | [torch](https://huggingface.co/roberta-base) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_roberta-base.py) |
| deberta_v2| IDEA Erlangshen-DeBERTa-v2| [torch](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese/tree/main) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_deberta_v2.py)|
| guwenbert | 古文bert| [torch](https://huggingface.co/ethanyt/guwenbert-base) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_guwenbert-base.py)|
| xlnet | 哈工大xlnet | [tf/torch](https://github.com/ymcui/Chinese-XLNet) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/README.md)|
| electra | 哈工大electra | [tf](https://github.com/ymcui/Chinese-ELECTRA)，[torch](https://huggingface.co/hfl/chinese-electra-base-discriminator) | |
| macbert | 哈工大macbert | [tf](https://github.com/ymcui/MacBERT)，[torch](https://huggingface.co/hfl/chinese-macbert-base) | |
| albert| brightmart| [tf](https://github.com/brightmart/albert_zh)，[torch](https://huggingface.co/voidful)，[torch](https://github.com/lonePatient/albert_pytorch) | |
| ernie | 百度文心| [paddle](https://github.com/PaddlePaddle/ERNIE)，[torch](https://huggingface.co/nghuyong)| |
| roformer| 追一科技| [tf](https://github.com/ZhuiyiTechnology/roformer)，[torch](https://huggingface.co/junnyu/roformer_chinese_base) | |
| roformer_v2 | 追一科技| [tf](https://github.com/ZhuiyiTechnology/roformer-v2)，[torch](https://huggingface.co/junnyu/roformer_v2_chinese_char_base)| |
| simbert | 追一科技| [tf](https://github.com/ZhuiyiTechnology/simbert)，[torch_base](https://huggingface.co/peterchou/simbert-chinese-base/tree/main) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_simbert.py) |
| simbert_v2/roformer-sim | 追一科技| [tf](https://github.com/ZhuiyiTechnology/roformer-sim)，[torch](https://huggingface.co/junnyu/roformer_chinese_sim_char_base)| |
| gau-alpha | 追一科技| [tf](https://github.com/ZhuiyiTechnology/GAU-alpha)| [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_GAU_alpha.py) |
| wobert| 追一科技| [tf](https://github.com/ZhuiyiTechnology/WoBERT)，[torch_base](https://huggingface.co/junnyu/wobert_chinese_base)，[torch_plus_base](https://huggingface.co/junnyu/wobert_chinese_plus_base) | |
| nezha | 华为| [tf](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)，[torch](https://github.com/lonePatient/NeZha_Chinese_PyTorch) | |
| gpt | thu-coai/CDial-GPT| [torch](https://github.com/thu-coai/CDial-GPT) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_gpt_CDial-GPT-LCCC.py) |
| gpt2| 清华26亿 cmp_lm | [torch](https://github.com/TsinghuaAI/CPM-1-Generate)| [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_gpt2_cmp_lm_2.6b.py) |
| gpt2| 中文GPT2_ML模型 | [tf](https://github.com/imcaspar/gpt2-ml)，[torch](https://github.com/ghosthamlet/gpt2-ml-torch) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_gpt2_gpt2-ml.py) |
| gpt2| UER | [torch](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_gpt2_uer-gpt2-chinese.py)|
| t5| UER | [torch](https://huggingface.co/uer/t5-base-chinese-cluecorpussmall)| [config](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/README.md)|
| mt5 | 谷歌| [torch](https://huggingface.co/google/mt5-base)| [config](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/README.md)|
| t5_pegasus| 追一科技| [tf](https://github.com/ZhuiyiTechnology/t5-pegasus) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_t5_pegasus.py)|
| bart| 复旦| [torch](https://github.com/fastnlp/CPT), [v1.0](https://huggingface.co/fnlp/bart-base-chinese/tree/v1.0), [v2.0](https://huggingface.co/fnlp/bart-base-chinese/tree/v2.0)| [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_bart_fudanNLP.py) |
| text2vec| text2vec-base-chinese | [torch](https://huggingface.co/shibing624/text2vec-base-chinese) | |
| chatyuan v1&v2| clue-ai | [torch](https://github.com/clue-ai/ChatYuan) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/README.md)|
| PromptCLUE| clue-ai | [torch](https://github.com/clue-ai/PromptCLUE) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/README.md)|
| chatglm-6b | THUDM | [github](https://github.com/THUDM/ChatGLM-6B), [v0.1.0](https://huggingface.co/THUDM/chatglm-6b/tree/v0.1.0), [v1.1.0](https://huggingface.co/THUDM/chatglm-6b/tree/v1.1.0), [int8](https://huggingface.co/THUDM/chatglm-6b-int8), [int4](https://huggingface.co/THUDM/chatglm-6b-int4) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_chatglm.py) |
| chatglm2-6b | THUDM | [github](https://github.com/THUDM/ChatGLM2-6B), [v2](https://huggingface.co/THUDM/chatglm2-6b), [int4](https://huggingface.co/THUDM/chatglm2-6b-int4) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_chatglm.py) |
| llama | facebook| [github](https://github.com/facebookresearch/llama) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_llama_pth.py)|
| llama-2 | facebook| [github](https://github.com/facebookresearch/llama), [7b](https://huggingface.co/meta-llama/Llama-2-7b-hf), [7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [13b](https://huggingface.co/meta-llama/Llama-2-13b-hf), [13b-chat](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_llama_hf.py)|
|chinese_llama_alpaca|Yiming Cui|[github](https://github.com/ymcui/Chinese-LLaMA-Alpaca) |[转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_llama_pth.py)|
| vicuna | FastChat| [torch](https://huggingface.co/AlekseyKorshuk/vicuna-7b) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_llama_hf.py)|
| Belle_llama| LianjiaTech| [github](https://github.com/LianjiaTech/BELLE), [7B-2M-enc](https://huggingface.co/BelleGroup/BELLE-LLaMA-7B-2M-enc) | [合成说明](https://github.com/LianjiaTech/BELLE/tree/main/models)、[转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_llama_hf.py)|
| Ziya | IDEA-CCNL | [v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1), [v1.1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1.1), [pretrain-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-Pretrain-v1) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_llama_pth.py) |
| Baichuan | baichuan-inc | [7B](https://github.com/baichuan-inc/Baichuan-7B), [13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base), [13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_llama_hf.py) |
| bloom | bigscience | [560m](https://huggingface.co/bigscience/bloom-560m) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_bloom.py) |
| Qwen | 阿里云 | [github](https://github.com/QwenLM/Qwen-7B), [7B](https://huggingface.co/Qwen/Qwen-7B), [7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/convert_script/convert_Qwen.py) |


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

- Wechat Discussions

<table border="0">
  <tbody>
    <tr align="center" >
      <td>
         <a href="https://github.com/Tongjilibo"><img width="200" height="250" src="./docs/pics/wechat.jpg" alt="pic"></a><br>
         <a href="https://github.com/Tongjilibo">微信号</a> 
      </td>
      <td>
         <a href="https://github.com/Tongjilibo"><img width="200" height="300" src="./docs/pics/wechat_group.jpg" alt="pic"></a><br>
         <a href="https://github.com/Tongjilibo">微信群</a> 
      </td>
    </tr>
  </tbody>
</table>

- Star History Chart

<table border="0">
  <tbody>
    <tr align="center" >
      <td>
         <a href="https://star-history.com/#Tongjilibo/bert4torch&Date"><img width="400" height="250" src="https://api.star-history.com/svg?repos=Tongjilibo/bert4torch&type=Date" alt="pic"></a><br>
         <a href="https://star-history.com/#Tongjilibo/bert4torch&Date">Star History Chart</a> 
      </td>
    </tr>
  </tbody>
</table>
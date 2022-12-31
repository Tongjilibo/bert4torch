# bert4torch
**一款用pytorch来复现bert4keras的简洁训练框架**

[![licence](https://img.shields.io/github/license/Tongjilibo/bert4torch.svg?maxAge=3600)](https://github.com/Tongjilibo/bert4torch/blob/master/LICENSE) 
[![GitHub release](https://img.shields.io/github/release/Tongjilibo/bert4torch.svg?maxAge=3600)](https://github.com/Tongjilibo/bert4torch/releases) 
[![PyPI](https://img.shields.io/pypi/v/bert4torch?label=pypi%20package)](https://pypi.org/project/bert4torch/) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/bert4torch)](https://pypistats.org/packages/bert4torch)
[![GitHub stars](https://img.shields.io/github/stars/Tongjilibo/bert4torch?style=social)](https://github.com/Tongjilibo/bert4torch)
[![GitHub Issues](https://img.shields.io/github/issues/Tongjilibo/bert4torch.svg)](https://github.com/Tongjilibo/bert4torch/issues)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Tongjilibo/bert4torch/issues)

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
pip install git+https://www.github.com/Tongjilibo/bert4torch.git
```
- **注意事项**：pip包的发布慢于git上的开发版本，git clone**注意引用路径**，注意权重是否需要转换
- **测试用例**：`git clone https://github.com/Tongjilibo/bert4torch`，修改example中的预训练模型文件路径和数据路径即可启动脚本
- **自行训练**：针对自己的数据，修改相应的数据处理代码块
- **开发环境**：使用`torch==1.10`版本进行开发，如其他版本遇到不适配，欢迎反馈

## 2. 功能
- **核心功能**：加载bert、roberta、albert、xlnet、nezha、bart、RoFormer、RoFormer_V2、ELECTRA、GPT、GPT2、T5、GAU-alpha、ERNIE等预训练权重继续进行finetune、并支持在bert基础上灵活定义自己模型
- [**丰富示例**](https://github.com/Tongjilibo/bert4torch/blob/master/examples/)：包含[pretrain](https://github.com/Tongjilibo/bert4torch/blob/master/examples/pretrain)、[sentence_classfication](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication)、[sentence_embedding](https://github.com/Tongjilibo/bert4torch/tree/master/examples/sentence_embedding)、[sequence_labeling](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling)、[relation_extraction](https://github.com/Tongjilibo/bert4torch/blob/master/examples/relation_extraction)、[seq2seq](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq)、[serving](https://github.com/Tongjilibo/bert4torch/blob/master/examples/serving/)等多种解决方案
- **实验验证**：已在公开数据集实验验证，使用如下[examples数据集](https://github.com/Tongjilibo/bert4torch/blob/master/examples/README.md)
- **易用trick**：集成了常见的[trick](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick)，即插即用
- **其他特性**：[加载transformers库模型](https://github.com/Tongjilibo/bert4torch/blob/master/examples/tutorials/tutorials_load_transformers_model.py)一起使用；调用方式简洁高效；有训练进度条动态展示；配合torchinfo打印参数量；默认Logger和Tensorboard简便记录训练过程；自定义fit过程，满足高阶需求
- **训练过程**：

    ```text
    2022-10-28 23:16:10 - Start Training
    2022-10-28 23:16:10 - Epoch: 1/5
    5000/5000 [==============================] - 13s 3ms/step - loss: 0.1351 - acc: 0.9601
    Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 798.09it/s] 
    test_acc: 0.98045. best_test_acc: 0.98045

    2022-10-28 23:16:27 - Epoch: 2/5
    5000/5000 [==============================] - 13s 3ms/step - loss: 0.0465 - acc: 0.9862
    Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 635.78it/s] 
    test_acc: 0.98280. best_test_acc: 0.98280

    2022-10-28 23:16:44 - Epoch: 3/5
    5000/5000 [==============================] - 15s 3ms/step - loss: 0.0284 - acc: 0.9915
    Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 673.60it/s] 
    test_acc: 0.98365. best_test_acc: 0.98365

    2022-10-28 23:17:03 - Epoch: 4/5
    5000/5000 [==============================] - 15s 3ms/step - loss: 0.0179 - acc: 0.9948
    Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 692.34it/s] 
    test_acc: 0.98265. best_test_acc: 0.98365

    2022-10-28 23:17:21 - Epoch: 5/5
    5000/5000 [==============================] - 14s 3ms/step - loss: 0.0129 - acc: 0.9958
    Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 701.77it/s] 
    test_acc: 0.98585. best_test_acc: 0.98585

    2022-10-28 23:17:37 - Finish Training
    ```

## 3. 快速上手
- [Quick-Start](https://bert4torch.readthedocs.io/en/latest//Quick-Start.html)
- [快速上手教程](https://github.com/Tongjilibo/bert4torch/blob/master/examples/tutorials/Tutorials.md)，[教程示例](https://github.com/Tongjilibo/bert4torch/blob/master/examples/tutorials)，[实战示例](https://github.com/Tongjilibo/bert4torch/blob/master/examples)
- [bert4torch介绍(知乎)](https://zhuanlan.zhihu.com/p/486329434)，[bert4torch快速上手(知乎)](https://zhuanlan.zhihu.com/p/508890807)，[bert4torch又双叒叕更新啦(知乎)](https://zhuanlan.zhihu.com/p/560885427?)

## 4. 版本说明
- **v0.2.6**：20221231 build_transformer_model需显式指定add_trainer才从BaseModel继承, 增加guwenbert, macbert，text2vec-bert-chinese, wobert预训练模型，允许position_ids从padding开始, transformer.configs支持点操作，可以使用torch4keras的Trainer(net)来初始化, 修复tokenizer的切分subtoken的bug, 允许embedding_size!=hidden_size
- **v0.2.5**：20221127 对抗训练从compile转为使用Callback来实现，修复1.7.1版本兼容bug, uie模型内置
- **v0.2.4**：20221120 删除SpTokenizer基类中的rematch, 增加deberta_v2模型
- **v0.2.3**：20221023 虚拟对抗VAT在多个ouput时支持指定，把Trainer抽象到[torch4keras](https://github.com/Tongjilibo/torch4keras)中，修复DP和DDP出现resume_epoch不存在的bug, tokenizer的never_split去除None, transformer_xl的bug, 增加gradient_checkpoint
- **v0.2.2**：20220922 修复t5的norm_mode问题，允许hidden_size不整除num_attention_heads，支持多个schedule(如同时ema+warmup)
- **v0.2.1**：20220905 兼容torch<=1.7.1的torch.div无rounding_mode，增加自定义metrics，支持断点续训，增加默认Logger和Tensorboard日志
- **v0.2.0**：20220823 兼容torch<1.9.0的缺失take_along_dim，修复bart中位置向量514的问题，修复Sptokenizer对符号不转换，打印Epoch开始的时间戳，增加parallel_apply
- **v0.1.9**：20220808 增加mixup/manifold_mixup/temporal_ensembling策略，修复pgd策略param.grad为空的问题，修改tokenizer支持批量
- **v0.1.8**：20220717 修复原来CRF训练中loss陡增的问题，修复xlnet的token_type_ids输入显存占用大的问题
- **v0.1.7**：20220710 增加EarlyStop，CRF中自带转bool类型
- **v0.1.6**：20220605 增加transformer_xl、xlnet、t5_pegasus模型，prompt、预训练等示例，支持增加embedding输入，EMA策略，修复tokenizer和sinusoid的bug
- **v0.1.5**：20220504 增加GAU-alpha，混合梯度，梯度裁剪，单机多卡(DP、DDP)
- **v0.1.4**：20220421 增加了VAT，修复了linux下apply_embedding返回项有问题的情况
- **v0.1.3**：20220409 初始版本

## 5. 更新：
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


## 6. 预训练权重
- 部分权重是要加载修改的[config.json](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/PLM_config.md)

| 模型分类 |  权重来源 | 权重链接 | 备注(若有) | 
|  ----  |  ----  | ----  | ----  |
|  bert  | 谷歌原版bert(即bert-base-chinese) | [tf](https://github.com/google-research/bert)，[torch](https://huggingface.co/bert-base-chinese) | [tf转pytorch命令](https://huggingface.co/docs/transformers/converting_tensorflow_models)，[转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_bert-base-chinese.py)
|  bert  | 哈工大chinese-bert-wwm-ext | [tf/torch](https://github.com/ymcui/Chinese-BERT-wwm)，[torch](https://huggingface.co/hfl/chinese-bert-wwm-ext) |
|  macbert  | 哈工大chinese-macbert-base/large | [tf/torch](https://github.com/ymcui/MacBERT)，[torch](https://huggingface.co/hfl/chinese-macbert-base) |
| robert | 哈工大chinese-robert-wwm-ext | [tf/torch](https://github.com/ymcui/Chinese-BERT-wwm)，[torch](https://huggingface.co/hfl/chinese-roberta-wwm-ext)
| deberta_v2| IDEA Erlangshen-DeBERTa-v2 | [torch](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese/tree/main) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_deberta_v2.py) |
| guwenbert | 古文bert | [torch](https://huggingface.co/ethanyt/guwenbert-base)|[转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_guwenbert-base.py)|
| xlnet | 哈工大xlnet | [tf/torch](https://github.com/ymcui/Chinese-XLNet) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/PLM_config.md)
| electra | 哈工大electra | [tf](https://github.com/ymcui/Chinese-ELECTRA)，[torch](https://huggingface.co/hfl/chinese-electra-base-discriminator)
| macbert | 哈工大macbert | [tf](https://github.com/ymcui/MacBERT)，[torch](https://huggingface.co/hfl/chinese-macbert-base)
| albert | brightmart | [tf](https://github.com/brightmart/albert_zh)，[torch](https://github.com/lonePatient/albert_pytorch)
| ernie | 百度文心 |[paddle](https://github.com/PaddlePaddle/ERNIE)，[torch](https://huggingface.co/nghuyong) | 
| roformer | 追一科技 | [tf](https://github.com/ZhuiyiTechnology/roformer)，[torch](https://huggingface.co/junnyu/roformer_chinese_base) |  
| roformer_v2 | 追一科技 | [tf](https://github.com/ZhuiyiTechnology/roformer-v2)，[torch](https://huggingface.co/junnyu/roformer_v2_chinese_char_base) | 
| simbert | 追一科技 | [tf](https://github.com/ZhuiyiTechnology/simbert)，[torch_base](https://huggingface.co/peterchou/simbert-chinese-base/tree/main) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_simbert.py) |
| roformer-sim | 追一科技 | [tf](https://github.com/ZhuiyiTechnology/roformer-sim)，[torch](https://huggingface.co/junnyu/roformer_chinese_sim_char_base) | 
| gau-alpha | 追一科技 | [tf](https://github.com/ZhuiyiTechnology/GAU-alpha) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_GAU_alpha.py)
| wobert | 追一科技 | [tf](https://github.com/ZhuiyiTechnology/WoBERT)，[torch_base](https://huggingface.co/junnyu/wobert_chinese_base)，[torch_plus_base](https://huggingface.co/junnyu/wobert_chinese_plus_base)||
| nezha | 华为 | [tf](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow)，[torch](https://github.com/lonePatient/NeZha_Chinese_PyTorch) | 
| gpt | CDial-GPT | [torch](https://github.com/thu-coai/CDial-GPT) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_gpt__CDial-GPT-LCCC.py)
| gpt2 | 清华26亿 cmp_lm | [torch](https://github.com/TsinghuaAI/CPM-1-Generate) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_gpt2__cmp_lm_2.6b.py)
| gpt2 | 中文GPT2_ML模型 | [tf](https://github.com/imcaspar/gpt2-ml)，[torch](https://github.com/ghosthamlet/gpt2-ml-torch) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_gpt2__gpt2-ml.py)
| t5 | UER | [torch](https://huggingface.co/uer/t5-base-chinese-cluecorpussmall) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/PLM_config.md)
| mt5 | 谷歌 | [torch](https://huggingface.co/google/mt5-base) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/PLM_config.md)
| t5_pegasus | 追一科技 | [tf](https://github.com/ZhuiyiTechnology/t5-pegasus) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_t5_pegasus.py)
| bart | 复旦 | [torch](https://github.com/fastnlp/CPT) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_bart_fudanNLP.py)
| text2vec | text2vec-base-chinese | [torch](https://huggingface.co/shibing624/text2vec-base-chinese) | 


## 7. 鸣谢
- 感谢苏神实现的[bert4keras](https://github.com/bojone/bert4keras)，本实现有不少地方参考了bert4keras的源码，在此衷心感谢大佬的无私奉献; 
- 其次感谢项目[bert4pytorch](https://github.com/MuQiuJun-AI/bert4pytorch)，也是在该项目的指引下给了我用pytorch来复现bert4keras的想法和思路。
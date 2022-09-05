# bert4torch
**一款用pytorch来复现bert4keras的简洁训练框架**

[![licence](https://img.shields.io/github/license/Tongjilibo/bert4torch.svg?maxAge=3600)](https://github.com/Tongjilibo/bert4torch/blob/master/LICENSE) 
[![GitHub release](https://img.shields.io/github/release/Tongjilibo/bert4torch.svg?maxAge=3600)](https://github.com/Tongjilibo/bert4torch/releases) 
[![PyPI](https://img.shields.io/pypi/v/bert4torch?label=pypi%20package)](https://pypi.org/project/bert4torch/) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/bert4torch)](https://pypistats.org/packages/bert4torch)
[![GitHub stars](https://img.shields.io/github/stars/Tongjilibo/bert4torch?style=social)](https://github.com/Tongjilibo/bert4torch)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Tongjilibo/bert4torch/issues)

## 下载安装
安装稳定版
```shell
pip install bert4torch
```
安装最新版
```shell
pip install git+https://www.github.com/Tongjilibo/bert4torch.git
```
- **注意事项**：pip包的发布慢于git上的开发版本，git clone**注意引用路径**
- **测试用例**：`git clone https://github.com/Tongjilibo/bert4torch`，修改example中的预训练模型文件路径和数据路径即可启动脚本
- **自行训练**：针对自己的数据，修改相应的数据处理代码块
- **开发环境**：使用`torch==1.10`版本进行开发，如其他版本遇到不适配，欢迎反馈

## 功能
- **核心功能**：加载bert、roberta、albert、xlnet、nezha、bart、RoFormer、RoFormer_V2、ELECTRA、GPT、GPT2、T5、GAU-alpha等预训练权重继续进行finetune、并支持在bert基础上灵活定义自己模型
- **丰富示例**：包含[pretrain](https://github.com/Tongjilibo/bert4torch/blob/master/examples/pretrain)、[sentence_classfication](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication)、[sentence_embedding](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_embedding)、[sequence_labeling](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling)、[relation_extraction](https://github.com/Tongjilibo/bert4torch/blob/master/examples/relation_extraction)、[seq2seq](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq)等多种解决方案
- **实验验证**：已在公开数据集[实验验证](https://github.com/Tongjilibo/bert4torch/blob/master/examples/Performance.md), 使用如下[examples数据集](https://github.com/Tongjilibo/bert4torch/blob/master/examples/README.md)
- **易用trick**：集成了常见的[trick](https://github.com/Tongjilibo/bert4torch/blob/master/examples/training_trick), 即插即用
- **其他特性**：[加载transformers库模型](https://github.com/Tongjilibo/bert4torch/blob/master/examples/others/task_load_transformers_model.py)一起使用；调用方式简洁高效；有训练进度条动态展示；配合torchinfo打印参数量；默认Logger和Tensorboard简便记录训练过程；自定义fit过程，满足高阶需求

## 快速上手
- [快速上手教程](https://github.com/Tongjilibo/bert4torch/blob/master/examples/tutorials/Tutorials.md), [教程示例](https://github.com/Tongjilibo/bert4torch/blob/master/examples/tutorials), [实战示例](https://github.com/Tongjilibo/bert4torch/blob/master/examples)
- [bert4torch介绍(知乎)](https://zhuanlan.zhihu.com/p/486329434)，[bert4torch快速上手(知乎)](https://zhuanlan.zhihu.com/p/508890807), [bert4torch又双叒叕更新啦](https://zhuanlan.zhihu.com/p/560885427?)

## 版本说明
- **v0.2.1**：兼容torch<=1.7.1的torch.div无rounding_mode, 增加自定义metrics，支持断点续训，增加默认Logger和Tensorboard日志
- **v0.2.0**：兼容torch<1.9.0的缺失take_along_dim，修复bart中位置向量514的问题，修复Sptokenizer对符号不转换，打印Epoch开始的时间戳，增加parallel_apply
- **v0.1.9**：增加mixup/manifold_mixup/temporal_ensembling策略, 修复pgd策略param.grad为空的问题，修改tokenizer支持批量
- **v0.1.8**：修复原来CRF训练中loss陡增的问题，修复xlnet的token_type_ids输入显存占用大的问题
- **v0.1.7**：增加EarlyStop，CRF中自带转bool类型
- **v0.1.6**：增加transformer_xl、xlnet、t5_pegasus模型，prompt、预训练等示例，支持增加embedding输入，EMA策略，修复tokenizer和sinusoid的bug
- **v0.1.5**：增加GAU-alpha，混合梯度，梯度裁剪，单机多卡(DP、DDP)
- **v0.1.4**：增加了VAT，修复了linux下apply_embedding返回项有问题的情况
- **v0.1.3**：初始版本

## 更新：
- **2022年9月04更新**：增加默认Logger和Tensorboard日志
- **2022年8月28更新**：增加nl2sql示例, 增加自定义metrics，支持断点续训
- **2022年8月21更新**：增加W2NER和DiffCSE示例，打印Epoch开始的时间戳，增加parallel_apply, 兼容torch<=1.7.1的torch.div无rounding_mode
- **2022年8月14更新**：增加有监督句向量、关系抽取、文本生成实验指标，兼容torch<1.9.0的缺失take_along_dim，修复bart中位置向量514的问题，修复Sptokenizer对符号不转换
- **2022年7月27更新**：增加mixup/manifold_mixup/temporal_ensembling策略, 修复pgd策略param.grad为空的问题，修改tokenizer支持批量，增加uie示例
- **2022年7月16更新**：修复原来CRF训练中loss陡增的问题，修复xlnet的token_type_ids输入显存占用大的问题
- **2022年7月10更新**：增加金融中文FAQ示例，天池新闻分类top1案例，增加EarlyStop，CRF中自带转bool类型
- **2022年6月29更新**：增加ner的实验，测试crf不同初始化的效果，bert-whitening中文实验
- **2022年6月13更新**：增加seq2seq+前缀树，增加SimCSE/ESimCSE/PromptBert等无监督语义相似度的中文实验
- **2022年6月05更新**：增加PromptBert、PET、P-tuning示例，修改tokenizer对special_tokens分词错误的问题，增加t5_pegasus
- **2022年5月29更新**：transformer_xl、xlnet模型, 修改sinusoid位置向量被init_weight的bug, EMA，sohu情感分类示例
- **2022年5月17更新**：增加预训练代码，支持增加embedding输入(如词性，word粒度embedding)
- **2022年5月01更新**：增加了混合梯度，梯度裁剪，单机多卡训练(DP、DDP)
- **2022年4月25更新**：增加了VAT、GAU-alpha等示例，增加了梯度累积，自定义fit()示例
- **2022年4月15更新**：增加了ner_mrc、ner_span、roformer_v2、roformer-sim等示例
- **2022年4月05更新**：增加了GPLinker、TPlinker、SimBERT等示例
- **2022年3月29更新**：增加了CoSENT、R-Drop、UDA等示例
- **2022年3月22更新**：添加GPT、GPT2、T5模型
- **2022年3月12更新**：初版提交

## 背景
- 用pytorch复现苏神的[bert4keras](https://github.com/bojone/bert4keras)
- 初版参考了[bert4pytorch](https://github.com/MuQiuJun-AI/bert4pytorch)

## 预训练权重
- 部分权重是要加载修改的[config.json](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/PLM_config.md)

| 模型分类 |  权重来源 | 权重链接 | 备注(若有) | 
|  ----  |  ----  | ----  | ----  |
|  bert  | 谷歌原版bert | [tf](https://github.com/google-research/bert) | [转pytorch命令](https://huggingface.co/docs/transformers/converting_tensorflow_models)
|  bert  | 哈工大chinese-bert-wwm-ext | [tf/pytorch](https://github.com/ymcui/Chinese-BERT-wwm), [pytorch](https://huggingface.co/hfl/chinese-bert-wwm-ext) |
| bert | bert-base-chinese | [pytorch](https://huggingface.co/bert-base-chinese) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_bert-base-chinese.py)
| robert | 哈工大chinese-robert-wwm-ext | [tf/pytorch](https://github.com/ymcui/Chinese-BERT-wwm), [pytorch](https://huggingface.co/hfl/chinese-roberta-wwm-ext)
| albert | brightmart | [tf](https://github.com/brightmart/albert_zh), [pytorch](https://github.com/lonePatient/albert_pytorch)
| xlnet | 哈工大xlnet | [tf/pytorch](https://github.com/ymcui/Chinese-XLNet) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/PLM_config.md)
| electra | 哈工大electra | [tf](https://github.com/ymcui/Chinese-ELECTRA), [pytorch](https://huggingface.co/hfl/chinese-electra-base-discriminator)
| macbert | 哈工大macbert | [tf](https://github.com/ymcui/MacBERT), [pytorch](https://huggingface.co/hfl/chinese-macbert-base)
| roformer | 追一科技 | [tf](https://github.com/ZhuiyiTechnology/roformer), [pytorch](https://huggingface.co/junnyu/roformer_chinese_base) |  
| roformer_v2 | 追一科技 | [tf](https://github.com/ZhuiyiTechnology/roformer-v2), [pytorch](https://huggingface.co/junnyu/roformer_v2_chinese_char_base) | 
| simbert | 追一科技 | [tf](https://github.com/ZhuiyiTechnology/simbert), [pytorch](https://huggingface.co/peterchou/simbert-chinese-base/tree/main) | 
| roformer-sim | 追一科技 | [ft](https://github.com/ZhuiyiTechnology/roformer-sim), [pytorch](https://huggingface.co/junnyu/roformer_chinese_sim_char_base) | 
| gau-alpha | 追一科技 | [tf](https://github.com/ZhuiyiTechnology/GAU-alpha) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_GAU_alpha.py)
| nezha | 华为 | [tf](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow), [pytorch](https://github.com/lonePatient/NeZha_Chinese_PyTorch) | 
| gpt | CDial-GPT | [pytorch](https://github.com/thu-coai/CDial-GPT) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_gpt__CDial-GPT-LCCC.py)
| gpt2 | 清华26亿 cmp_lm | [pytorch](https://github.com/TsinghuaAI/CPM-1-Generate) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_gpt2__cmp_lm_2.6b.py)
| gpt2 | 中文GPT2_ML模型 | [tf](https://github.com/imcaspar/gpt2-ml), [pytorch](https://github.com/ghosthamlet/gpt2-ml-torch) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_gpt2__gpt2-ml.py)
| t5 | UER | [pytorch](https://huggingface.co/uer/t5-base-chinese-cluecorpussmall) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/PLM_config.md)
| mt5 | 谷歌 | [pytorch](https://huggingface.co/google/mt5-base) | [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/PLM_config.md)
| t5_pegasus | 追一科技 | [tf](https://github.com/ZhuiyiTechnology/t5-pegasus) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_t5_pegasus.py)
| bart | 复旦 | [pytorch](https://github.com/fastnlp/CPT) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/convert_script/convert_bart_fudanNLP.py)

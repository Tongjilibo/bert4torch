# bert4torch
**一款用pytorch来复现bert4keras的简洁训练框架**

## 下载安装
- `pip install bert4torch`
- pip包的发布慢于git上的开发版本，如需要使用最新开发的代码，可直接git clone最新代码，**注意引用路径**
- 跑测试用例：`git clone https://github.com/Tongjilibo/bert4torch`，修改example中的预训练模型文件路径和数据路径即可启动脚本，examples中用到的数据文件后续会放链接
- 自行训练：针对自己的数据，修改相应的数据处理代码块

## 快速上手
- [快速上手教程](https://github.com/Tongjilibo/bert4torch/blob/master/Tutorials.md)
- [实战示例](https://github.com/Tongjilibo/bert4torch/blob/master/examples)
- [bert4torch介绍(知乎图文版)](https://zhuanlan.zhihu.com/p/486329434)

## 更新：
- **2022年4月25更新**：增加了VAT、GAU-alpha等示例，增加了梯度累积，自定义fit()示例
- **2022年4月15更新**：增加了ner_mrc、ner_span、roformer_v2、roformer-sim等示例
- **2022年4月05更新**：增加了GPLinker、TPlinker、SimBERT等示例
- **2022年3月29更新**：增加了CoSENT、R-Drop、UDA等示例
- **2022年3月22更新**：添加GPT、GPT2、T5模型
- **2022年3月12更新**：初版提交

## 背景
- 用pytorch复现苏神的[bert4keras](https://github.com/bojone/bert4keras)
- 初版参考了[bert4pytorch](https://github.com/MuQiuJun-AI/bert4pytorch)

## 功能
- **核心功能**：加载预训练权重继续进行finetune、并支持在bert基础上灵活定义自己模型
- **丰富示例**：包含[sentence_classfication](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication)、[sentence_embedding](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_embedding)、[sequence_labeling](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling)、[relation_extraction](https://github.com/Tongjilibo/bert4torch/blob/master/examples/relation_extraction)、[seq2seq](https://github.com/Tongjilibo/bert4torch/blob/master/examples/seq2seq)等多种解决方案
- **其他特性**：可[加载transformers库模型](https://github.com/Tongjilibo/bert4torch/blob/master/examples/others/task_load_transformers_model.py)一起使用；调用方式和bert4keras基本一致，简洁高效；实现基于keras的训练进度条动态展示；兼容torchinfo，实现打印各层参数量功能；自定义fit过程，满足高阶需求

### 现在已经实现
- 加载bert、roberta、albert、nezha、bart、RoFormer、RoFormer_V2、ELECTRA、GPT、GPT2、T5、GAU-alpha模型进行fintune
- 对抗训练（FGM, PGD, 梯度惩罚）

### 未来将实现
- 前沿的各类模型idea实现，如苏神科学空间网站的诸多idea
**
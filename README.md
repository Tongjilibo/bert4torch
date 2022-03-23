# bert4torch

## 下载安装
- `pip install bert4torch`
- 跑测试用例：`git clone https://github.com/Tongjilibo/bert4torch`，修改example中的预训练模型文件路径和数据路径即可启动脚本，examples中用到的数据文件后续会放链接
- 自行训练：针对自己的数据，修改相应的数据处理代码块

## 更新：

- **2022年3月22更新**：添加GPT、GPT2、T5模型
- **2022年3月12更新**：初版提交

## 背景
- [bert4torch介绍(知乎图文版)](https://zhuanlan.zhihu.com/p/486329434)
- 用pytorch复现苏神的[bert4keras](https://github.com/bojone/bert4keras)
- 初版参考了[bert4pytorch](https://github.com/MuQiuJun-AI/bert4pytorch)

## 功能
- 加载预训练权重继续进行finetune
- 在bert基础上灵活定义自己模型
- 调用方式和bert4keras基本一致，简洁高效
- 实现基于keras的训练进度条动态展示
- 兼容torchinfo，实现打印各层参数量功能
- 集成多个example，可以作为自己的训练框架，方便在同一个数据集上尝试多种解决方案

### 现在已经实现
- 加载bert、roberta、albert、nezha、bart、RoFormer、ELECTRA、GPT、GPT2、T5模型进行fintune
- 对抗训练（FGM, PGD, 梯度惩罚）

### 未来将实现
- Transformer-XL、XLnet等其他网络架构
- 前沿的各类模型idea实现，如苏神科学空间网站的诸多idea

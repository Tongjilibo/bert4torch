# bert4torch

## 更新：

- **2022年3月12更新**：初版提交
  
  

## 背景
- 用pytorch复现苏神的[bert4keras](https://github.com/bojone/bert4keras)
- 初版参考了[bert4pytorch](https://github.com/MuQiuJun-AI/bert4pytorch)

## 功能
- 加载预训练权重继续进行finetune
- 在bert基础上灵活定义自己模型
- 调用方式和bert4keras基本一致，简洁高效
- 集成多个example，可以作为自己的训练框架，方便在同一个数据集上尝试多种解决方案

### 现在已经实现

- 加载bert、roberta、albert、nezha、bart、RoFormer、ELECTRA模型进行fintune
- 对抗训练

### 未来将实现
- Transformer-XL、XLnet、T5、GPT等其他网络架构
- 前沿的各类模型idea实现

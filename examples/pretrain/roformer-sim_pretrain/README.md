# 使用说明

## 思路
1. stage1: 训练方式和simbert类似+[MASK预测]
2. stage2: 把simbert的相似度蒸馏到roformer-sim上
3. supervised: 监督训练代码supervised部分

## reference
- [roformer-sim](https://github.com/ZhuiyiTechnology/roformer-sim)
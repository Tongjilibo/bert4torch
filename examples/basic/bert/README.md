# 转换huggingface上bert-base-chinese权重
## tf版本：chinese_L-12_H-768_A-12
- [权重链接](https://github.com/google-research/bert)
- 需使用transformer自带命令转换tf权重, [转换命令](https://huggingface.co/docs/transformers/v4.28.1/en/converting_tensorflow_models)

## torch版本：bert-base-chinese
- [权重链接](https://huggingface.co/bert-base-chinese)
- 里面用的都是Laynorm.gamma和Laynorm.beta来保存权重和偏置

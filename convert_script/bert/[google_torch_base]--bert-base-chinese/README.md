# 转换huggingface上bert-base-chinese权重
- 权重链接：https://huggingface.co/bert-base-chinese
- 由于key和框架的key没有完全对齐，主要里面用的都是Laynorm.gamma和Laynorm.beta来保存权重和偏置

- 也可使用transformer自带命令转换tf权重https://github.com/google-research/bert
- 转换命令https://huggingface.co/docs/transformers/converting_tensorflow_models
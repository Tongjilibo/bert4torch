# ONNX+TensorRT
本文以情感二分类为例，使用ONNX+TensorRT来部署

## 1. pytorch权重转onnx
1. 首先需要运行[情感分类任务](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentiment_classification.py)，并保存pytorch的权重

2. 使用了pytorch自带的`torch.onnx.export()`来转换，转换脚本见[ONNX转换bert权重](https://github.com/Tongjilibo/bert4torch/blob/master/examples/serving/task_bert_cls_onnx.py)

## 2. tensorrt环境安装
参考[TensorRT 8.2.1.8 安装笔记(超全超详细)|Docker 快速搭建 TensorRT 环境](https://zhuanlan.zhihu.com/p/446477459)中的半自动安装流程，可直接阅读源文档

1. 官网下载对应版本的镜像(个人根据具体cuda版本选择) 
```shell
docker pull nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
```
2. 运行镜像/创建容器
```shell
docker run -it --name trt_test --gpus all -v /home/tensorrt:/tensorrt nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04 /bin/bash
```
3. [下载TensorRT包](https://developer.nvidia.com/zh-cn/tensorrt)，这一步需要注册账号，我下载的是`TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz`
4. 回到容器安装TensorRT(cd到容器内的tensorrt路径下解压刚才下载的tar包)
```shell
tar -zxvf  TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz
```
5. 添加环境变量
```
# 安装vim
apt-get update
apt-get install vim

vim ~/.bashrc
export LD_LIBRARY_PATH=/tensorrt/TensorRT-8.4.1.5/lib:$LD_LIBRARY_PATH
source ~/.bashrc
```
6. 安装 python(安装之后输入python查看安装的版本，下一步要用到)
```shell
apt-get install -y --no-install-recommends \
python3 \
python3-pip \
python3-dev \
python3-wheel &&\
cd /usr/local/bin &&\
ln -s /usr/bin/python3 python &&\
ln -s /usr/bin/pip3 pip;
```
7. pip安装对应的TensorRT库
注意一定要使用pip本地安装tar附带的对应python版本的whl包
```shell
cd TensorRT-8.4.1.5/python/
pip3 install tensorrt-8.2.1.8-cp36-none-linux_x86_64.whl
```
8. 测试TensorRT的python接口
```python
import tensorrt
print(tensorrt.__version__)
```

## 3. onnx转trt权重
- 转换命令
```shell
./trtexec --onnx=/tensorrt/bert_cls.onnx --saveEngine=/tensorrt/bert_cls.trt --workspace=10000 --minShapes=input_ids:1x1,segment_ids:1x1 --optShapes=input_ids:20x512,segment_ids:20x512 --maxShapes=input_ids:20x512,segment_ids:20x512 --device=0
```

## 4. tensorrt加载模型推理
- 参考文档：[基于 TensorRT 实现 Bert 预训练模型推理加速(超详细-附核心代码-避坑指南)](https://zhuanlan.zhihu.com/p/446477075)
- 推理代码
```python
import numpy as np
from bert4torch.tokenizers import Tokenizer
import tensorrt as trt
import common

"""
a、获取 engine，建立上下文
"""
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine

engine_model_path = "bert_cls.trt"
# Build a TensorRT engine.
engine = get_engine(engine_model_path)
# Contexts are used to perform inference.
context = engine.create_execution_context()


"""
b、从engine中获取inputs, outputs, bindings, stream 的格式以及分配缓存
"""
def to_numpy(tensor):
    return np.array(tensor, np.int32)

dict_path = '/tensorrt/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)
input_ids, segment_ids = tokenizer.encode('我的心情很差')

tokens_id = to_numpy([input_ids])
# print(tokens_id)
segment_ids = to_numpy([segment_ids])

context.active_optimization_profile = 0
origin_inputshape = context.get_binding_shape(0)                # (1,-1) 
origin_inputshape[0],origin_inputshape[1] = tokens_id.shape     # (batch_size, max_sequence_length)
context.set_binding_shape(0, (origin_inputshape))               
context.set_binding_shape(1, (origin_inputshape))

"""
c、输入数据填充
"""
inputs, outputs, bindings, stream = common.allocate_buffers_v2(engine, context)
inputs[0].host = tokens_id
inputs[1].host = segment_ids
# print(tokens_id, tokens_id.dtype)
# print(segment_ids, segment_ids.dtype)

"""
d、tensorrt推理
"""
trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
preds = np.argmax(trt_outputs, axis=1)
print("====preds====:",preds)
```

- 所需[common.py](https://github.com/NVIDIA/TensorRT/blob/96e23978cd6e4a8fe869696d3d8ec2b47120629b/samples/python/common.py)
- 运行结果
```shell
Reading engine from file bert_cls.trt
inference.py:39: DeprecationWarning: Use set_optimization_profile_async instead.
  context.active_optimization_profile = 0
====preds====: [0]
```

# 5. 速度比较
- 测试句长=200
- 测试方式: 跑500个循环求均值

| 方案 | cpu(ms) | gpu(ms) |
|----|----|----|
|pytorch|144|29|
|onnx|66||
|onnx+tensorrt||101|
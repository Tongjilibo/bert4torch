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
docker run -it --name trt_test --gpus all -v /home/tensorrt:/tensorrt nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04 /bin/bash
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
./trtexec --onnx=/tensorrt/bert_cls.onnx --saveEngine=/tensorrt/bert_cls.trt --minShapes=input_ids:1x512,segment_ids:1x512 --optShapes=input_ids:1x512,segment_ids:1x512 --maxShapes=input_ids:20x512,segment_ids:20x512 --device=0
```
- 注意项：1）测试中如果把batch_size维度和seq_len维度都设置成动态速度会很慢(100ms+)，因此这里只保留动态的batchsize维度，seq_len都padding到512；2）[参考资料](https://github.com/NVIDIA/TENSORRT/issues/976)

## 4. tensorrt加载模型推理
- 参考文档：[基于 TensorRT 实现 Bert 预训练模型推理加速(超详细-附核心代码-避坑指南)](https://zhuanlan.zhihu.com/p/446477075)
- 推理代码
```python
import numpy as np
from bert4torch.tokenizers import Tokenizer
import tensorrt as trt
import common
import time
import numpy as np
from tqdm import tqdm

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
    for i, item in enumerate(tensor):
        tensor[i] = item + [0] * (512-len(item))
    return np.array(tensor, np.int32)

dict_path = '/tensorrt/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)
sentences = ['你在干嘛呢？这几天外面的天气真不错啊，万里无云，阳光明媚的，我的心情也特别的好，我特别想出门去转转呢。你在干嘛呢？这几天外面的天气真不错啊，万里无云，阳光明媚的，我的心情也特别的好，我特别想出门去转转呢。你在干嘛呢？这几天外面的天气真不错啊，万里无云，阳光明媚的，我的心情也特别的好，我特别想出门去转转呢。你在干嘛呢？这几天外面的天气真不错啊，万里无云，阳光明媚的，我的心情也特别的好，我特别想出门。']
input_ids, segment_ids = tokenizer.encode(sentences)
tokens_id = to_numpy(input_ids)
segment_ids = to_numpy(segment_ids)

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

"""
d、tensorrt推理
"""
trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
preds = np.argmax(trt_outputs, axis=1)
print("====preds====:",preds)

"""
e、测试耗时(不含构造数据)
"""
steps = 100
start = time.time()
for i in tqdm(range(steps)):
    common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    preds = np.argmax(trt_outputs, axis=1)
print('onnx+tensorrt（不含构造数据）: ',  (time.time()-start)*1000/steps, ' ms')

"""
f、测试耗时(含构造数据)
"""
steps = 100
start = time.time()
for i in tqdm(range(steps)):
    input_ids, segment_ids = tokenizer.encode(sentences)
    tokens_id = to_numpy(input_ids)
    segment_ids = to_numpy(segment_ids)
    inputs, outputs, bindings, stream = common.allocate_buffers_v2(engine, context)
    inputs[0].host = tokens_id
    inputs[1].host = segment_ids

    common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    preds = np.argmax(trt_outputs, axis=1)
print('onnx+tensorrt（含构造数据）: ',  (time.time()-start)*1000/steps, ' ms')
```

- 所需[common.py](https://github.com/NVIDIA/TensorRT/blob/96e23978cd6e4a8fe869696d3d8ec2b47120629b/samples/python/common.py)
- 运行结果
```shell
Reading engine from file bert_cls.trt
onnx_tensorrt.py:44: DeprecationWarning: Use set_optimization_profile_async instead.
  context.active_optimization_profile = 0
====preds====: [1]
100%|██████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 78.87it/s]
onnx+tensorrt（不含构造数据）:  12.6955246925354  ms
100%|██████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 71.50it/s]
onnx+tensorrt（含构造数据）:  13.990252017974854  ms
```

# 5. 速度比较
- 测试方式: btz=1, seq_len=202（对于tensorrt测试了seq_len=202和512）, iterations=100

| 方案 | cpu | gpu |
|----|----|----|
|pytorch|144ms|29ms|
|onnx|66ms|——|
|onnx+tensorrt|——|7ms (len=202), 12ms (len=512)|

# 6. 实验文件
- [文件树](https://pan.baidu.com/s/1vX3yK7BWQScnK_5Zb-pAkQ?pwd=rhq9)
```shell
tensorrt
├─common.py
├─onnx_tensorrt.py
├─bert_cls.onnx
├─bert_cls.trt
└─TensorRT-8.4.1.5
```
- docker镜像: 1)可按上述方式自行构建，2)直接pull笔者上传的镜像
```shell
docker pull tongjilibo/tensorrt:11.3.0-cudnn8-devel-ubuntu20.04-tensorrt8.4.1.5

docker run -it --name trt_torch --gpus all -v /home/libo/tensorrt:/tensorrt tongjilibo/tensorrt:11.3.0-cudnn8-devel-ubuntu20.04-tensorrt8.4.1.5 /bin/bash
```
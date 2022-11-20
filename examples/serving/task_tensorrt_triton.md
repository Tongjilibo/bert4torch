# TensorRT+Triton
- 本文以情感二分类为例，使用TensorRT+Triton来部署
- **注意**：注意版本，有些问题可能是版本导致，如转trt错误可能和tensorrt版本相关

## 1. pytorch权重转onnx
1. 首先需要运行[情感分类任务](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sentence_classfication/task_sentiment_classification.py)，并保存pytorch的权重
2. 使用了pytorch自带的`torch.onnx.export()`来转换，转换脚本见[ONNX转换bert权重](https://github.com/Tongjilibo/bert4torch/blob/master/examples/serving/task_bert_cls_onnx.py)

## 2. onnx转tensorrt权重
```shell
# 拉取tensorrt镜像
docker pull nvcr.io/nvidia/tensorrt:22.07-py3
# 启动tensorrt镜像
docker run --gpus all -it --rm -v /home/libo/Github/triton/model_repository:/models nvcr.io/nvidia/tensorrt:22.07-py3
# 使用trtexec来把onnx转成trt格式
trtexec --onnx=bert_cls.onnx --saveEngine=./model.plan --minShapes=input_ids:1x512,segment_ids:1x512 --optShapes=input_ids:1x512,segment_ids:1x512 --maxShapes=input_ids:20x512,segment_ids:20x512 --device=0
```

## 3. 模型文件和配置文件准备
- 文件目录
```shell
model_repository
└─sentence_classification
    └─1
        └─model.plan
    └─config.pbtxt
```

- config.pbtxt
```text
name: "sentence_classification"
platform: "tensorrt_plan"
max_batch_size: 8
version_policy: { latest { num_versions: 1 }}
input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: "segment_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]
```


## 4. 启动triton服务端
```shell
docker pull nvcr.io/nvidia/tritonserver:22.07-py3
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /home/libo/Github/triton/model_repository:/models nvcr.io/nvidia/tritonserver:22.07-py3 tritonserver --model-repository=/models
```

## 5. client调用
- 两种方式，一种直接request测试接口调用，一种使用trition client
```python
import requests

if __name__ == "__main__":
    request_data = {
    "inputs": [{
        "name": "input_ids",
        "shape": [1, 512],
        "datatype": "INT32",
        "data": [list(range(512))]
    },
    {
        "name": "segment_ids",
        "shape": [1, 512],
        "datatype": "INT32",
        "data": [list(range(512))]
    }
    ],
    "outputs": [{"name": "output"}]
}
    res = requests.post(url="http://localhost:8000/v2/models/sentence_classification/versions/1/infer",json=request_data).json()
    print(res)
# {'model_name': 'sentence_classification', 'model_version': '1', 'outputs': [{'name': 'output', 'datatype': 'FP32', 'shape': [1, 2], 'data': [0.703898549079895, 0.29610151052474976]}]}
```
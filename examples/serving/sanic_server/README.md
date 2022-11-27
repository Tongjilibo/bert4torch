# 说明
本项目是利用sanic来部署bert模型，说明如下：

1. 这里用的是onnx来推理，实际操作中可以用pytorch源代码、tensorrt、triton等多种部署方式
2. 本项目是异步的，也支持k8s多实例部署
3. `python server.py`启动服务端，执行`client.py`来发送请求获取模型推理结果
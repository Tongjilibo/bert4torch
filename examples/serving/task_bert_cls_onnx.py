# 转onnx并推理得到结果
# 默认export只会转forward结果，因此需要到处时候要把predict改成forward来完成转换

import numpy as np
import torch.onnx
import os
import numpy as np
import torch
import torch.nn as nn
from bert4torch.snippets import get_pool_emb
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
import time
from tqdm import tqdm


config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'


tokenizer = Tokenizer(dict_path, do_lower_case=True)

class Model(BaseModel):
    def __init__(self, pool_method='cls') -> None:
        super().__init__()
        self.pool_method = pool_method
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.bert.configs['hidden_size'], 2)

    def forward(self, token_ids, segment_ids):
        self.eval()
        with torch.no_grad():
            hidden_states, pooling = self.bert([token_ids, segment_ids])
            pooled_output = get_pool_emb(hidden_states, pooling, token_ids.gt(0).long(), self.pool_method)
            output = self.dropout(pooled_output)
            output = nn.Softmax(dim=-1)(self.dense(output))
            return output

torch_model = Model()
torch_model.load_weights('E:/Github/bert4torch/examples/sentence_classfication/best_cls_model.pt')

# 模型输入
sentence = '你在干嘛呢？这几天外面的天气真不错啊，万里无云，阳光明媚的，我的心情也特别的好，我特别想出门去转转呢。你在干嘛呢？这几天外面的天气真不错啊，万里无云，阳光明媚的，我的心情也特别的好，我特别想出门去转转呢。你在干嘛呢？这几天外面的天气真不错啊，万里无云，阳光明媚的，我的心情也特别的好，我特别想出门去转转呢。你在干嘛呢？这几天外面的天气真不错啊，万里无云，阳光明媚的，我的心情也特别的好，我特别想出门。'
input_ids, segment_ids = tokenizer.encode(sentence)
input_ids = torch.tensor([input_ids])
segment_ids = torch.tensor([segment_ids])
torch_out = torch_model(input_ids, segment_ids)

# 转onnx
if not os.path.exists("bert_cls.onnx"):
    torch.onnx.export(torch_model,               # model being run
                    (input_ids, segment_ids),    # model input (or a tuple for multiple inputs)
                    "bert_cls.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input_ids', 'segment_ids'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input_ids' : {0 : 'batch_size', 1: 'seq_len'},    # variable length axes
                                  'segment_ids' : {0 : 'batch_size', 1: 'seq_len'},    # variable length axes
                                    'output' : {0 : 'batch_size', 1: 'seq_len'}})

# 模型验证
import onnx
onnx_model = onnx.load("bert_cls.onnx")
onnx.checker.check_model(onnx_model)

# 模型推理
import onnxruntime
ort_session = onnxruntime.InferenceSession("bert_cls.onnx")
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# 计算ONNX输出
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids),
             ort_session.get_inputs()[1].name: to_numpy(segment_ids)}
ort_outs = ort_session.run(None, ort_inputs)

# 比较两者数据
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
print('torch_out: ', torch_out[0])
print('ort_outs: ', ort_outs[0][0])

# =====================================测试两者的速度(不含tokenizer构造数据的耗时)
# torch cpu
steps = 100
start = time.time()
for i in tqdm(range(steps)):
    torch_model(input_ids, segment_ids)
print('pytorch cpu: ',  (time.time()-start)*1000/steps, ' ms')

# torch gpu
torch_model = torch_model.to('cuda')
input_ids = input_ids.to('cuda')
segment_ids = segment_ids.to('cuda')
start = time.time()
for i in tqdm(range(steps)):
    torch_model(input_ids, segment_ids)
print('pytorch gpu: ',  (time.time()-start)*1000/steps, ' ms')

# onnx cpu
start = time.time()
for i in tqdm(range(steps)):
    ort_session.run(None, ort_inputs)
print('onnx_runtime cpu: ',  (time.time()-start)*1000/steps, ' ms')

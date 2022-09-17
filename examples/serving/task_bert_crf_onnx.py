# 转onnx并推理得到结果
# 默认export只会转forward结果，因此需要到处时候要把predict改成forward来完成转换

import numpy as np
import torch.onnx
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from bert4torch.layers import CRF
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from tqdm import tqdm

config_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/pytorch_model.bin'
dict_path = 'F:/Projects/pretrain_ckpt/bert/[google_tf_base]--chinese_L-12_H-768_A-12/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = Tokenizer(dict_path, do_lower_case=True)
categories = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']

# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0)
        self.fc = nn.Linear(768, len(categories))  # 包含首尾
        self.crf = CRF(len(categories))

    def forward(self, token_ids):
        self.eval()
        with torch.no_grad():
            sequence_output = self.bert([token_ids])  # [btz, seq_len, hdsz]
            emission_score = self.fc(sequence_output)  # [btz, seq_len, tag_size]
            attention_mask = token_ids.gt(0).long()
            best_path = self.crf.decode(emission_score, attention_mask)  # [btz, seq_len]
        return best_path
torch_model = Model().to(device)
torch_model.load_weights('E:/Github/bert4torch/examples/sequence_labeling/best_crf_model.pt')

# 模型输入
input_ids = tokenizer.encode('我想去北京天安门转一转')[0]
x = torch.tensor([input_ids], device=device)
torch_out = torch_model(x)

# Export the model
if not os.path.exists("bert_crf.onnx"):
    torch.onnx.export(torch_model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "bert_crf.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size', 1: 'seq_len'},    # variable length axes
                                    'output' : {0 : 'batch_size', 1: 'seq_len'}})

# 模型验证
import onnx
onnx_model = onnx.load("bert_crf.onnx")
onnx.checker.check_model(onnx_model)

# 模型推理
import onnxruntime
ort_session = onnxruntime.InferenceSession("bert_crf.onnx")
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
print('torch_out: ', torch_out[0])
print('ort_outs: ', ort_outs[0][0])
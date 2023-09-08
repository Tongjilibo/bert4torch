import torch
from torch4keras.trainer import Trainer
from bert4torch.models import BaseModel
import copy


class DPOTrainer(Trainer):
    '''使用dpo算法进行人类偏好对齐

    :param model: 待训练模型
    :param ref_model: 参考模型
    '''
    def __init__(self, model:BaseModel, ref_model:BaseModel=None):
        super().__init__()
        self.model = model
        self.model.print_trainable_parameters()
        self.ref_model = ref_model or copy.deepcopy(self.model)
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.print_trainable_parameters()

    def _forward(self, *inputs, **input_kwargs):
        '''修改父类的_forward来获取输出'''
        policy_logits = self._argparse_forward(self.model, *inputs, **input_kwargs).to(torch.float32)
        self.ref_model.eval()
        with torch.no_grad():
            reference_logits = self._argparse_forward(self.ref_model, *inputs, **input_kwargs).to(torch.float32)
        
        return policy_logits, reference_logits

    def unwrap_model(self):
        '''返回nn.Module模块
        '''
        return self.model
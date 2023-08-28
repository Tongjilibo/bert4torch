'''修改trl包的DPOTrainer, 用于支持bert4torch框架
'''
import torch
from torch import nn
from torch4keras.trainer import Trainer
from torch4keras.snippets import log_warn
from bert4torch.models import BaseModel
try:
    from trl.trainer import DPOTrainer
except:
    DPOTrainer = object


class DPOTrainerTrl(DPOTrainer, Trainer):
    def __init__(self, *args, **kwargs):
        if DPOTrainer == object:
            raise ValueError('Please install trl by running `pip install trl`')
        Trainer.__init__(self)
        DPOTrainer.__init__(self, *args, **kwargs)
'''修改trl包的PPOTrainer, 正在修改中
'''
from torch import nn
from torch4keras.trainer import Trainer
from bert4torch.generation import SeqGeneration

try:
    from trl.trainer import PPOTrainer
except:
    PPOTrainer = object


class PPOTrainerTrl(Trainer, PPOTrainer):
    def __init__(self, *args, **kwargs):
        if PPOTrainer == object:
            raise ValueError('Please install trl by running `pip install trl`')
        Trainer.__init__(self)
        PPOTrainer.__init__(self, *args, **kwargs)


from torch import nn
from bert4torch.generation import SeqGeneration
try:
    from torch4keras.trainer import Trainer
    from trl.trainer import PPOTrainer
except:
    PPOTrainer = object


class PPOTrainerTrl(Trainer, PPOTrainer):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        Trainer.__init__(self, *args, **kwargs)

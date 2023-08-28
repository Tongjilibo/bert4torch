'''在bert4torch中使用transfromers包的Trainer
'''
from torch4keras.trainer import Trainer
try:
    from transformers import Trainer as HfTrainer 
except:
    HfTrainer = object


class TransformersTrainer(HfTrainer, Trainer):
    def __init__(self, *args, **kwargs):
        if HfTrainer == object:
            raise ValueError('Please install transformers by running `pip install transformers`')
        Trainer.__init__(self)
        HfTrainer.__init__(self, *args, **kwargs)

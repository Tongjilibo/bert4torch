'''Trainer
通用的Trainer定义在torch4keras中，这里定义NLP中独有的Trainer
'''

try:
    from torch4keras.trainer import *  # torch4keras>=0.1.3
except:
    pass
from bert4torch.trainer.ppo_trainer_trl import *
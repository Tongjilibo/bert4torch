'''Trainer
通用的Trainer定义在torch4keras中，这里定义NLP中独有的Trainer
'''

from torch4keras.trainer import *  # torch4keras>=0.1.2.post2
from bert4torch.trainer.ppo_trainer import PPOTrainer
from bert4torch.trainer.dpo_trainer import DPOTrainer